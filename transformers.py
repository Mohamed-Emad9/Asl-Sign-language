import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm


DATASET_PATH = r"C:\Users\afsao\Desktop\final_data-20251227T050411Z-1-001\final_data"

BATCH_SIZE = 32
NUM_WORKERS = 2  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Transforms & Data Setup
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
    transforms.Grayscale(num_output_channels=3), # ViT needs 3 channels
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.05)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. Early Stopping Class
# ==========================================
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='best_vit_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ==========================================
# 4. Main Execution Function
# ==========================================
def main():
    print(f"Using Device: {DEVICE}")
    
    # --- Load Data ---
    train_dir = os.path.join(DATASET_PATH, "train")
    val_dir = os.path.join(DATASET_PATH, "val")
    test_dir = os.path.join(DATASET_PATH, "test")

    if not os.path.exists(train_dir):
        print(f"❌ Error: Path not found: {train_dir}")
        return

    print("Loading Datasets...")
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(root=val_dir, transform=val_transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=val_transform)

    print(f"Classes: {train_data.class_to_idx}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- Build Model ---
    print("Building ViT Model...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(train_data.classes),
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    # Replace Classifier
    model.classifier = nn.Sequential(
        nn.Linear(model.config.hidden_size, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, len(train_data.classes))
    ).to(DEVICE)

    # Freeze Backbone initially
    for param in model.vit.parameters():
        param.requires_grad = False

    # Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=5, verbose=True, path='best_vit_model.pth')

    # ==========================================
    # Phase 1: Train Classifier Only
    # ==========================================
    print("\n=== Phase 1: Training Classifier ===")
    num_epochs = 20
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(torch.load('best_vit_model.pth'))
            break

    # ==========================================
    # Phase 2: Fine-tuning Backbone
    # ==========================================
    print("\n=== Phase 2: Fine-Tuning Backbone ===")
    
    # Reload best model from Phase 1
    model.load_state_dict(torch.load('best_vit_model.pth'))

    # Unfreeze last 4 layers
    for param in model.vit.parameters():
        param.requires_grad = False
    for param in model.vit.encoder.layer[-4:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters(): # Keep classifier trainable
        param.requires_grad = True

    # Update Optimizer with different learning rates
    optimizer = torch.optim.Adam([
        {'params': model.classifier.parameters(), 'lr': 1e-4},
        {'params': model.vit.encoder.layer[-4:].parameters(), 'lr': 2e-5}
    ])

    early_stopping = EarlyStopping(patience=3, verbose=True, path='best_vit_finetuned.pth')

    fine_tune_epochs = 10
    for epoch in range(fine_tune_epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f"FineTune Epoch {epoch+1}/{fine_tune_epochs}")
        
        for images, labels in train_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            loss = criterion(model(pixel_values=images).logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(pixel_values=images).logits
                val_loss += criterion(outputs, labels).item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        print(f"FineTune Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            model.load_state_dict(torch.load('best_vit_finetuned.pth'))
            break

    # ==========================================
    # Final Testing
    # ==========================================
    print("\n=== Final Testing ===")
    if os.path.exists('best_vit_finetuned.pth'):
        model.load_state_dict(torch.load('best_vit_finetuned.pth'))
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=images).logits
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Total Test Images: {total}")
    print(f"Correct: {correct}")
    print(f"Wrong: {total - correct}")
    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()