from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
import torch

# ============================
# Configuration
# ============================
IMAGE_SIZE = 224
MEAN = [0.5, 0.5, 0.5]  # RGB
STD = [0.5, 0.5, 0.5]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# Transforms
# ============================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# ============================
# Preprocessing function
# ============================
def preprocess_image(image_source, mode='val'):
    """
    Accepts: File Path (str) OR PIL Image object.
    Returns: Preprocessed torch.Tensor ready for model input.
    """
    # Load image
    if isinstance(image_source, str):
        img = Image.open(image_source).convert('RGB')
    else:
        img = image_source.convert('RGB')

    # Auto-Inversion: إذا الخلفية فاتحة نقلبها
    img_np = np.array(img)
    if img_np.mean() > 127:
        img = ImageOps.invert(img)

    # Apply transforms
    if mode == 'train':
        img = train_transform(img)
    else:
        img = val_transform(img)

    return img
