import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import numpy as np
import os
import pyttsx3
import threading

# ==========================================
# 1. Configuration & Constants
# ==========================================
st.set_page_config(page_title="Silent Voice",page_icon= "🤟", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- UPDATE PATHS HERE ---
LETTERS_MODEL_PATH = r"D:\ASL_models\final_best_model.pth"
WORDS_MODEL_PATH = r"D:\ASL_models\best_vit_model.pth"
WORDS_DATA_FOLDER = r"D:\final_data\train"

# ==========================================
# 2. Session State (Crucial for Streamlit UI)
# ==========================================
# Store the last predicted result so it doesn't disappear
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_confidence' not in st.session_state:
    st.session_state.last_confidence = 0.0
if 'sentence' not in st.session_state:
    st.session_state.sentence = ""

# ==========================================
# 3. Model Architectures
# ==========================================
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(SignLanguageCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==========================================
# 4. Helpers (TTS & Loading)
# ==========================================
def speak_text(text):
    """Function to run TTS in a separate thread to prevent UI freezing"""
    def _run():
        try:
            engine = pyttsx3.init()
            # Optional: Adjust rate/volume
            engine.setProperty('rate', 150) 
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    # Threading is needed because pyttsx3 blocks execution
    t = threading.Thread(target=_run)
    t.start()

@st.cache_resource
def load_letters_engine():
    try:
        model = SignLanguageCNN(num_classes=29)
        model.load_state_dict(torch.load(LETTERS_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading Letters Model: {e}")
        return None
@st.cache_resource
def load_words_engine(num_classes):
    try:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        model.classifier = nn.Sequential(
            nn.Linear(model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        model.load_state_dict(torch.load(WORDS_MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading Words Model: {e}")
        return None

def get_word_classes():
    if os.path.exists(WORDS_DATA_FOLDER):
        return sorted([d for d in os.listdir(WORDS_DATA_FOLDER) if os.path.isdir(os.path.join(WORDS_DATA_FOLDER, d))])
    return ["Unknown"]

LETTERS_MAP = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 
    26: 'del', 27: 'nothing', 28: 'space'
}

# ==========================================
# 5. GUI Logic
# ==========================================

st.sidebar.title("🎛 Control Panel")
mode = st.sidebar.selectbox("Select Mode:", ["Letters (Alphabet)", "Words (Expressions)"])

# --- Letters Sentence Controls ---
if mode == "Letters (Alphabet)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Sentence Builder")
    # Show current sentence in sidebar
    st.sidebar.text_area("Current Text:", value=st.session_state.sentence, height=100, disabled=True)
    
    if st.sidebar.button("🔊 Speak Sentence"):
        if st.session_state.sentence:
            speak_text(st.session_state.sentence)
        else:
            st.sidebar.warning("Sentence is empty!")
            
    if st.sidebar.button("❌ Clear Sentence"):
        st.session_state.sentence = ""
        st.rerun()

st.title(f"🤟 ASL Detector: {mode}")
st.markdown(
    """
    <p style="
        font-size: 15px;
        color: #9aa0a6;
        letter-spacing: 1px;
        margin-top: -8px;
        margin-bottom: 20px;
    ">
        "Because Silence Deserves to Be Heard"
    </p>
    """,
    unsafe_allow_html=True
)


# --- Image Input ---
img_file = st.file_uploader(
    "📤 Upload Image",
    type=['jpg', 'png', 'jpeg']
)
if img_file is None:
    st.info("⬆️ Please upload an image to start detection")
if img_file:
    col1, col2 = st.columns(2)
    original_pil = Image.open(img_file).convert("RGB")
    
    with col1:
        st.image(original_pil, caption="Input", width=400)

    # --- Analysis Button ---
    # We use a callback or just standard flow. Here standard flow updates session state.
    if st.button("🔍 Analyze Sign", type="primary"):
        with st.spinner("Processing..."):
            
            # --- PREDICTION LOGIC ---
            if "Letters" in mode:
                model = load_letters_engine()
                tf = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                tensor = tf(original_pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = model(tensor)
                    prob = torch.nn.functional.softmax(out, dim=1)
                    conf, idx = torch.max(prob, 1)
                    st.session_state.last_result = LETTERS_MAP.get(idx.item(), "Unknown")
                    st.session_state.last_confidence = conf.item()
            else: # Words
                classes = get_word_classes()
                model = load_words_engine(len(classes))
                tf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                tensor = tf(original_pil).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = model(pixel_values=tensor).logits
                    prob = torch.nn.functional.softmax(out, dim=1)
                    conf, idx = torch.max(prob, 1)
                    idx_val = idx.item()
                    st.session_state.last_result = classes[idx_val] if idx_val < len(classes) else "Unknown"
                    st.session_state.last_confidence = conf.item()

    # --- RESULT DISPLAY (Outside the button 'if' so it persists) ---
    if st.session_state.last_result:
        with col2:
            st.subheader("Prediction Result")
            res = st.session_state.last_result
            score = st.session_state.last_confidence
            
            # Visual Badge
            color = "green" if score > 0.8 else "orange"
            st.markdown(f"""
            <div style="border: 2px solid {color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h1 style="color: {color}; margin:0; font-size: 50px;">{res}</h1>
                <p>Confidence: {score*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- Action Buttons based on Mode ---
            if "Letters" in mode:
                c1, c2 = st.columns(2)
                # Logic to determine what to add
                char_to_add = res
                if res == "space": char_to_add = " "
                elif res == "del": char_to_add = "DEL"
                elif res == "nothing": char_to_add = ""

                if c1.button(f"➕ Add '{res}'"):
                    if char_to_add == "DEL":
                        st.session_state.sentence = st.session_state.sentence[:-1]
                    else:
                        st.session_state.sentence += char_to_add
                    st.rerun() # Refresh to update sidebar
                
                # Speak only the current letter
                if c2.button(f"🔊 Speak '{res}'"):
                    speak_text(res)
                    
            else: # Words Mode
                if st.button(f"🔊 Speak '{res}'", use_container_width=True):
                    speak_text(res)
