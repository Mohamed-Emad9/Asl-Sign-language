import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
import time
from collections import deque
from statistics import mode

# ==========================================
# 1. Configuration & Constants
# ==========================================
MODEL_PATH = "modeeeeel.pth"  # تأكد إن الاسم صح
NUM_CLASSES = 28
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'del', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
    11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'space',
    21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'X', 26: 'Y', 27: 'Z'
}


# ==========================================
# 2. Model Architecture
# ==========================================
class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(256 * 14 * 14, 512), nn.ReLU(), nn.Dropout(0.6),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ==========================================
# 3. Helper Functions
# ==========================================
@st.cache_resource
def load_model():
    model = SignLanguageCNN(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

def process_image_for_model(image_pil):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    return transform(image_pil).unsqueeze(0).to(DEVICE)

# ==========================================
# 4. Streamlit App Logic
# ==========================================
st.set_page_config(page_title="Sign Language AI", layout="wide", page_icon="✋")

# --- Sidebar for Controls (احترافية أكتر) ---
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Adjust camera sensitivity here.")
detection_conf = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
tracking_conf = st.sidebar.slider("Tracking Confidence", 0.1, 1.0, 0.5)

st.title("✋ Sign Language Recognition")
st.markdown("### AI Powered | Real-time Detection")
st.markdown("---")

# تحميل الموديل
model = load_model()

# إعداد ميديا بايب
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=3) 
connection_style = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)

if model is not None:
    # زر التشغيل في المنتصف
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        run_camera = st.checkbox('🔴 Start Camera', value=False)
    
    # Layout: عمودين (واحد للكاميرا وواحد للنتيجة والسكلتون)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Live Feed")
        st_frame = st.empty()
    
    with col2:
        st.subheader("🧠 AI Vision (Skeleton)")
        st_skeleton = st.empty()
        st.markdown("---")
        st.subheader("Prediction:")
        st_prediction = st.empty()

    if run_camera:
        cap = cv2.VideoCapture(0)
        
        # قائمة لتخزين آخر 5 توقعات عشان نمنع "رعشة" النتيجة
        prediction_history = deque(maxlen=5)
        
        # لحساب الـ FPS
        prev_time = 0

        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf) as hands:
            
            while cap.isOpened() and run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera disconnected.")
                    break
                
                # حساب الـ FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                
                # معالجة الصورة
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = hands.process(image_rgb)
                
                # الخلفية السوداء
                skeleton_image = np.zeros((h, w, c), dtype=np.uint8)
                
                final_label = "..."
                confidence_score = 0.0

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # رسم السكلتون
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(
                            skeleton_image, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=landmark_style,
                            connection_drawing_spec=connection_style
                        )
                        
                        # التوقع
                        pil_img = Image.fromarray(skeleton_image)
                        img_tensor = process_image_for_model(pil_img)
                        
                        with torch.no_grad():
                            output = model(img_tensor)
                            probabilities = torch.nn.functional.softmax(output, dim=1)
                            confidence, predicted_idx = torch.max(probabilities, 1)
                            
                            idx = predicted_idx.item()
                            label = CLASS_NAMES.get(idx, "?")
                            
                            # إضافة التوقع للتاريخ لتثبيت النتيجة
                            prediction_history.append(label)
                            
                            confidence_score = confidence.item() * 100

                    # التثبيت: بناخد الحرف الأكثر تكراراً في آخر 5 فريمات
                    if prediction_history:
                        try:
                            final_label = mode(prediction_history)
                        except:
                            final_label = prediction_history[-1]

                # كتابة الـ FPS على الفيديو
                cv2.putText(frame, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # عرض الفيديوهات
                st_frame.image(frame, channels="BGR", use_container_width=True)
                st_skeleton.image(skeleton_image, channels="BGR", use_container_width=True)
                
                # عرض النتيجة بشكل احترافي
                if confidence_score > 60:
                    st_prediction.markdown(
                        f"""
                        <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #28a745;">
                            <h1 style="color: #155724; margin:0;">{final_label}</h1>
                            <p style="color: #155724; margin:0;">Confidence: {confidence_score:.1f}%</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                else:
                    st_prediction.markdown(
                        f"""
                        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #ddd;">
                            <h1 style="color: #6c757d; margin:0;">...</h1>
                            <p style="color: #6c757d; margin:0;">Show hand to camera</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

            cap.release()
    else:
        st.info("👋 Welcome! Click 'Start Camera' to begin translation.")