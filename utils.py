import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

class HandExtractor:
    def __init__(self, target_size=(128, 128), padding=40):
        self.target_size = target_size
        self.padding = padding
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def process_frame(self, frame_np):
        if frame_np is None: return None, None
        
        results = self.hands.process(frame_np)

        if not results.multi_hand_landmarks:
            return None, frame_np
        
        hand = results.multi_hand_landmarks[0]
        h, w, c = frame_np.shape
        
        xs = [lm.x * w for lm in hand.landmark]
        ys = [lm.y * h for lm in hand.landmark]
        
        x_min = int(max(0, min(xs) - self.padding))
        y_min = int(max(0, min(ys) - self.padding))
        x_max = int(min(w, max(xs) + self.padding))
        y_max = int(min(h, max(ys) + self.padding))
        
        hand_img_np = frame_np[y_min:y_max, x_min:x_max]
        
        if hand_img_np.size == 0:
            return None, frame_np
            
        pil_img = Image.fromarray(hand_img_np)
        cv2.rectangle(frame_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return pil_img, frame_np
