import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import get_model
from src.dataset import NORMALIZE_MEAN, NORMALIZE_STD

MODEL_PATH = os.path.join("saved_models", "best_model.pt")

class AgeGenderPredictor:
    """
    Inference pipeline for Age & Gender prediction with face detection fallback.
    """
    def __init__(self, model_path=MODEL_PATH, device=None, img_size=128):
        self.img_size = img_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Face Detector safely
        self.face_cascade = None
        try:
            if hasattr(cv2, 'CascadeClassifier'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            print(f"[WARNING] Could not initialize OpenCV CascadeClassifier: {e}")
        
        # Load PyTorch Model
        self.model = get_model(pretrained=False, backbone_name="efficientnet_b0").to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[PREDICT] PyTorch model loaded from: {model_path}")
        else:
            print(f"[WARNING] Model checkpoint '{model_path}' not found. Using initialized weights.")
            
        self.model.eval()
        
        # Preprocessing Transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
        ])
        
    def detect_face(self, bgr_image):
        if self.face_cascade is None:
            return bgr_image, False, None
            
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        if len(faces) > 0:
            # Pick largest detected face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Add padding
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            h_img, w_img = bgr_image.shape[:2]
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w_img, x + w + pad_w)
            y2 = min(h_img, y + h + pad_h)
            
            face_crop = bgr_image[y1:y2, x1:x2]
            return face_crop, True, [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        else:
            return bgr_image, False, None

    def predict(self, input_image):
        """
        Accepts PIL Image, file path, or numpy array.
        Returns prediction dictionary.
        """
        if isinstance(input_image, str):
            bgr_image = cv2.imread(input_image)
            if bgr_image is None:
                raise ValueError(f"Could not read image file: {input_image}")
        elif isinstance(input_image, Image.Image):
            rgb_image = np.array(input_image)
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        elif isinstance(input_image, np.ndarray):
            bgr_image = input_image
        else:
            raise TypeError("Unsupported image input type.")
            
        # Detect and crop face
        face_img, face_detected, bbox = self.detect_face(bgr_image)
        
        # Convert BGR face crop to RGB PIL Image
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_face = Image.fromarray(rgb_face)
        
        # Apply transformation tensor
        tensor_img = self.transform(pil_face).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            gender_logit, age_pred = self.model(tensor_img)
            
            gender_prob = torch.sigmoid(gender_logit).item()
            predicted_age = float(age_pred.item()) * 100.0
            
        # Process output values
        predicted_age_int = max(0, int(round(predicted_age)))
        gender_label = "Female" if gender_prob > 0.5 else "Male"
        gender_confidence = gender_prob if gender_prob > 0.5 else (1.0 - gender_prob)
        
        age_min = max(0, predicted_age_int - 3)
        age_max = predicted_age_int + 3
        age_range = f"{age_min} - {age_max}"
        
        # Age group classification
        if predicted_age_int <= 12:
            age_group = "Child"
        elif predicted_age_int <= 21:
            age_group = "Teens / Young Adult"
        elif predicted_age_int <= 45:
            age_group = "Adult"
        elif predicted_age_int <= 60:
            age_group = "Middle Age"
        else:
            age_group = "Senior"
            
        return {
            "gender": gender_label,
            "gender_confidence": round(gender_confidence * 100, 2),
            "age": predicted_age_int,
            "age_raw": round(predicted_age, 1),
            "age_range": age_range,
            "age_group": age_group,
            "face_detected": face_detected,
            "bbox": bbox
        }

if __name__ == "__main__":
    predictor = AgeGenderPredictor()
    # Test on a sample synthetic image
    dummy_img = np.full((300, 300, 3), 128, dtype=np.uint8)
    res = predictor.predict(dummy_img)
    print("Inference test result:", res)
