from pathlib import Path
from collections import deque
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2

class EmotionPredictor:
    """Loads a compatible MobileNetV3-Small checkpoint from models/."""
    def __init__(self, model_path=None, device=None, smoothing_window=7):
        project_root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path) if model_path else (
            project_root / "models" / "best_emotion_model.pth"
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained checkpoint not found: {self.model_path}. "
                "Train the model and place best_emotion_model.pth in models/."
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
        )
        self.class_names = checkpoint.get(
            "class_names",
            ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        )
        self.image_size = checkpoint.get("image_size", 224)

        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features, len(self.class_names)
        )
        state = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.history = deque(maxlen=smoothing_window)

    @torch.inference_mode()
    def predict(self, face_image):
        if face_image is None or face_image.size == 0:
            raise ValueError("Empty face image supplied to predictor.")
        if len(face_image.shape) == 2:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        self.history.append(probabilities)
        averaged = sum(self.history) / len(self.history)
        index = int(averaged.argmax())
        return self.class_names[index], float(averaged[index])
