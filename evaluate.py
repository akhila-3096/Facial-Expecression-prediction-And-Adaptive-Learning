from pathlib import Path
import torch
from sklearn.metrics import classification_report, accuracy_score
from torchvision import models
import torch.nn as nn
from training.preprocess import create_loaders

def main():
    project = Path(__file__).resolve().parents[1]
    _, test_loader, class_names = create_loaders(
        project / "data" / "raw" / "fer2013"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        project / "models" / "best_emotion_model.pth",
        map_location=device, weights_only=False
    )
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            y_pred.extend(outputs.argmax(1).cpu().tolist())
            y_true.extend(labels.tolist())
    print(classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    ))
    print("Accuracy:", accuracy_score(y_true, y_pred))

if __name__ == "__main__":
    main()
