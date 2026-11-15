from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from training.preprocess import create_loaders

def main():
    project = Path(__file__).resolve().parents[1]
    train_loader, test_loader, class_names = create_loaders(
        project / "data" / "raw" / "fer2013"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(class_names)
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 6

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{epochs} | loss={total_loss/total:.4f} "
              f"| accuracy={correct/total:.4f}")

    model_dir = project / "models"
    model_dir.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": 224,
        "architecture": "MobileNetV3-Small"
    }, model_dir / "best_emotion_model.pth")
    print("Saved checkpoint to models/best_emotion_model.pth")

if __name__ == "__main__":
    main()
