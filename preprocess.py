from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_loaders(data_dir="data/raw/fer2013", image_size=224,
                   batch_size=16, num_workers=0):
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225])
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406], [.229,.224,.225])
    ])
    base = Path(data_dir)
    train_ds = datasets.ImageFolder(base / "train", transform=train_transform)
    test_ds = datasets.ImageFolder(base / "test", transform=test_transform)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
        train_ds.classes
    )
