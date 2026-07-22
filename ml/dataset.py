from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = Path("data/stanford_dogs/Images")
IMG_SIZE = 224          # стандартный вход для ResNet
BATCH_SIZE = 32

# нормализация под ImageNet — обязательна для предобученных моделей
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def get_datasets():
    full = datasets.ImageFolder(DATA_DIR)
    n_train = int(0.8 * len(full))
    train_ds, test_ds = random_split(
        full, [n_train, len(full) - n_train],
        generator=torch.Generator().manual_seed(42),
    )
    train_ds.dataset.transform = train_tf
    return train_ds, test_ds, full.classes

if __name__ == "__main__":
    train_ds, test_ds, classes = get_datasets()
    print(f"Классов: {len(classes)}, train: {len(train_ds)}, test: {len(test_ds)}")
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    images, labels = next(iter(loader))
    print(images.shape)   # ожидаем: torch.Size([32, 3, 224, 224])
    print(labels[:5])