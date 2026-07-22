from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = Path("data/stanford_dogs/Images")
IMG_SIZE = 224          
BATCH_SIZE = 32

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
    # два взгляда на одни и те же файлы, разная обработка
    train_full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    test_full = datasets.ImageFolder(DATA_DIR, transform=test_tf)

    n = len(train_full)
    n_train = int(0.8 * n)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g).tolist()

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_ds = Subset(train_full, train_idx)   
    test_ds = Subset(test_full, test_idx)      

    return train_ds, test_ds, train_full.classes

if __name__ == "__main__":
    train_ds, test_ds, classes = get_datasets()
    print(f"Классов: {len(classes)}, train: {len(train_ds)}, test: {len(test_ds)}")

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    images, labels = next(iter(loader))
    print(images.shape)   
    print(labels[:5])

    # проверка, что подвох устранён
    a, _ = test_ds[0]
    b, _ = test_ds[0]
    print("test детерминирован:", torch.equal(a, b))        # True

    c, _ = train_ds[0]
    d, _ = train_ds[0]
    print("train аугментируется:", not torch.equal(c, d))   # True