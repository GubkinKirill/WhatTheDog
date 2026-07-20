from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path('../data/stanford_dogs/Images')

breads = sorted(p.name for p in DATA_DIR.iterdir())
print(f'Пород: {len(breads)}')

total = sum(len(list(p.glob("*.jpg"))) for p in DATA_DIR.iterdir())
print(f'Всего изображений: {total}')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax in axes:
    breed_dir = DATA_DIR / random.choice(breads)
    img_path = random.choice(list(breed_dir.glob('*.jpg')))
    ax.imshow(Image.open(img_path))
    ax.set_title(breed_dir.name.split('-', 1)[1])
    ax.axis('off')
plt.savefig("sample_dogs.png")  