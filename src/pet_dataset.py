
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


# ==========================================
# 1. UNSERE 10 KLASSEN
# ==========================================

CLASS_NAMES = [
    "British Shorthair",
    "Russian Blue",
    "Persian",
    "Ragdoll",
    "Bengal",
    "Samoyed",
    "Pomeranian",
    "Keeshond",
    "Great Pyrenees",
    "Newfoundland",
]

CLASS_TO_IDX = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


# ==========================================
# 2. PYTORCH DATASET
# ==========================================

class PetDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.transform = transform

        csv_path = Path(csv_file)

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"CSV-Datei nicht gefunden: {csv_path}"
            )

        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:
            self.samples = list(csv.DictReader(file))

        if not self.samples:
            raise ValueError(
                f"CSV-Datei enthält keine Bilder: {csv_path}"
            )

        for sample in self.samples:
            if sample["breed"] not in CLASS_TO_IDX:
                raise ValueError(
                    f"Unbekannte Klasse: {sample['breed']}"
                )

            if not Path(sample["path"]).is_file():
                raise FileNotFoundError(
                    f"Bild nicht gefunden: {sample['path']}"
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_path = Path(sample["path"])
        breed = sample["breed"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        label = CLASS_TO_IDX[breed]

        return image, label


# ==========================================
# 3. EINFACHER FUNKTIONSTEST
# ==========================================

if __name__ == "__main__":

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    splits = {
        "Training": "data/final_splits/train.csv",
        "Validierung": "data/final_splits/val.csv",
        "Test": "data/final_splits/test.csv",
    }

    print("=== DATASET TEST ===")

    for name, csv_file in splits.items():

        dataset = PetDataset(
            csv_file=csv_file,
            transform=transform
        )

        image, label = dataset[0]

        print(f"\n{name}: {len(dataset)} Bilder")
        print("Bild-Tensor:", image.shape)
        print("Klassenindex:", label)
        print("Rasse:", CLASS_NAMES[label])

    print("\nDataset-Test erfolgreich!")
