
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch_directml

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from dataset_paths import add_training_arguments, check_run, output_dir, resolve_dataset, shuffle_generator, write_run_config
from pet_dataset import PetDataset, CLASS_NAMES


# ==========================================
# 1. Einstellungen
# ==========================================

parser = argparse.ArgumentParser()
add_training_arguments(parser, epochs=10, batch_size=16, learning_rate=0.001, seed=42)
args = parser.parse_args()

SEED = args.seed
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate

dataset = resolve_dataset(args.dataset)

TRAIN_CSV = dataset["train"]
VAL_CSV = dataset["val"]

OUTPUT_DIR = output_dir(args, dataset, "resnet18")

MODEL_PATH = OUTPUT_DIR / "resnet18_best.pth"
HISTORY_PATH = OUTPUT_DIR / "resnet18_history.json"

random.seed(SEED)
torch.manual_seed(SEED)

device = torch_directml.device()

print("Verwendetes Gerät:", device)


# ==========================================
# 2. Vortrainiertes ResNet18
# ==========================================

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Alle vortrainierten Parameter einfrieren
for parameter in model.parameters():
    parameter.requires_grad = False

# Klassifikationskopf für unsere 10 Rassen
num_classes = len(CLASS_NAMES)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Ausgabeklassen:", num_classes)
print("Gesamte Parameter:", total_parameters)
print("Trainierbare Parameter:", trainable_parameters)


# ==========================================
# 3. Daten vorbereiten
# ==========================================

# ImageNet-Vorverarbeitung passend zu ResNet18
transform = weights.transforms()

train_transform = transform

if args.online_flip == "on":
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transform
    ])

train_dataset = PetDataset(
    TRAIN_CSV,
    transform=train_transform
)

val_dataset = PetDataset(
    VAL_CSV,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=shuffle_generator(args, dataset)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Trainingsbilder:", len(train_dataset))
print("Validierungsbilder:", len(val_dataset))


# ==========================================
# 4. Loss Function und Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

if args.check_only:
    model.eval()
    check_run(model, train_loader, dict(train=train_dataset, val=val_dataset), dataset, CLASS_NAMES, criterion, device)
    sys.exit(0)

write_run_config(OUTPUT_DIR, args, dataset, "resnet18", image_size=224, normalization="ImageNet weights.transforms()", optimizer="AdamW (head only)", parameters=total_parameters, trainable_parameters=trainable_parameters)


# ==========================================
# 5. Eine Trainings-Epoche
# ==========================================

def train_one_epoch():

    # Backbone bleibt im Evaluierungsmodus.
    # Dadurch ändern sich die BatchNorm-Statistiken
    # des eingefrorenen ResNet18 nicht.
    model.eval()

    # Nur der Klassifikationskopf wird trainiert.
    model.fc.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ==========================================
# 6. Validierung
# ==========================================

def validate():

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ==========================================
# 7. Training starten
# ==========================================

history = []

best_val_accuracy = -1.0

print("\n===================================")
print("ResNet18 Transfer Learning startet")
print("===================================\n")

training_start = time.time()

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    train_loss, train_accuracy = train_one_epoch()

    val_loss, val_accuracy = validate()

    epoch_time = time.time() - epoch_start

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.2f}% | "
        f"Zeit: {epoch_time:.1f}s",
        flush=True
    )

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "epoch_seconds": epoch_time
    })

    # Bestes Modell anhand der Validation speichern
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.cpu().state_dict(),
            MODEL_PATH
        )

        model = model.to(device)

        print(
            f"  Neues bestes Modell gespeichert: "
            f"{best_val_accuracy:.2f}%",
            flush=True
        )


# ==========================================
# 8. Ergebnisse speichern
# ==========================================

total_time = time.time() - training_start

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        {
            "model": "ResNet18",
            "method": "Transfer Learning - Frozen Backbone",
            "classes": CLASS_NAMES,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "train_images": len(train_dataset),
            "val_images": len(val_dataset),
            "best_val_accuracy": best_val_accuracy,
            "total_training_seconds": total_time,
            "history": history
        },
        file,
        indent=4
    )


print("\n===================================")
print("Training abgeschlossen!")
print("===================================")

print(f"Beste Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"Gesamtdauer: {total_time:.1f} Sekunden")
print("Modell gespeichert unter:", MODEL_PATH)
print("Trainingsverlauf:", HISTORY_PATH)
