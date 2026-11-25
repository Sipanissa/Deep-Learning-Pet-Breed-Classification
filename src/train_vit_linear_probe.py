"""Linear probing of ViT-B/16 (IMAGENET1K_V1): frozen backbone, only the 10-class head trains.

The backbone runs on DirectML. By default its class-token features are computed once and cached: the training images
are augmented offline, the preprocessing is deterministic and the backbone has no dropout, so every epoch would see
exactly the same features. The head (7,690 parameters) then trains on the CPU with AdamW. The best epoch is chosen
by validation macro-F1. The Oxford test split is never loaded; only the byte check of test.csv in resolve_dataset runs.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch_directml
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import transforms

from dataset_paths import add_training_arguments, output_dir, resolve_dataset, sha256, write_run_config
from pet_dataset import CLASS_NAMES, PetDataset
from vit_linear_probe import WEIGHTS, build_vit_linear_probe, set_probe_mode, vit_features

MODEL_NAME = "vit_b16_linear_probe"
DEFAULT_DATASET = "data/dataset_expansion/augmented_versions/20261106T110038Z_15000"
CACHE_ROOT = Path("results/vit_feature_cache")


# Einstellungen von der Kommandozeile lesen
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_training_arguments(parser, epochs=30, batch_size=32, learning_rate=0.001, seed=42)
    parser.set_defaults(dataset=DEFAULT_DATASET, online_flip="off")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--features", choices=["cache", "online"], default="cache", help="cache: compute frozen features once (default); online: recompute every batch")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--limit-train", type=int, default=0, help="technical test: stratified subset of N training images")
    parser.add_argument("--limit-val", type=int, default=0, help="technical test: stratified subset of N validation images")
    args = parser.parse_args(argv)
    # Gespeicherte Merkmale passen nicht zu zufällig gespiegelten Bildern
    if args.features == "cache" and args.online_flip == "on":
        parser.error("--online-flip on changes the images every epoch; use --features online")
    # Ein Test mit wenigen Bildern darf keinen echten Lauf überschreiben
    if (args.limit_train or args.limit_val) and not args.output_dir:
        parser.error("a limited technical test needs --output-dir, so it cannot overwrite a real run")
    return args


# Zufallszahlen festlegen, damit der Lauf wiederholbar ist
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# Kleine Teilmenge mit möglichst gleich vielen Bildern je Klasse
def stratified_subset(dataset, limit, seed):
    if not limit:
        return list(range(len(dataset)))
    by_class = {name: [] for name in CLASS_NAMES}
    for index, sample in enumerate(dataset.samples):
        by_class[sample["breed"]].append(index)
    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)
    chosen = []
    while len(chosen) < limit:
        for name in CLASS_NAMES:
            if by_class[name] and len(chosen) < limit:
                chosen.append(by_class[name].pop())
    return sorted(chosen)


# F1 je Klasse und Mittelwert über alle Klassen
def macro_f1(confusion):
    true_positive = confusion.diag().double()
    precision = true_positive / confusion.sum(0).clamp(min=1)
    recall = true_positive / confusion.sum(1).clamp(min=1)
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    return float(f1.mean()), [float(x) for x in f1]


# Merkmale aller Bilder mit dem eingefrorenen Backbone berechnen
def extract(backbone, dataset, indices, device, batch_size):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    features, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            features.append(vit_features(backbone, images.to(device)).cpu())
            labels.append(targets)
    return torch.cat(features), torch.cat(labels)


# Merkmale aus dem Cache laden oder neu berechnen
def cached_features(backbone, dataset, indices, csv_path, device, args, cache_dir):
    paths = [dataset.samples[i]["path"] for i in indices]
    # Der Schlüssel zeigt, ob der Cache noch zu Daten und Modell passt
    key = dict(csv_sha256=sha256(csv_path), paths_sha256=hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest(),
               weights=str(WEIGHTS), preprocess=repr(WEIGHTS.transforms()), torch=str(torch.__version__))
    file = cache_dir / f"{Path(csv_path).stem}_{key['paths_sha256'][:16]}.pt"
    if file.exists():
        stored = torch.load(file, map_location="cpu", weights_only=True)
        if stored["key"] == key:
            print(f"Merkmale aus Cache: {file} ({len(paths)} Bilder)", flush=True)
            return stored["features"], stored["labels"], str(file)
        print(f"Cache passt nicht mehr und wird neu berechnet: {file}", flush=True)
    start = time.time()
    features, labels = extract(backbone, dataset, indices, device, args.feature_batch_size)
    file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(key=key, features=features, labels=labels), file)
    print(f"Merkmale berechnet: {len(paths)} Bilder in {time.time() - start:.0f}s, gespeichert in {file}", flush=True)
    return features, labels, str(file)


# Prüfsumme des Backbones, um Änderungen zu erkennen
def fingerprint(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if not name.startswith("heads."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# Kopf auf der Validierung auswerten
def evaluate(head, features, labels, criterion, batch_size):
    head.eval()
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.long)
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            x, y = features[start:start + batch_size], labels[start:start + batch_size]
            outputs = head(x)
            total_loss += criterion(outputs, y).item() * len(y)
            confusion += torch.bincount(y * len(CLASS_NAMES) + outputs.argmax(1), minlength=len(CLASS_NAMES) ** 2).view(len(CLASS_NAMES), -1)
    f1, per_class = macro_f1(confusion)
    return dict(loss=total_loss / len(labels), accuracy=100.0 * float(confusion.diag().sum()) / len(labels), macro_f1=f1, per_class_f1=per_class), confusion


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    dataset = resolve_dataset(args.dataset)
    out = output_dir(args, dataset, MODEL_NAME)
    device = torch_directml.device()
    print("Backbone auf:", torch_directml.device_name(0).strip("\x00"), "| Kopf auf: CPU", flush=True)

    # Modell aufbauen und prüfen, dass nur der Kopf trainierbar ist
    model, preprocess = build_vit_linear_probe()
    head = model.heads.head
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable == ["heads.head.weight", "heads.head.bias"] and sum(p.numel() for p in head.parameters()) == 7690
    backbone_before = fingerprint(model)

    # Datensatz laden
    train_transform = preprocess if args.online_flip != "on" else transforms.Compose([transforms.RandomHorizontalFlip(), preprocess])
    train_data = PetDataset(dataset["train"], transform=train_transform)
    val_data = PetDataset(dataset["val"], transform=preprocess)
    train_indices = stratified_subset(train_data, args.limit_train, args.seed)
    val_indices = stratified_subset(val_data, args.limit_val, args.seed)
    # Kein Testbild darf in Training oder Validierung vorkommen
    with open(dataset["test"], encoding="utf-8-sig", newline="") as handle:
        test_paths = {row["path"] for row in csv.DictReader(handle)}
    assert not test_paths & {train_data.samples[i]["path"] for i in train_indices} and not test_paths & {val_data.samples[i]["path"] for i in val_indices}
    print(f"Datensatz {dataset['name']}: Training {len(train_indices)}, Validierung {len(val_indices)}; Test wird nicht geladen", flush=True)

    # Backbone auf die GPU, Kopf auf die CPU
    backbone = model.to(device)
    set_probe_mode(backbone)
    head_cpu = nn.Linear(768, len(CLASS_NAMES))
    head_cpu.load_state_dict({k: v.cpu() for k, v in head.state_dict().items()})
    # Nur prüfen, ob alles läuft, dann beenden
    if args.check_only:
        images, labels = next(iter(DataLoader(Subset(train_data, train_indices[:args.batch_size]), batch_size=args.batch_size)))
        outputs = head_cpu(vit_features(backbone, images.to(device)).cpu())
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        assert tuple(outputs.shape) == (len(labels), 10) and all(p.grad is not None for p in head_cpu.parameters())
        print(f"CHECK OK: Batch {tuple(images.shape)}, Loss {loss.item():.4f}; kein Optimierungsschritt, nichts gespeichert", flush=True)
        return None

    # Merkmale für Training und Validierung vorbereiten
    cache_dir = (out / "feature_cache") if (args.limit_train or args.limit_val) else CACHE_ROOT / dataset["name"]
    if args.features == "cache":
        train_x, train_y, train_cache = cached_features(backbone, train_data, train_indices, dataset["train"], device, args, cache_dir)
        val_x, val_y, val_cache = cached_features(backbone, val_data, val_indices, dataset["val"], device, args, cache_dir)
    else:
        val_x, val_y = extract(backbone, val_data, val_indices, device, args.feature_batch_size)
        train_cache = val_cache = None

    # Fehlerfunktion und Optimizer festlegen
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head_cpu.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    if args.features == "cache":
        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True, generator=generator)
    else:
        train_loader = DataLoader(Subset(train_data, train_indices), batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)

    # Einstellungen des Laufs speichern
    write_run_config(out, args, dataset, MODEL_NAME, weights=str(WEIGHTS), weights_url=WEIGHTS.url, preprocessing=repr(preprocess),
                     optimizer="AdamW (head only, CPU)", weight_decay=args.weight_decay, loss="CrossEntropyLoss", selection="best validation macro-F1",
                     features=args.features, feature_cache=[train_cache, val_cache], train_images=len(train_indices), val_images=len(val_indices),
                     parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=7690, backbone_sha256=backbone_before)
    history, best = [], None
    log_file = out / "training_log.csv"
    fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "seconds", "best"]
    with log_file.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(fields)
    start_all = time.time()
    # Training über alle Epochen
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        head_cpu.train()
        total_loss = correct = seen = 0
        for x, y in train_loader:
            if args.features == "online":
                with torch.no_grad():
                    x = vit_features(backbone, x.to(device)).cpu()
            # Fehler berechnen und Gewichte aktualisieren
            optimizer.zero_grad()
            outputs = head_cpu(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += int((outputs.argmax(1) == y).sum())
            seen += len(y)
        # Auf der Validierung prüfen
        val, confusion = evaluate(head_cpu, val_x, val_y, criterion, args.batch_size)
        row = dict(epoch=epoch, train_loss=total_loss / seen, train_accuracy=100.0 * correct / seen, val_loss=val["loss"], val_accuracy=val["accuracy"],
                   val_macro_f1=val["macro_f1"], val_per_class_f1=dict(zip(CLASS_NAMES, val["per_class_f1"])), seconds=time.time() - start)
        # Bestes Modell nach Macro-F1 speichern
        improved = best is None or val["macro_f1"] > best["val_macro_f1"]
        row["best"] = improved
        history.append(row)
        if improved:
            best = dict(row, confusion=confusion.tolist())
            torch.save(dict(head=head_cpu.state_dict(), epoch=epoch, metrics=best, weights=str(WEIGHTS), classes=CLASS_NAMES, dataset=dataset["name"], seed=args.seed), out / "best_head.pt")
        # Letzten Stand und Protokoll nach jeder Epoche speichern
        torch.save(dict(head=head_cpu.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, history=history, weights=str(WEIGHTS), classes=CLASS_NAMES, seed=args.seed), out / "last_checkpoint.pt")
        with log_file.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([row[f] if not isinstance(row[f], float) else round(row[f], 6) for f in fields])
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss {row['train_loss']:.4f} | Train Acc {row['train_accuracy']:.2f}% | Val Loss {val['loss']:.4f} | "
              f"Val Acc {val['accuracy']:.2f}% | Val Macro-F1 {val['macro_f1']:.4f}{' | bestes Modell' if improved else ''} | {row['seconds']:.1f}s", flush=True)

    # Der Backbone darf sich nicht verändert haben
    assert fingerprint(model) == backbone_before, "der eingefrorene Backbone hat sich veraendert"
    # Zusammenfassung des Laufs speichern
    summary = dict(model="ViT-B/16", method="Linear Probing", weights=str(WEIGHTS), classes=CLASS_NAMES, dataset=dataset["name"], seed=args.seed,
                   epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay, features=args.features,
                   train_images=len(train_indices), val_images=len(val_indices), best_epoch=best["epoch"], best_val_macro_f1=best["val_macro_f1"],
                   best_val_accuracy=best["val_accuracy"], best_confusion=best["confusion"], backbone_unchanged=True, total_seconds=time.time() - start_all, history=history)
    (out / "history.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fertig. Beste Validierungs-Macro-F1 {best['val_macro_f1']:.4f} in Epoche {best['epoch']}; Ausgabe: {out}", flush=True)
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
