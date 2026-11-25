"""ViT-B/16 (IMAGENET1K_V1) with LoRA on query and value in all 12 layers; only LoRA and the 10-class head train.

Uses vit_lora.py (DirectML-safe layer norm, attention without CPU fallback, AdamW without lerp). After every epoch a
resumable checkpoint is written with trainable weights, optimizer state, epoch, history and all random states
(Python, NumPy, Torch CPU, DirectML, loader order). The best epoch by validation macro-F1 is kept separately.
The Oxford test split is never loaded; resolve_dataset only checks the bytes of test.csv.
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
from torch.utils.data import DataLoader, Subset

from dataset_paths import add_training_arguments, output_dir, resolve_dataset, sha256, write_run_config
from pet_dataset import CLASS_NAMES, PetDataset
from train_vit_linear_probe import DEFAULT_DATASET, macro_f1, stratified_subset
from vit_lora import AdamWNoLerp, WEIGHTS, build_vit_lora, save_lora_checkpoint, trainable_state

# Diese Einstellungen müssen beim Fortsetzen gleich bleiben
SETTINGS = ["dataset", "rank", "lora_alpha", "lora_dropout", "batch_size", "learning_rate", "head_learning_rate", "weight_decay", "seed", "limit_train", "limit_val"]


# Einstellungen von der Kommandozeile lesen
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_training_arguments(parser, epochs=10, batch_size=32, learning_rate=0.0005, seed=42)
    parser.set_defaults(dataset=DEFAULT_DATASET, online_flip="off")
    parser.add_argument("--rank", type=int, choices=[4, 8, 16], default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--head-learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0, help="image loading processes; does not change the results")
    parser.add_argument("--resume", default=None, help="'auto' (last_checkpoint.pt in the output folder) or a checkpoint path")
    parser.add_argument("--log-every", type=int, default=50, help="progress line every N training batches")
    parser.add_argument("--limit-train", type=int, default=0, help="technical test: stratified subset of N training images")
    parser.add_argument("--limit-val", type=int, default=0, help="technical test: stratified subset of N validation images")
    parser.add_argument("--stop-after-epoch", type=int, default=0, help="technical test: end the process after this epoch, as an interruption")
    args = parser.parse_args(argv)
    if args.online_flip == "on":
        parser.error("the training images are augmented offline; --online-flip on is not part of this setup")
    # Ein technischer Test darf keinen echten Lauf überschreiben
    if (args.limit_train or args.limit_val or args.stop_after_epoch) and not args.output_dir:
        parser.error("technical tests need --output-dir, so they cannot touch a real run")
    if args.output_dir is None:
        args.output_dir = str(Path("results") / Path(args.dataset).name / f"seed_{args.seed}" / f"vit_b16_lora_r{args.rank}")
    return args


# Prüfsumme der eingefrorenen Gewichte, um Änderungen zu erkennen
def fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# Alle Zufallszustände sichern, damit ein fortgesetzter Lauf
# genauso weiterrechnet wie ohne Unterbrechung
def random_states(loader_generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                directml=torch_directml.default_generator.get_state(), loader=loader_generator.get_state())


# Gesicherte Zufallszustände wiederherstellen
def restore_random_states(states, loader_generator):
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    torch_directml.default_generator.set_state(states["directml"])
    loader_generator.set_state(states["loader"])


# Zustand des Optimizers zum Speichern auf die CPU holen
def optimizer_state_cpu(optimizer):
    state = optimizer.state_dict()
    state["state"] = {k: {n: v.cpu() if torch.is_tensor(v) else v for n, v in s.items()} for k, s in state["state"].items()}
    return state


# Modell auf der Validierung auswerten
def evaluate(model, loader, device, criterion):
    model.eval()
    confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.long)
    total_loss, seen = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device)).cpu()
            total_loss += criterion(outputs, labels).item() * len(labels)
            seen += len(labels)
            confusion += torch.bincount(labels * len(CLASS_NAMES) + outputs.argmax(1), minlength=len(CLASS_NAMES) ** 2).view(len(CLASS_NAMES), -1)
    f1, per_class = macro_f1(confusion)
    return dict(loss=total_loss / seen, accuracy=100.0 * float(confusion.diag().sum()) / seen, macro_f1=f1, per_class_f1=dict(zip(CLASS_NAMES, per_class))), confusion


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_dir)
    # Zufallszahlen festlegen, damit der Lauf wiederholbar ist
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = resolve_dataset(args.dataset)
    device = torch_directml.device()

    # Modell mit LoRA aufbauen und trainierbare Teile prüfen
    model, preprocess, config = build_vit_lora(args.rank, args.lora_alpha, args.lora_dropout)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n or n.startswith("heads.head.") for n in trainable)
    assert config["trainable_parameters"] == args.rank * 12 * 2 * 2 * 768 + 7690
    frozen_sha = fingerprint(model)
    model = model.to(device)
    # LoRA und Kopf haben eigene Lernraten
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("heads.head.")]
    optimizer = AdamWNoLerp([dict(params=lora_params, lr=args.learning_rate), dict(params=head_params, lr=args.head_learning_rate)], weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Datensatz laden
    train_data = PetDataset(dataset["train"], transform=preprocess)
    val_data = PetDataset(dataset["val"], transform=preprocess)
    train_indices = stratified_subset(train_data, args.limit_train, args.seed)
    val_indices = stratified_subset(val_data, args.limit_val, args.seed)
    # Kein Testbild darf in Training oder Validierung vorkommen
    with open(dataset["test"], encoding="utf-8-sig", newline="") as handle:
        test_paths = {row["path"] for row in csv.DictReader(handle)}
    assert not test_paths & ({train_data.samples[i]["path"] for i in train_indices} | {val_data.samples[i]["path"] for i in val_indices})
    # Bilder in fester, vom Seed abhängiger Reihenfolge laden
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=args.batch_size, shuffle=True, generator=loader_generator, num_workers=args.num_workers)
    val_loader = DataLoader(Subset(val_data, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"ViT-B/16 LoRA r{args.rank} alpha {args.lora_alpha} | trainierbar {config['trainable_parameters']} | Training {len(train_indices)}, Validierung {len(val_indices)}; Test wird nicht geladen", flush=True)

    settings = {k: getattr(args, k) for k in SETTINGS}
    # Nur prüfen, ob alles läuft, dann beenden
    if args.check_only:
        model.train()
        images, labels = next(iter(train_loader))
        loss = criterion(model(images.to(device)), labels.to(device))
        loss.backward()
        assert {n for n, p in model.named_parameters() if p.grad is not None} == set(trainable)
        print(f"CHECK OK: Batch {tuple(images.shape)}, Loss {loss.item():.4f}; kein Optimierungsschritt, nichts gespeichert", flush=True)
        return None

    out.mkdir(parents=True, exist_ok=True)
    history, best, start_epoch, trained_seconds = [], None, 1, 0.0
    # Bei Bedarf einen unterbrochenen Lauf fortsetzen
    resume = out / "last_checkpoint.pt" if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        assert checkpoint["settings"] == settings, ("Einstellungen weichen vom Checkpoint ab", checkpoint["settings"], settings)
        # Nur fortsetzen, wenn Modell und Daten unverändert sind
        assert checkpoint["frozen_sha256"] == frozen_sha and checkpoint["dataset_sha256"] == dict(train=sha256(dataset["train"]), val=sha256(dataset["val"]))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(checkpoint["trainable"][name])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_random_states(checkpoint["random_states"], loader_generator)
        history, best, trained_seconds = checkpoint["history"], checkpoint["best"], checkpoint["trained_seconds"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Fortgesetzt aus {resume} nach Epoche {checkpoint['epoch']}", flush=True)
    else:
        write_run_config(out, args, dataset, f"vit_b16_lora_r{args.rank}", lora=config, head_learning_rate=args.head_learning_rate, weight_decay=args.weight_decay,
                         optimizer="AdamWNoLerp (gleich torch.optim.AdamW, ohne lerp), zwei Gruppen", loss="CrossEntropyLoss", selection="best validation macro-F1",
                         preprocessing=repr(preprocess), train_images=len(train_indices), val_images=len(val_indices), frozen_sha256=frozen_sha,
                         directml=torch_directml.device_name(0).strip("\x00"), torch=str(torch.__version__))
        with (out / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "seconds", "best"])

    # Training über alle Epochen
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = correct = seen = 0
        for step, (images, labels) in enumerate(train_loader, 1):
            # Fehler berechnen und Gewichte aktualisieren
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            # Abbrechen, wenn der Fehlerwert ungültig wird
            if not np.isfinite(loss_value):
                raise RuntimeError(f"Loss nicht endlich in Epoche {epoch}, Batch {step}")
            total_loss += loss_value * len(labels)
            correct += int((outputs.argmax(1).cpu() == labels).sum())
            seen += len(labels)
            if args.log_every and step % args.log_every == 0:
                print(f"  Epoche {epoch} Batch {step}/{len(train_loader)} | Loss {total_loss / seen:.4f} | {time.time() - start:.0f}s", flush=True)
        # Auf der Validierung prüfen
        val, confusion = evaluate(model, val_loader, device, criterion)
        seconds = time.time() - start
        trained_seconds += seconds
        row = dict(epoch=epoch, train_loss=total_loss / seen, train_accuracy=100.0 * correct / seen, val_loss=val["loss"], val_accuracy=val["accuracy"],
                   val_macro_f1=val["macro_f1"], val_per_class_f1=val["per_class_f1"], seconds=seconds)
        # Bestes Modell nach Macro-F1 speichern
        row["best"] = best is None or val["macro_f1"] > best["val_macro_f1"]
        history.append(row)
        if row["best"]:
            best = dict(row, confusion=confusion.tolist())
            save_lora_checkpoint(out / "best_lora.pt", model, config, epoch=epoch, metrics=best, dataset=dataset["name"], seed=args.seed)
        # Stand zum Fortsetzen speichern. Erst als .tmp, damit bei einem
        # Abbruch keine halbe Datei entsteht.
        torch.save(dict(settings=settings, epoch=epoch, trainable=trainable_state(model), optimizer=optimizer_state_cpu(optimizer), random_states=random_states(loader_generator),
                        history=history, best=best, trained_seconds=trained_seconds, frozen_sha256=frozen_sha,
                        dataset_sha256=dict(train=sha256(dataset["train"]), val=sha256(dataset["val"])), config=config), out / "last_checkpoint.tmp")
        (out / "last_checkpoint.tmp").replace(out / "last_checkpoint.pt")
        # Ergebnisse in einer CSV-Datei speichern
        with (out / "training_log.csv").open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([epoch] + [round(row[k], 6) for k in ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "seconds")] + [row["best"]])
        print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss {row['train_loss']:.4f} | Train Acc {row['train_accuracy']:.2f}% | Val Loss {val['loss']:.4f} | "
              f"Val Acc {val['accuracy']:.2f}% | Val Macro-F1 {val['macro_f1']:.4f}{' | bestes Modell' if row['best'] else ''} | {seconds / 60:.1f} min", flush=True)
        # Nur für den Test: Unterbrechung nachstellen
        if args.stop_after_epoch and epoch == args.stop_after_epoch and epoch < args.epochs:
            print(f"Technischer Test: Prozess endet nach Epoche {epoch} wie bei einer Unterbrechung", flush=True)
            return dict(interrupted_after=epoch, history=history)

    # Eingefrorene Gewichte dürfen sich nicht verändert haben
    model_cpu_check = fingerprint(model)
    assert model_cpu_check == frozen_sha, "eingefrorene Gewichte haben sich veraendert"
    # Zusammenfassung des Laufs speichern
    summary = dict(model="ViT-B/16", method="LoRA", weights=str(WEIGHTS), lora=config, dataset=dataset["name"], settings=settings, epochs=args.epochs,
                   train_images=len(train_indices), val_images=len(val_indices), best_epoch=best["epoch"], best_val_macro_f1=best["val_macro_f1"],
                   best_val_accuracy=best["val_accuracy"], best_val_loss=best["val_loss"], best_per_class_f1=best["val_per_class_f1"], best_confusion=best["confusion"],
                   frozen_weights_unchanged=True, trained_seconds=trained_seconds, history=history)
    (out / "history.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fertig. Beste Validierungs-Macro-F1 {best['val_macro_f1']:.4f} in Epoche {best['epoch']}; Trainingszeit {trained_seconds / 60:.1f} min; Ausgabe: {out}", flush=True)
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
