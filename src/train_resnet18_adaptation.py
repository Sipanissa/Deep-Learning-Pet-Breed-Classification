"""ResNet18 (IMAGENET1K_V1) as the CNN comparison model, trained like the ViT runs; default full fine-tuning.

--mode finetune: all 11.2M parameters train, backbone and 10-class head with their own learning rates, BatchNorm in
train mode. --mode frozen: backbone frozen in eval mode, only the head trains (behaviour of train_resnet18.py).
Same dataset, class order, loader order, metrics, selection by validation macro-F1 and resumable checkpoints as
train_vit_lora.py; AdamWNoLerp avoids the DirectML CPU fallback of lerp in torch.optim.AdamW.
The Oxford test split is never loaded; resolve_dataset only checks the bytes of test.csv.
train_resnet18.py stays unchanged as the protected script of the data phase.
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
from torchvision.models import ResNet18_Weights, resnet18

from dataset_paths import add_training_arguments, resolve_dataset, sha256, write_run_config
from pet_dataset import CLASS_NAMES, PetDataset
from train_vit_linear_probe import DEFAULT_DATASET, stratified_subset
from train_vit_lora import evaluate, optimizer_state_cpu, random_states, restore_random_states
from vit_lora import AdamWNoLerp

WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
EXPECTED_CLASSES = ["British Shorthair", "Russian Blue", "Persian", "Ragdoll", "Bengal", "Samoyed", "Pomeranian", "Keeshond", "Great Pyrenees", "Newfoundland"]
# Diese Einstellungen müssen beim Fortsetzen gleich bleiben
SETTINGS = ["dataset", "mode", "batch_size", "learning_rate", "head_learning_rate", "weight_decay", "seed", "limit_train", "limit_val"]


# Einstellungen von der Kommandozeile lesen
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_training_arguments(parser, epochs=10, batch_size=32, learning_rate=0.0001, seed=42)
    parser.set_defaults(dataset=DEFAULT_DATASET, online_flip="off")
    parser.add_argument("--mode", choices=["finetune", "frozen"], default="finetune", help="finetune: all layers; frozen: head only")
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
        args.output_dir = str(Path("results") / Path(args.dataset).name / f"seed_{args.seed}" / f"resnet18_{args.mode}")
    return args


def build_resnet18(mode="finetune", num_classes=10):
    """Pretrained ResNet18 with a new head Linear(512, 10); in frozen mode only the head requires gradients."""
    # Vortrainiertes Modell laden und Kopf für zehn Rassen ersetzen
    model = resnet18(weights=WEIGHTS)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Im Modus frozen wird nur der Kopf trainiert
    if mode == "frozen":
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    config = dict(architecture="resnet18", weights=str(WEIGHTS), mode=mode, classes=CLASS_NAMES, head="Linear(512, 10), torch default init",
                  parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  head_parameters=sum(p.numel() for p in model.fc.parameters()))
    return model, WEIGHTS.transforms(), config


# Trainingsmodus passend zum Verfahren setzen
def set_train_mode(model, mode):
    model.train()
    if mode == "frozen":
        model.eval()       # BatchNorm behält die ImageNet-Werte
        model.fc.train()


# Prüfsumme der eingefrorenen Teile, um Änderungen zu erkennen
def frozen_fingerprint(model):
    """sha256 over all tensors that must not change: frozen parameters and, in frozen mode, the BatchNorm buffers."""
    digest = hashlib.sha256()
    frozen = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    tensors = frozen + (list(model.named_buffers()) if frozen else [])
    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# Gewichte zum Speichern auf die CPU kopieren
def model_state_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# Gespeichertes Modell laden
def load_resnet18_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = json.loads(checkpoint["config"])
    assert config["weights"] == str(WEIGHTS) and config["classes"] == CLASS_NAMES
    model, preprocess, _ = build_resnet18(config["mode"])
    model.load_state_dict(checkpoint["model"])
    return model, preprocess, config, checkpoint


def main(argv=None):
    args = parse_args(argv)
    assert CLASS_NAMES == EXPECTED_CLASSES, CLASS_NAMES
    out = Path(args.output_dir)
    # Zufallszahlen festlegen, damit der Lauf wiederholbar ist
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = resolve_dataset(args.dataset)
    device = torch_directml.device()

    # Modell aufbauen
    model, preprocess, config = build_resnet18(args.mode)
    frozen_sha = frozen_fingerprint(model)
    model = model.to(device)
    # Backbone und Kopf haben eigene Lernraten
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("fc.")]
    head_params = list(model.fc.parameters())
    groups = ([dict(params=backbone_params, lr=args.learning_rate)] if backbone_params else []) + [dict(params=head_params, lr=args.head_learning_rate)]
    optimizer = AdamWNoLerp(groups, weight_decay=args.weight_decay)
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
    print(f"ResNet18 {args.mode} | Parameter {config['parameters']}, trainierbar {config['trainable_parameters']} | Training {len(train_indices)}, Validierung {len(val_indices)}; Test wird nicht geladen", flush=True)

    settings = {k: getattr(args, k) for k in SETTINGS}
    # Nur prüfen, ob alles läuft, dann beenden
    if args.check_only:
        for split, data in (("train", train_data), ("val", val_data)):
            counts = {name: sum(s["breed"] == name for s in data.samples) for name in CLASS_NAMES}
            assert all(counts.values()), (split, counts)
            print(f"CHECK {split}: {len(data)} |", counts, flush=True)
        set_train_mode(model, args.mode)
        images, labels = next(iter(train_loader))
        assert images.shape[1:] == (3, 224, 224)
        loss = criterion(model(images.to(device)), labels.to(device))
        loss.backward()
        assert {n for n, p in model.named_parameters() if p.grad is not None} == {n for n, p in model.named_parameters() if p.requires_grad}
        print(f"CHECK OK: Batch {tuple(images.shape)}, Loss {loss.item():.4f}; kein Optimierungsschritt, nichts gespeichert", flush=True)
        return None

    out.mkdir(parents=True, exist_ok=True)
    history, best, start_epoch, trained_seconds = [], None, 1, 0.0
    dataset_sha = dict(train=sha256(dataset["train"]), val=sha256(dataset["val"]))
    # Bei Bedarf einen unterbrochenen Lauf fortsetzen
    resume = out / "last_checkpoint.pt" if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        assert checkpoint["settings"] == settings, ("Einstellungen weichen vom Checkpoint ab", checkpoint["settings"], settings)
        # Nur fortsetzen, wenn Modell und Daten unverändert sind
        assert checkpoint["frozen_sha256"] == frozen_sha and checkpoint["dataset_sha256"] == dataset_sha
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_random_states(checkpoint["random_states"], loader_generator)
        history, best, trained_seconds = checkpoint["history"], checkpoint["best"], checkpoint["trained_seconds"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Fortgesetzt aus {resume} nach Epoche {checkpoint['epoch']}", flush=True)
    else:
        write_run_config(out, args, dataset, f"resnet18_{args.mode}", resnet18=config, head_learning_rate=args.head_learning_rate, weight_decay=args.weight_decay,
                         optimizer="AdamWNoLerp (gleich torch.optim.AdamW, ohne lerp), " + ("Backbone und Kopf als zwei Gruppen" if backbone_params else "nur Kopf"),
                         loss="CrossEntropyLoss", selection="best validation macro-F1", batchnorm="train mode, statistics update" if args.mode == "finetune" else "eval mode, frozen",
                         preprocessing=repr(preprocess), train_images=len(train_indices), val_images=len(val_indices), frozen_sha256=frozen_sha,
                         directml=torch_directml.device_name(0).strip("\x00"), torch=str(torch.__version__))
        with (out / "training_log.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1", "seconds", "best"])

    # Training über alle Epochen
    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        set_train_mode(model, args.mode)
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
        state = model_state_cpu(model)
        if row["best"]:
            best = dict(row, confusion=confusion.tolist())
            torch.save(dict(config=json.dumps(config, sort_keys=True), model=state, epoch=epoch, metrics=best, dataset=dataset["name"], seed=args.seed), out / "best_resnet18.pt")
        # Stand zum Fortsetzen speichern. Erst als .tmp, damit bei einem
        # Abbruch keine halbe Datei entsteht.
        torch.save(dict(settings=settings, epoch=epoch, model=state, optimizer=optimizer_state_cpu(optimizer), random_states=random_states(loader_generator),
                        history=history, best=best, trained_seconds=trained_seconds, frozen_sha256=frozen_sha, dataset_sha256=dataset_sha, config=config),
                   out / "last_checkpoint.tmp")
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
    assert frozen_fingerprint(model) == frozen_sha, "eingefrorene Gewichte haben sich veraendert"
    # Zusammenfassung des Laufs speichern
    summary = dict(model="ResNet18", method="full fine-tuning" if args.mode == "finetune" else "frozen backbone, head only", weights=str(WEIGHTS), resnet18=config,
                   dataset=dataset["name"], settings=settings, epochs=args.epochs, train_images=len(train_indices), val_images=len(val_indices),
                   best_epoch=best["epoch"], best_val_macro_f1=best["val_macro_f1"], best_val_accuracy=best["val_accuracy"], best_val_loss=best["val_loss"],
                   best_per_class_f1=best["val_per_class_f1"], best_confusion=best["confusion"], trained_seconds=trained_seconds, history=history)
    (out / "history.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fertig. Beste Validierungs-Macro-F1 {best['val_macro_f1']:.4f} in Epoche {best['epoch']}; Trainingszeit {trained_seconds / 60:.1f} min; Ausgabe: {out}", flush=True)
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
