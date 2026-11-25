import collections
import hashlib
import json
from pathlib import Path


# Pfade zum ursprünglichen und aktuellen Datensatz
ORIGINAL = Path("data/final_splits")
POINTER = Path("data/dataset_expansion/LATEST_COMBINED_VERSION.txt")

# Erwartete Anzahl der Bilder für Training, Validierung und Test
EXPECTED_COUNTS = {"20261106T110038Z": (4898, 1233, 999)}


# Prüfsumme einer Datei berechnen
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# Einstellungen für das Training festlegen
def add_training_arguments(parser, epochs, batch_size, learning_rate, seed):
    parser.add_argument("--dataset", default="original", help="original (default), active, or a folder with train/val/test.csv")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--learning-rate", type=float, default=learning_rate)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--online-flip", choices=["default", "on", "off"], default="default", help="random horizontal flip during training; default keeps the behaviour of the script")
    parser.add_argument("--output-dir", default=None, help="default: results/<model>, or results/<dataset>/seed_<seed>/<model> for another dataset")
    parser.add_argument("--check-only", action="store_true", help="check data and model with one batch, then stop; no optimizer step, nothing saved")


# Passenden Datensatz auswählen
def resolve_dataset(name):

    # Ursprünglichen Datensatz verwenden
    if name == "original":
        return dict(name="original", folder=str(ORIGINAL), train=f"{ORIGINAL.as_posix()}/train.csv", val=f"{ORIGINAL.as_posix()}/val.csv", test=f"{ORIGINAL.as_posix()}/test.csv")

    # Pfad zum gewünschten Datensatz bestimmen
    folder = Path(POINTER.read_text(encoding="utf-8").strip()) if name == "active" else Path(name)

    files = {
        split: folder / f"{split}.csv"
        for split in ("train", "val", "test")
    }

    # Prüfen, ob alle CSV-Dateien vorhanden sind
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"CSV-Datei nicht gefunden: {path}")

    # Prüfen, ob die Oxford-Testdatei unverändert ist
    if sha256(files["test"]) != sha256(ORIGINAL / "test.csv"):
        raise ValueError(f"test.csv weicht vom unveraenderten Oxford-Test ab: {files['test']}")

    return dict(name=folder.name, folder=str(folder), **{split: str(path) for split, path in files.items()})


# Ordner für die Ergebnisse festlegen
def output_dir(args, dataset, model_name):

    if args.output_dir:
        path = Path(args.output_dir)

    elif dataset["name"] == "original":
        path = Path("results") / model_name

    else:
        path = Path("results") / dataset["name"] / f"seed_{args.seed}" / model_name

    # Ordner nur erstellen, wenn es kein Prüflauf ist
    if not args.check_only:
        path.mkdir(parents=True, exist_ok=True)

    return path


# Einstellungen des Trainings speichern
def write_run_config(path, args, dataset, model_name, **details):
    """Records what a run on a non-original dataset used; original runs keep their previous output files."""

    if dataset["name"] == "original" or args.check_only:
        return

    # Datensatz, Prüfsummen und Trainingsparameter festhalten
    config = dict(
        model=model_name,
        dataset=dataset,
        train_sha256=sha256(dataset["train"]),
        val_sha256=sha256(dataset["val"]),
        test_sha256=sha256(dataset["test"]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        online_flip=args.online_flip,
        **details
    )

    # Einstellungen als JSON-Datei speichern
    (path / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# Zufällige Reihenfolge der Bilder mit einem festen Seed erzeugen
def shuffle_generator(args, dataset):
    """A seeded loader order on non-original datasets, so all models see the same order for a seed."""

    if dataset["name"] == "original":
        return None

    import torch

    return torch.Generator().manual_seed(args.seed)


# Datensatz und Modell mit einem Batch überprüfen
def check_run(model, loader, datasets, dataset, class_names, criterion, device):
    """Validates data and one forward and backward pass; the optimizer is never called."""

    from pet_dataset import CLASS_TO_IDX, PetDataset

    # Prüfen, ob alle 10 Klassen richtig zugeordnet sind
    assert list(CLASS_TO_IDX) == list(class_names) and len(class_names) == 10

    # Anzahl der Bilder in den drei Datensätzen bestimmen
    counts = dict(datasets, test=PetDataset(dataset["test"]))
    sizes = tuple(len(counts[split]) for split in ("train", "val", "test"))

    # Bildanzahl mit den bekannten Werten vergleichen
    if dataset["name"] in EXPECTED_COUNTS:
        assert sizes == EXPECTED_COUNTS[dataset["name"]], sizes

    # Anzahl der Bilder pro Rasse prüfen und ausgeben
    for split, data in counts.items():
        per_class = collections.Counter(
            sample["breed"] for sample in data.samples
        )

        assert set(per_class) == set(class_names), split

        print(
            f"CHECK {split}: {len(data)} |",
            {name: per_class[name] for name in class_names},
            flush=True
        )

    # Prüfen, ob dieselben Bildpfade in Training und Validierung vorkommen
    assert not {s["path"] for s in counts["train"].samples} & {s["path"] for s in counts["val"].samples}

    # Einen Batch mit Bildern und Klassen laden
    images, labels = next(iter(loader))

    # Bildgröße und Farbkanäle prüfen
    assert images.shape[1:] == (3, 224, 224)

    # Prüfen, ob die Klassen zwischen 0 und 9 liegen
    expected = [
        CLASS_TO_IDX[breed]
        for breed in (s["breed"] for s in datasets["train"].samples)
    ]

    assert sorted(set(expected)) == list(range(10)) and all(0 <= int(label) < 10 for label in labels)

    # Bilder durch das Modell schicken
    outputs = model(images.to(device))

    # Anzahl der Ausgabewerte prüfen
    assert tuple(outputs.shape) == (len(labels), 10)

    # Fehler zwischen Vorhersage und richtiger Klasse berechnen
    loss = criterion(outputs, labels.to(device))

    # Gradienten berechnen, ohne die Gewichte zu ändern
    if loss.requires_grad:
        loss.backward()

    # Ergebnisse der Prüfung ausgeben
    print(
        "CHECK batch",
        tuple(images.shape),
        "loss",
        round(float(loss.item()), 4),
        "| dataset",
        dataset["name"],
        "| test.csv",
        sha256(dataset["test"])[:16],
        flush=True
    )

    print(
        "CHECK OK: no optimizer step, nothing saved",
        flush=True
    )