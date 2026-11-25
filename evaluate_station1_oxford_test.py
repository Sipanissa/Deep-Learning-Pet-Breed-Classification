"""Einmalige finale Oxford-Testauswertung der fünf besten Checkpoints von Praktikum 1, Station 1 (Adaptation).

Modi (genau einer):
  --manifest-erstellen   friert Prüfsummen, beste Epochen und Validierungswerte der fünf Checkpoints ein und
                         berechnet die pHash-Werte der ImageNet-Bilder, die dem Oxford-Test nahe kommen. Öffnet keine
                         Oxford-Testbilder.
  --selbsttest --out X   gesamte Auswertung auf synthetischen Bildern in einem temporären Ordner, mit allen Prüfungen.
  --validierung --out X  wertet die Checkpoints auf dem Validierungssplit aus und vergleicht die Konfusionsmatrix mit
                         der beim Training gespeicherten (muss identisch sein). Öffnet keine Oxford-Testbilder.
  --oxford-test --freigabe OXFORD-TEST-EINMALIG
                         die eigentliche, einmalige Auswertung auf den 999 Oxford-Testbildern. Bricht ab, wenn der
                         Ergebnisordner schon existiert; nichts wird überschrieben.

In jedem Modus: keine Trainingsschritte (Optimizer und backward sind während der Auswertung gesperrt), keine
Gewichtsänderung (Prüfsumme des Modellzustands vor und nach der Auswertung, Prüfsumme der Checkpoint-Dateien).
"""
import argparse
import contextlib
import copy
import csv
import datetime
import hashlib
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch_directml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from pet_dataset import CLASS_NAMES, PetDataset  # noqa: E402

# Feste Reihenfolge der Klassen
EXPECTED_CLASSES = ["British Shorthair", "Russian Blue", "Persian", "Ragdoll", "Bengal", "Samoyed", "Pomeranian", "Keeshond", "Great Pyrenees", "Newfoundland"]
# Pfade zu Läufen, Daten und Ergebnissen
RUNS_DIR = Path("results/20261106T110038Z_15000/seed_42")
VAL_CSV = Path("data/dataset_expansion/augmented_versions/20261106T110038Z_15000/val.csv")
TEST_CSV = Path("data/final_splits/test.csv")
TEST_CSV_SHA = "158e85fc7cf32cbecd75ebe89a913f91864e5494af9ef7150ab6f2107b174355"
PROTECTED = Path("data/dataset_expansion/protected_hashes.json")
EXCLUSIONS = Path("data/dataset_expansion/candidate_versions/KANDIDAT_20261106T101453Z/ausschluss_manifest.csv")
IMAGENET_CANDIDATES = Path("data/dataset_expansion/academic_pool/imagenet/candidates.csv")
PREP = Path("results/praktikum1_station1_adaptation/oxford_test_vorbereitung")
MANIFEST = PREP / "checkpoint_manifest.json"
IMAGENET_NEAR = PREP / "imagenet_testnaehe_phash.csv"
FINAL_OUT = Path("results/praktikum1_station1_adaptation/oxford_test")
# Ohne diesen Text startet die Testauswertung nicht
RELEASE_PHRASE = "OXFORD-TEST-EINMALIG"
MODELS = [
    dict(key="lp", label="ViT-B/16 Linear Probing", folder="vit_b16_linear_probe", ckpt="best_head.pt"),
    dict(key="r4", label="ViT-B/16 LoRA r = 4", folder="vit_b16_lora_r4", ckpt="best_lora.pt"),
    dict(key="r8", label="ViT-B/16 LoRA r = 8", folder="vit_b16_lora_r8", ckpt="best_lora.pt"),
    dict(key="r16", label="ViT-B/16 LoRA r = 16", folder="vit_b16_lora_r16", ckpt="best_lora.pt"),
    dict(key="rn", label="ResNet18 Fine-Tuning", folder="resnet18_finetune", ckpt="best_resnet18.pt"),
]
BATCH = 32
BOOTSTRAP = 2000
# Bis zu diesem Abstand gelten zwei Bilder als sehr ähnlich
PHASH_NEAR = 10


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# Prüfsumme aller Gewichte, um Änderungen zu erkennen
def state_sha(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

@contextlib.contextmanager
def no_training():
    """Sperrt Optimizer und backward; ein Aufruf bricht die Auswertung ab."""
    def refuse(*_a, **_k):
        raise RuntimeError("Training ist in der Auswertung gesperrt")
    saved = (torch.optim.Optimizer.__init__, torch.Tensor.backward, torch.autograd.backward)
    torch.optim.Optimizer.__init__ = refuse
    torch.Tensor.backward = refuse
    torch.autograd.backward = refuse
    try:
        with torch.no_grad():
            yield
    finally:
        torch.optim.Optimizer.__init__, torch.Tensor.backward, torch.autograd.backward = saved




def load_model(spec):
    """Baut das Modell aus Vortrainingsgewichten und bestem Checkpoint; gibt Modell, Vorverarbeitung, Metadaten zurück."""
    path = ROOT / RUNS_DIR / spec["folder"] / spec["ckpt"]
    if spec["key"] == "lp":
        from vit_linear_probe import load_linear_probe
        model, preprocess, ck = load_linear_probe(path)
        classes, epoch, metrics = ck["classes"], ck["epoch"], ck["metrics"]
    elif spec["key"] == "rn":
        from train_resnet18_adaptation import load_resnet18_checkpoint
        model, preprocess, config, ck = load_resnet18_checkpoint(path)
        classes, epoch, metrics = config["classes"], ck["epoch"], ck["metrics"]
    else:
        from vit_lora import load_lora_checkpoint
        model, preprocess, config, ck = load_lora_checkpoint(path)
        classes, epoch, metrics = config["classes"], ck["epoch"], ck["metrics"]
    # Alle Gewichte einfrieren
    for p in model.parameters():
        p.requires_grad_(False)
    return model, preprocess, dict(classes=list(classes), epoch=int(epoch), metrics=metrics, path=path)


# Funktion für die Vorhersage vorbereiten
def make_forward(spec, model, device):
    if spec["key"] == "lp":
        from vit_linear_probe import set_probe_mode, vit_features
        head = copy.deepcopy(model.heads.head).cpu().eval()
        model.to(device)
        set_probe_mode(model)  # wie beim Training: Merkmale auf der GPU, Kopf auf der CPU
        return lambda x: head(vit_features(model, x.to(device)).cpu())
    model.to(device).eval()
    return lambda x: model(x.to(device)).cpu()


# Vorhersagen für alle Bilder berechnen
def predict(forward, dataset):
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0)
    logits, labels = [], []
    for images, targets in loader:
        logits.append(forward(images).float())
        labels.append(targets)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


# Konfusionsmatrix berechnen
def confusion_of(y, p, k=10):
    return np.bincount(y * k + p, minlength=k * k).reshape(k, k)


# Accuracy, Precision, Recall und F1 berechnen
def metrics_of(cm):
    cm = np.asarray(cm, dtype=np.int64); tp = np.diag(cm).astype(float)
    pred, true = cm.sum(0), cm.sum(1)
    precision = np.where(pred > 0, tp / np.maximum(pred, 1), 0.0)
    recall = np.where(true > 0, tp / np.maximum(true, 1), 0.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0.0)
    n = int(cm.sum()); correct = int(tp.sum())
    return dict(n=n, correct=correct, accuracy=100.0 * correct / n, macro_f1=float(f1.mean()), macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()), per_class={c: dict(precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]), support=int(true[i]))
                                                              for i, c in enumerate(CLASS_NAMES)})


# Wilson-Intervall für die Accuracy in Prozent
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# Unsicherheit der Macro-F1: Bilder viele Male zufällig neu ziehen
def bootstrap_macro_f1(y, p, seed=42, reps=BOOTSTRAP):
    rng = np.random.default_rng(seed); n = len(y); vals = np.empty(reps)
    for r in range(reps):
        i = rng.integers(0, n, n)
        vals[r] = metrics_of(confusion_of(y[i], p[i]))["macro_f1"]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# Exakter McNemar-Test: vergleicht nur Bilder, die genau ein Modell richtig erkennt
def mcnemar_exact(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b)); c = int(np.sum(~correct_a & correct_b)); n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)



# Checkpoints vor dem Test festhalten, damit sie nicht mehr verändert werden
def build_manifest():
    if (ROOT / MANIFEST).exists():
        sys.exit(f"{MANIFEST} existiert bereits; wird nicht überschrieben")
    entries = {}
    for spec in MODELS:
        folder = ROOT / RUNS_DIR / spec["folder"]
        with open(folder / "training_log.csv", encoding="utf-8", newline="") as h:
            log = list(csv.DictReader(h))
        # Beste Epoche aus dem Protokoll muss zum Checkpoint passen
        best_epoch = int(max(log, key=lambda r: float(r["val_macro_f1"]))["epoch"])
        conf = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        model, preprocess, meta = load_model(spec)
        assert meta["epoch"] == best_epoch, (spec["key"], meta["epoch"], best_epoch)
        assert meta["classes"] == CLASS_NAMES == EXPECTED_CLASSES
        assert repr(preprocess) == conf["preprocessing"], spec["key"]
        entries[spec["key"]] = dict(label=spec["label"], checkpoint=str(RUNS_DIR / spec["folder"] / spec["ckpt"]).replace("\\", "/"),
                                    checkpoint_sha256=sha256_file(meta["path"]), model_state_sha256=state_sha(model), best_epoch=best_epoch,
                                    val_accuracy=meta["metrics"]["val_accuracy"], val_macro_f1=meta["metrics"]["val_macro_f1"],
                                    val_confusion=meta["metrics"]["confusion"], preprocessing=repr(preprocess), classes=meta["classes"],
                                    run_config_sha256=sha256_file(folder / "run_config.json"), training_log_sha256=sha256_file(folder / "training_log.csv"))
        print("Manifest", spec["key"], "Epoche", best_epoch, entries[spec["key"]]["checkpoint_sha256"][:16], flush=True)
        del model
    (ROOT / PREP).mkdir(parents=True, exist_ok=True)
    manifest = dict(erstellt=datetime.datetime.now().isoformat(timespec="seconds"), auswahl="beste Validierungs-Macro-F1", klassen=CLASS_NAMES,
                    test_csv=str(TEST_CSV).replace("\\", "/"), test_csv_sha256=TEST_CSV_SHA, modelle=entries)
    (ROOT / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    build_imagenet_near()


def build_imagenet_near():
    """pHash der ImageNet-Bilder, die in der Datenphase wegen Nähe zum Oxford-Test ausgeschlossen wurden (Testschutz)."""
    from common import inspect
    norm = lambda p: str(Path(p)).lower()
    with open(ROOT / IMAGENET_CANDIDATES, encoding="utf-8-sig", newline="") as h:
        cand = {norm(r["local_path"]): r for r in csv.DictReader(h)}
    with open(ROOT / EXCLUSIONS, encoding="utf-8-sig", newline="") as h:
        rows = [r for r in csv.DictReader(h) if r["kategorie"] == "Testschutz" and r["quelle"].startswith("ImageNet")]
    out = []
    for r in rows:
        c = cand.get(norm(r["path"]))
        info = inspect(Path(r["path"]))
        assert not info["error"], r["path"]
        name = c["original_filename"] if c else Path(r["path"]).name
        out.append(dict(pfad=r["path"], rasse=r["breed"], ilsvrc_split="validation" if name.startswith("ILSVRC2012_val") else "train",
                        datei=name, phash=info["phash"], sha256=info["sha256"]))
    with open(ROOT / IMAGENET_NEAR, "w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("ImageNet-Testnähe:", len(out), "Bilder,", sum(o["ilsvrc_split"] == "train" for o in out), "davon im ILSVRC-Trainingssplit", flush=True)


# Prüfen, dass Checkpoints und Protokolle unverändert sind
def check_manifest():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    assert CLASS_NAMES == EXPECTED_CLASSES == manifest["klassen"], "Klassenreihenfolge weicht ab"
    for spec in MODELS:
        e = manifest["modelle"][spec["key"]]
        assert sha256_file(ROOT / e["checkpoint"]) == e["checkpoint_sha256"], f"Checkpoint verändert: {e['checkpoint']}"
        folder = ROOT / RUNS_DIR / spec["folder"]
        assert sha256_file(folder / "run_config.json") == e["run_config_sha256"] and sha256_file(folder / "training_log.csv") == e["training_log_sha256"]
    return manifest


# Testbilder suchen, die einem ImageNet-Bild sehr ähnlich sind

def overlap_flags(paths):
    """Kleinster pHash-Abstand (8 Ausrichtungen) jedes Bildes zu den ImageNet-Bildern nahe am Oxford-Test."""
    import imagehash
    from PIL import Image, ImageOps
    from common import variants
    with open(ROOT / IMAGENET_NEAR, encoding="utf-8", newline="") as h:
        near = [(int(r["phash"], 16), r) for r in csv.DictReader(h)]
    result = []
    for path in paths:
        with Image.open(path) as im:
            ident = int(str(imagehash.phash(ImageOps.exif_transpose(im).convert("RGB"))), 16)
        # Auch gedrehte und gespiegelte Fassungen vergleichen
        hashes = [ident] + list(variants(Path(path)).values())
        best = min(((bin(a ^ k).count("1"), r) for a in hashes for k, r in near), key=lambda t: t[0])
        result.append(dict(min_abstand=best[0], imagenet_datei=best[1]["datei"], ilsvrc_split=best[1]["ilsvrc_split"]))
    return result


# Alle fünf Modelle auswerten und die Ergebnisse speichern

def evaluate_all(dataset, ids, shas, out, manifest, split_name, overlap=None, compare_val=False):
    device = torch_directml.device()
    results, preds, protokoll = {}, {}, {}
    y_ref = None
    for spec in MODELS:
        start = time.time()
        # Modell laden und mit dem Manifest vergleichen
        model, preprocess, meta = load_model(spec)
        entry = manifest["modelle"][spec["key"]]
        assert meta["classes"] == CLASS_NAMES and meta["epoch"] == entry["best_epoch"]
        assert repr(preprocess) == entry["preprocessing"], "Vorverarbeitung weicht ab"
        before = state_sha(model)
        assert before == entry["model_state_sha256"], "Modellzustand weicht vom Manifest ab"
        dataset.transform = preprocess
        # Vorhersagen berechnen, Training ist dabei gesperrt
        with no_training():
            forward = make_forward(spec, model, device)
            logits, y = predict(forward, dataset)
        model.cpu()
        # Gewichte müssen nach der Auswertung gleich sein
        after = state_sha(model)
        assert after == before, "Gewichte haben sich in der Auswertung verändert"
        assert not any(p.requires_grad for p in model.parameters())
        p = logits.argmax(1); cm = confusion_of(y, p); m = metrics_of(cm)
        if y_ref is None:
            y_ref = y
        assert np.array_equal(y, y_ref)
        # Die gespeicherte Validierung muss genau wieder herauskommen
        if compare_val:
            assert cm.tolist() == entry["val_confusion"], (spec["key"], "Validierung nicht reproduziert")
        lo, hi = wilson(m["correct"], m["n"]); flo, fhi = bootstrap_macro_f1(y, p)
        m.update(accuracy_wilson95=[lo, hi], macro_f1_bootstrap95=[flo, fhi], confusion=cm.tolist(), best_epoch=entry["best_epoch"],
                 val_accuracy=entry["val_accuracy"], val_macro_f1=entry["val_macro_f1"], label=spec["label"])
        # Kennzahlen zusätzlich ohne die ImageNet-nahen Bilder
        if overlap is not None:
            keep = np.array([o["min_abstand"] > PHASH_NEAR for o in overlap])
            m["ohne_imagenet_naehe"] = metrics_of(confusion_of(y[keep], p[keep])) | dict(entfernt=int((~keep).sum()))
        results[spec["key"]] = m; preds[spec["key"]] = (logits, p)
        protokoll[spec["key"]] = dict(checkpoint_sha256=entry["checkpoint_sha256"], modellzustand_vorher=before, modellzustand_nachher=after, sekunden=round(time.time() - start, 1))
        print(f"{spec['label']}: Accuracy {m['accuracy']:.2f} %, Macro-F1 {m['macro_f1']:.4f} ({m['n']} Bilder, {split_name})"
              + (" | Validierung exakt reproduziert" if compare_val else ""), flush=True)
        del model, forward
    write_outputs(out, results, preds, y_ref, ids, shas, overlap, split_name)
    check_manifest()  # Checkpoints noch einmal prüfen
    return results, protokoll


# Vorhersagen, Kennzahlen und Tabellen speichern
def write_outputs(out, results, preds, y, ids, shas, overlap, split_name):
    out.mkdir(parents=True, exist_ok=True)
    # Wahrscheinlichkeiten aus den Modellausgaben berechnen
    softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    for spec in MODELS:
        logits, p = preds[spec["key"]]; prob = softmax(logits)
        with open(out / f"vorhersagen_{spec['key']}.csv", "w", encoding="utf-8", newline="") as h:
            w = csv.writer(h)
            w.writerow(["image_id", "sha256", "klasse_wahr", "klasse_vorhergesagt", "richtig"] + [f"p_{c}" for c in CLASS_NAMES])
            for i in range(len(y)):
                w.writerow([ids[i], shas[i], CLASS_NAMES[y[i]], CLASS_NAMES[p[i]], int(p[i] == y[i])] + [f"{v:.6f}" for v in prob[i]])
        (out / f"metriken_{spec['key']}.json").write_text(json.dumps(results[spec["key"]], indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "vorhersagen_alle_modelle.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h)
        w.writerow(["image_id", "sha256", "klasse_wahr"] + [f"vorhersage_{s['key']}" for s in MODELS] + (["min_phash_abstand_imagenet", "imagenet_datei"] if overlap else []))
        for i in range(len(y)):
            w.writerow([ids[i], shas[i], CLASS_NAMES[y[i]]] + [CLASS_NAMES[preds[s["key"]][1][i]] for s in MODELS]
                       + ([overlap[i]["min_abstand"], overlap[i]["imagenet_datei"]] if overlap else []))
    with open(out / "ergebnisse_tabelle.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h, delimiter=";")
        head = ["konfiguration", "beste_epoche", "val_accuracy", "val_macro_f1", "test_richtig", "test_n", "test_accuracy", "acc_wilson95_unten", "acc_wilson95_oben",
                "test_macro_f1", "f1_bootstrap95_unten", "f1_bootstrap95_oben", "test_macro_precision", "test_macro_recall"]
        w.writerow(head + (["accuracy_ohne_imagenet_naehe", "macro_f1_ohne_imagenet_naehe", "entfernt"] if overlap else []))
        for s in MODELS:
            m = results[s["key"]]
            row = [m["label"], m["best_epoch"], round(m["val_accuracy"], 4), round(m["val_macro_f1"], 6), m["correct"], m["n"], round(m["accuracy"], 4),
                   round(m["accuracy_wilson95"][0], 2), round(m["accuracy_wilson95"][1], 2), round(m["macro_f1"], 6), round(m["macro_f1_bootstrap95"][0], 4),
                   round(m["macro_f1_bootstrap95"][1], 4), round(m["macro_precision"], 6), round(m["macro_recall"], 6)]
            if overlap:
                o = m["ohne_imagenet_naehe"]; row += [round(o["accuracy"], 4), round(o["macro_f1"], 6), o["entfernt"]]
            w.writerow(row)
    with open(out / "klassenmetriken.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h, delimiter=";"); w.writerow(["konfiguration", "klasse", "support", "precision", "recall", "f1"])
        for s in MODELS:
            for c, v in results[s["key"]]["per_class"].items():
                w.writerow([s["label"], c, v["support"], round(v["precision"], 4), round(v["recall"], 4), round(v["f1"], 4)])
    with open(out / "mcnemar_paarweise.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h, delimiter=";"); w.writerow(["modell_a", "modell_b", "nur_a_richtig", "nur_b_richtig", "p_exakt_zweiseitig", "p_holm"])
        pairs = []
        for i, a in enumerate(MODELS):
            for b in MODELS[i + 1:]:
                ca = preds[a["key"]][1] == y; cb = preds[b["key"]][1] == y
                pairs.append([a["label"], b["label"], *mcnemar_exact(ca, cb)])
        # Holm-Korrektur, weil zehn Paare verglichen werden
        order = sorted(range(len(pairs)), key=lambda k: pairs[k][4]); holm = [0.0] * len(pairs); running = 0.0
        for rank, k in enumerate(order):
            running = max(running, min(1.0, (len(pairs) - rank) * pairs[k][4])); holm[k] = running
        for k, row in enumerate(pairs):
            w.writerow(row[:4] + [f"{row[4]:.4g}", f"{holm[k]:.4g}"])
    write_markdown(out, results, overlap, split_name)
    make_plots(out, results, split_name)


# Ergebnistabelle als Markdown speichern
def write_markdown(out, results, overlap, split_name):
    de = lambda x, d: f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    lines = [f"# Ergebnisse auf dem {split_name}", "", "| Konfiguration | Beste Epoche | Val-Macro-F1 | Richtig | Accuracy (95-%-Wilson) | Macro-F1 (95-%-Bootstrap) | Macro-Precision | Macro-Recall |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in MODELS:
        m = results[s["key"]]
        lines.append(f"| {m['label']} | {m['best_epoch']} | {de(m['val_macro_f1'], 4)} | {m['correct']}/{m['n']} | {de(m['accuracy'], 2)} % "
                     f"({de(m['accuracy_wilson95'][0], 1)}–{de(m['accuracy_wilson95'][1], 1)}) | {de(m['macro_f1'], 4)} ({de(m['macro_f1_bootstrap95'][0], 3)}–"
                     f"{de(m['macro_f1_bootstrap95'][1], 3)}) | {de(m['macro_precision'], 4)} | {de(m['macro_recall'], 4)} |")
    if overlap:
        lines += ["", f"Ohne Bilder mit pHash-Abstand ≤ {PHASH_NEAR} zu ImageNet-Bildern (mögliche Vortrainingsüberschneidung):", "",
                  "| Konfiguration | Entfernt | Accuracy | Macro-F1 |", "|---|---:|---:|---:|"]
        for s in MODELS:
            o = results[s["key"]]["ohne_imagenet_naehe"]
            lines.append(f"| {results[s['key']]['label']} | {o['entfernt']} | {de(o['accuracy'], 2)} % | {de(o['macro_f1'], 4)} |")
    (out / "ergebnisse_tabelle.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Diagramme zeichnen
def make_plots(out, results, split_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FuncFormatter
    import plot_station1_adaptation as P
    style = {r["key"]: r for r in P.RUNS}
    d = out / "diagramme"; d.mkdir(exist_ok=True)

    def save(fig, name):
        if any((d / f"{name}.{e}").exists() for e in ("png", "pdf")):  # gültige Fassung: correct_station1_diagrams.py
            print(f"{d.name}/{name} vorhanden, nicht überschrieben", flush=True); plt.close(fig); return
        fig.savefig(d / f"{name}.png", bbox_inches="tight", facecolor="white"); fig.savefig(d / f"{name}.pdf", bbox_inches="tight", facecolor="white"); plt.close(fig)

    # Validierung und Test im Vergleich
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    ys = np.arange(len(MODELS))[::-1]
    for ax, key, lab, fac, digits in ((axes[0], "accuracy", "Accuracy [%]", 1, 1), (axes[1], "macro_f1", "Macro-F1", 1, 3)):
        for y0, s in zip(ys, MODELS):
            m = results[s["key"]]; st = style[s["key"]]
            val = m["val_accuracy"] if key == "accuracy" else m["val_macro_f1"]; test = m[key]
            ci = m["accuracy_wilson95"] if key == "accuracy" else m["macro_f1_bootstrap95"]
            ax.plot(ci, [y0, y0], color=st["color"], lw=2, solid_capstyle="round")
            ax.plot(test, y0, marker=st["marker"], ms=9, color=st["color"], mec="white", ls="none", zorder=3)
            ax.plot(val, y0 + 0.28, marker="|", ms=12, mew=2, color=P.INK2, ls="none")
            ax.annotate(P.de(test, 2 if key == "accuracy" else 4), (test, y0), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=7.5)
        ax.set_yticks(ys); ax.set_yticklabels([s["label"].replace("ViT-B/16 ", "ViT ") for s in MODELS]); ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab); P.de_axis(ax, "x", digits=digits)
        ax.set_title(("Accuracy, 95-%-Wilson-Intervall" if key == "accuracy" else "Macro-F1, 95-%-Bootstrap-Intervall") + "; | = Validierung", loc="left")
    fig.suptitle(f"Ergebnisse auf dem {split_name} (bester Checkpoint nach Val-Macro-F1)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "test_vs_validierung")
    # F1 je Klasse
    fig, ax = plt.subplots(figsize=(10, 6.2), constrained_layout=True)
    offs = np.linspace(-0.3, 0.3, len(MODELS)); ys = np.arange(10)[::-1]
    for k in range(0, 10, 2):
        ax.axhspan(ys[k] - 0.5, ys[k] + 0.5, color="#f4f3f0", zorder=0)
    for o, s in zip(offs, MODELS):
        st = style[s["key"]]; f1 = [results[s["key"]]["per_class"][c]["f1"] for c in CLASS_NAMES]
        ax.plot(f1, ys + o, ls="none", marker=st["marker"], ms=6.5, color=st["color"], mec="white", label=f"{s['label']} (Macro-F1 {P.de(results[s['key']]['macro_f1'], 4)})", zorder=3)
    sup = [results[MODELS[0]["key"]]["per_class"][c]["support"] for c in CLASS_NAMES]
    ax.set_yticks(ys); ax.set_yticklabels([f"{c} (n = {n})" for c, n in zip(CLASS_NAMES, sup)]); ax.grid(axis="y", visible=False)
    lo = min(min(results[s["key"]]["per_class"][c]["f1"] for c in CLASS_NAMES) for s in MODELS)
    ax.set_xlim(max(0.0, math.floor(lo * 20) / 20 - 0.02), 1.005); ax.set_xlabel(f"F1-Score ({split_name})"); P.de_axis(ax, "x", digits=2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.suptitle(f"F1-Score je Klasse, {split_name}", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "f1_pro_klasse")
    # Konfusionsmatrizen
    cmap = LinearSegmentedColormap.from_list("blau", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    short = ["Brit. Shorthair", "Russian Blue", "Persian", "Ragdoll", "Bengal", "Samoyed", "Pomeranian", "Keeshond", "Gr. Pyrenees", "Newfoundland"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    for ax, s in zip(axes.ravel(), MODELS):
        m = results[s["key"]]; cm = np.asarray(m["confusion"]); rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if rn[i, j] > 0.55 else P.INK)
        ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.set_xticklabels(short, rotation=55, ha="right", fontsize=6.5); ax.set_yticklabels(short, fontsize=6.5)
        ax.set_xlabel("Vorhergesagte Klasse"); ax.set_ylabel("Tatsächliche Klasse"); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{m['label']}\nAcc. {P.de(m['accuracy'], 2)} %, Macro-F1 {P.de(m['macro_f1'], 4)}", loc="left", fontsize=8.5)
    axes.ravel()[-1].axis("off")
    cb = fig.colorbar(im, ax=axes.ravel()[-1], location="left", shrink=0.6); cb.set_label("Anteil der tatsächlichen Klasse (Recall je Zeile)")
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: P.de(v, 1)))
    fig.suptitle(f"Konfusionsmatrizen, {split_name}; Zahlen = Bilder", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "konfusionsmatrizen")


class RecordingOpen:
    """Zeichnet jede mit PIL geöffnete Datei auf."""
    def __enter__(self):
        import PIL.Image
        self.mod, self.original, self.opened = PIL.Image, PIL.Image.open, []
        def rec(fp, *a, **k):
            self.opened.append(str(fp)); return self.original(fp, *a, **k)
        PIL.Image.open = rec
        return self

    def __exit__(self, *exc):
        self.mod.open = self.original


# Pfade aller Testbilder, um versehentliches Öffnen zu erkennen
def test_paths():
    with open(ROOT / TEST_CSV, encoding="utf-8-sig", newline="") as h:
        return {str((ROOT / r["path"]).resolve()).lower() for r in csv.DictReader(h)}


# Kompletter Probelauf mit künstlichen Bildern, ohne Oxford-Test
def run_selftest(out):
    assert out.resolve() != (ROOT / FINAL_OUT).resolve()
    manifest = check_manifest()
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix="synth_oxford_"))
    try:
        # Künstliche Bilder erzeugen
        rng = np.random.default_rng(0); rows = []
        for ci, c in enumerate(CLASS_NAMES):
            for i in range(5):
                w, h = int(rng.integers(240, 500)), int(rng.integers(240, 500))
                arr = (rng.random((h, w, 3)) * 255).astype(np.uint8)
                p = tmp / f"synth_{ci}_{i}.jpg"; Image.fromarray(arr).save(p, quality=90)
                rows.append(dict(path=str(p), breed=c, source="synthetisch"))
        csv_path = tmp / "test.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as h:
            wr = csv.DictWriter(h, fieldnames=["path", "breed", "source"]); wr.writeheader(); wr.writerows(rows)
        # Prüfen, ob die Trainingssperre wirkt
        with no_training():
            for attempt in (lambda: torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1), lambda: torch.ones(1, requires_grad=True).sum().backward()):
                try:
                    attempt(); raise AssertionError("Trainingssperre wirkt nicht")
                except RuntimeError as e:
                    assert "gesperrt" in str(e)
        print("PASS Trainingssperre: Optimizer und backward werden abgewiesen", flush=True)
        with RecordingOpen() as rec:
            ds = PetDataset(csv_path)
            ids = [Path(r["path"]).stem for r in ds.samples]; shas = [sha256_file(r["path"]) for r in ds.samples]
            overlap = overlap_flags([r["path"] for r in ds.samples])
            results, prot = evaluate_all(ds, ids, shas, out, manifest, "synthetischen Testsatz (Selbsttest)", overlap=overlap)
        opened = {str(Path(p).resolve()).lower() for p in rec.opened}
        allowed_root = str(tmp.resolve()).lower()
        assert not opened & test_paths(), "Oxford-Testbild geöffnet"
        assert all(p.startswith(allowed_root) for p in opened), [p for p in opened if not p.startswith(allowed_root)][:3]
        print(f"PASS keine Oxford-Testbilder geöffnet; geöffnet nur {len(opened)} synthetische Bilder", flush=True)
        for name in ["vorhersagen_alle_modelle.csv", "ergebnisse_tabelle.csv", "ergebnisse_tabelle.md", "klassenmetriken.csv", "mcnemar_paarweise.csv",
                     "diagramme/test_vs_validierung.png", "diagramme/f1_pro_klasse.pdf", "diagramme/konfusionsmatrizen.png"] + [f"vorhersagen_{s['key']}.csv" for s in MODELS]:
            assert (out / name).exists(), name
        with open(out / "vorhersagen_alle_modelle.csv", encoding="utf-8", newline="") as h:
            rows_out = list(csv.DictReader(h))
        assert [r["image_id"] for r in rows_out] == ids and len(set(ids)) == len(ids)
        for s in MODELS:
            m = json.loads((out / f"metriken_{s['key']}.json").read_text(encoding="utf-8"))
            cm = np.asarray(m["confusion"]); assert cm.sum() == 50 and abs(m["accuracy"] - 100 * np.trace(cm) / 50) < 1e-9
        assert all(p["modellzustand_vorher"] == p["modellzustand_nachher"] for p in prot.values())
        print("PASS Ausgaben vollständig: Vorhersagen mit Bild-IDs, Kennzahlen, Tabelle, McNemar, Diagramme; Gewichte unverändert", flush=True)
        # Prüfen, dass der finale Lauf eine Freigabe verlangt
        for argv in (["--oxford-test"], ["--oxford-test", "--freigabe", "falsch"]):
            try:
                parse(argv); raise AssertionError("Freigabe nicht verlangt")
            except SystemExit:
                pass
        print("PASS finaler Lauf verlangt die Freigabe", RELEASE_PHRASE, flush=True)
        (out / "selbsttest_protokoll.json").write_text(json.dumps(dict(modelle=prot, synthetische_bilder=len(ids), geoeffnete_dateien=len(opened)), indent=2), encoding="utf-8")
        print("SELBSTTEST BESTANDEN", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# Prüfen, ob die Checkpoints ihre Validierungswerte wiederholen
def run_validation(out):
    assert out.resolve() != (ROOT / FINAL_OUT).resolve()
    manifest = check_manifest()
    with RecordingOpen() as rec:
        ds = PetDataset(ROOT / VAL_CSV)
        ids = [Path(r["path"]).stem for r in ds.samples]; shas = ["" for _ in ds.samples]
        evaluate_all(ds, ids, shas, out, manifest, "Validierungssplit", compare_val=True)
    opened = {str(Path(p).resolve()).lower() for p in rec.opened}
    assert not opened & test_paths(), "Oxford-Testbild geöffnet"
    print(f"PASS alle fünf Checkpoints reproduzieren ihre gespeicherte Validierungs-Konfusionsmatrix exakt; {len(opened)} Validierungsbilder, kein Testbild", flush=True)
    print("VALIDIERUNGSPRUEFUNG BESTANDEN", flush=True)


# Einmalige Auswertung auf dem Oxford-Test
def run_oxford_test():
    out = ROOT / FINAL_OUT
    # Vorhandene Ergebnisse nie überschreiben
    if out.exists():
        sys.exit(f"{FINAL_OUT} existiert bereits; die Auswertung ist einmalig und überschreibt nichts")
    manifest = check_manifest()
    assert sha256_file(ROOT / TEST_CSV) == TEST_CSV_SHA, "test.csv weicht ab"
    protected = json.loads((ROOT / PROTECTED).read_text(encoding="utf-8"))
    ds = PetDataset(ROOT / TEST_CSV)
    assert len(ds) == 999
    ids = [Path(r["path"]).stem for r in ds.samples]; assert len(set(ids)) == 999
    # Jedes Testbild gegen die geschützte Prüfsumme prüfen
    shas = []
    for r in ds.samples:
        s = sha256_file(ROOT / r["path"]); assert s == protected[str(ROOT / Path(r["path"]))], r["path"]; shas.append(s)
    print("Oxford-Test: 999 Bilder, Prüfsummen gleich den geschützten Werten", flush=True)
    overlap = overlap_flags([str(ROOT / r["path"]) for r in ds.samples])
    started = datetime.datetime.now().isoformat(timespec="seconds")
    results, prot = evaluate_all(ds, ids, shas, out, manifest, "Oxford-Test", overlap=overlap)
    with open(out / "imagenet_naehe_je_bild.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h); w.writerow(["image_id", "min_phash_abstand", "imagenet_datei", "ilsvrc_split"])
        for i, o in zip(ids, overlap):
            w.writerow([i, o["min_abstand"], o["imagenet_datei"], o["ilsvrc_split"]])
    (out / "lauf_protokoll.json").write_text(json.dumps(dict(start=started, ende=datetime.datetime.now().isoformat(timespec="seconds"), test_csv_sha256=TEST_CSV_SHA,
                                                             torch=str(torch.__version__), directml=torch_directml.device_name(0).strip("\x00"), modelle=prot,
                                                             bilder_nahe_imagenet=sum(o["min_abstand"] <= PHASH_NEAR for o in overlap),
                                                             bilder_gleiches_foto_imagenet=sum(o["min_abstand"] == 0 for o in overlap)), indent=2, ensure_ascii=False), encoding="utf-8")
    print("OXFORD-TESTAUSWERTUNG ABGESCHLOSSEN:", out, flush=True)


# Einstellungen lesen, genau ein Modus ist erlaubt
def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest-erstellen", action="store_true")
    g.add_argument("--selbsttest", action="store_true")
    g.add_argument("--validierung", action="store_true")
    g.add_argument("--oxford-test", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--freigabe", default=None, help=f"für --oxford-test genau: {RELEASE_PHRASE}")
    a = ap.parse_args(argv)
    if a.oxford_test and a.freigabe != RELEASE_PHRASE:
        ap.error(f"--oxford-test verlangt --freigabe {RELEASE_PHRASE}")
    if (a.selbsttest or a.validierung) and not a.out:
        ap.error("--selbsttest und --validierung brauchen --out")
    return a


def main(argv=None):
    a = parse(argv)
    if a.manifest_erstellen:
        build_manifest()
    elif a.selbsttest:
        run_selftest(ROOT / a.out)
    elif a.validierung:
        run_validation(ROOT / a.out)
    else:
        run_oxford_test()


if __name__ == "__main__":
    main(sys.argv[1:])
