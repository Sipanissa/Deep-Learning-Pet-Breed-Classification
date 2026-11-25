"""Diagramme für Praktikum 1, Station 1: Adaptation, ausschließlich aus gespeicherten Trainingsergebnissen.

Liest nur: training_log.csv, history.json, run_config.json und den besten Checkpoint (nach Val-Macro-F1) jedes Laufs,
dazu die im Linear-Probing-Protokoll gemessene Merkmalsberechnung. Schreibt nur in --out. Kein Training, kein
Oxford-Test: Konfusionsmatrizen und F1 je Klasse stammen aus der Validierung der besten Epoche.
Vor dem Zeichnen werden Protokoll, history.json und Checkpoint gegeneinander geprüft; nichts wird interpoliert.
"""
import argparse, csv, json, math, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
import numpy as np
import torch

from common import ROOT
from pet_dataset import CLASS_NAMES

RUNS_DIR = Path("results/20261106T110038Z_15000/seed_42")
LP_LOG = Path("results/praktikum1_station1_adaptation/vit_linear_probing_schritt3/training.log")
# Feste Farben, Marker und Linienarten je Modell
RUNS = [
    dict(key="lp", folder="vit_b16_linear_probe", ckpt="best_head.pt", label="ViT-B/16 Linear Probing", short="ViT LP", color="#2a78d6", marker="o", ls="-"),
    dict(key="r4", folder="vit_b16_lora_r4", ckpt="best_lora.pt", label="ViT-B/16 LoRA r = 4", short="LoRA r4", color="#eb6834", marker="s", ls="-"),
    dict(key="r8", folder="vit_b16_lora_r8", ckpt="best_lora.pt", label="ViT-B/16 LoRA r = 8", short="LoRA r8", color="#1baf7a", marker="^", ls="-"),
    dict(key="r16", folder="vit_b16_lora_r16", ckpt="best_lora.pt", label="ViT-B/16 LoRA r = 16", short="LoRA r16", color="#c98500", marker="D", ls="-"),
    dict(key="rn", folder="resnet18_finetune", ckpt="best_resnet18.pt", label="ResNet18 Fine-Tuning", short="ResNet18", color="#d55181", marker="v", ls="--"),
]
HINWEIS = ("Alle Werte aus den genannten Dateien gelesen; Wilson-Intervalle aus den Zählwerten der Konfusionsmatrix berechnet; Parameter gesamt bei LoRA = ViT-B/16 mit Kopf + LoRA-Parameter aus run_config.json. "
           "Macro-F1-Intervalle: gespeicherter Bootstrap über einzelne Bilder (validierungspruefung/metriken_*.json). Intervalle über geschätzte Tiere: Tieridentifikation aus dem Namen im Bildtitel geschätzt, nicht verifiziert "
           "(tools/animal_bootstrap_validation.py). Alle Intervalle beschreiben die Stichprobenunsicherheit bei festem Checkpoint, nicht die Schwankung zwischen Trainingsläufen (ein Lauf je Konfiguration, Seed 42). "
           "03, 04, 05, 08 und 08b zeichnet tools/correct_station1_diagrams.py, die übrigen tools/plot_station1_adaptation.py.")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
CLASS_DE = CLASS_NAMES

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "lines.markersize": 5,
    "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
})


def de(x, digits=None):
    """Zahl mit Dezimalkomma und Tausenderpunkt."""
    if digits is None:
        s = f"{x:,}" if float(x).is_integer() else f"{x:,.4g}"
    else:
        s = f"{x:,.{digits}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# Achse mit deutschem Zahlenformat beschriften
def de_axis(ax, axis="y", digits=None):
    f = FuncFormatter(lambda v, _: de(v, digits) if digits is not None else de(round(v, 6)))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


# F1 je Klasse und Mittelwert berechnen
def macro_f1(conf):
    c = np.asarray(conf, dtype=float); tp = np.diag(c); p = c.sum(0); r = c.sum(1)
    f1 = np.where(p + r > 0, 2 * tp / np.maximum(p + r, 1e-12), 0.0)
    return float(f1.mean()), f1


# Wilson-Intervall für die Accuracy in Prozent
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


# Gespeicherte Ergebnisse aller fünf Läufe laden
def load():
    data = []
    for run in RUNS:
        folder = ROOT / RUNS_DIR / run["folder"]
        with open(folder / "training_log.csv", encoding="utf-8", newline="") as h:
            log = [{k: (v == "True" if k == "best" else float(v)) for k, v in row.items()} for row in csv.DictReader(h)]
        hist = json.loads((folder / "history.json").read_text(encoding="utf-8"))
        conf = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
        ck = torch.load(folder / run["ckpt"], map_location="cpu", weights_only=True)
        m = ck["metrics"]
        # Protokoll, Verlauf und Checkpoint müssen zusammenpassen
        assert len(log) == len(hist["history"]) == conf["epochs"] == hist["epochs"], run["key"]
        for a, b in zip(log, hist["history"]):
            for k in ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1"):
                assert abs(a[k] - b[k]) < 1e-5 * max(1, abs(b[k])), (run["key"], k)
        best_epoch = max(log, key=lambda r: r["val_macro_f1"])["epoch"]
        assert int(best_epoch) == ck["epoch"] == hist["best_epoch"] == m["epoch"], run["key"]
        cm = np.asarray(m["confusion"]); n = int(cm.sum())
        assert n == 1233 and cm.shape == (10, 10)
        f1, per = macro_f1(cm)
        assert abs(f1 - m["val_macro_f1"]) < 1e-9 and abs(100 * np.trace(cm) / n - m["val_accuracy"]) < 1e-9
        assert all(abs(per[i] - m["val_per_class_f1"][c]) < 1e-9 for i, c in enumerate(CLASS_NAMES))
        if "best_confusion" in hist:
            assert hist["best_confusion"] == m["confusion"]
        d = dict(run, log=log, hist=hist, conf=conf, best=m, cm=cm, per_class=per, best_epoch=int(best_epoch), epochs=len(log))
        # Parameterzahl und Dauer bestimmen
        if run["key"] == "lp":
            d["trainable"] = conf["trainable_parameters"]; d["total"] = conf["parameters"]
            text = (ROOT / LP_LOG).read_text(encoding="utf-8")
            feats = [int(s) for s in re.findall(r"Merkmale berechnet: \d+ Bilder in (\d+)s", text)]
            assert len(feats) == 2, feats
            d["feature_seconds"] = sum(feats); d["head_seconds"] = sum(r["seconds"] for r in log)
            d["seconds_total"] = d["feature_seconds"] + d["head_seconds"]
            d["epoch_seconds"] = [r["seconds"] for r in log]
            d["scaling"] = None
        else:
            if run["key"] == "rn":
                d["trainable"] = conf["resnet18"]["trainable_parameters"]; d["total"] = conf["resnet18"]["parameters"]; d["scaling"] = None
            else:
                lora = conf["lora"]; d["trainable"] = lora["trainable_parameters"]
                d["total"] = data[0]["total"] + lora["lora_parameters"]  # ViT mit Kopf plus LoRA
                d["rank"], d["alpha"], d["scaling"] = lora["rank"], lora["alpha"], lora["scaling"]
                assert d["scaling"] == d["alpha"] / d["rank"]
            d["seconds_total"] = hist["trained_seconds"]
            assert abs(d["seconds_total"] - sum(r["seconds"] for r in log)) < 1.0
            d["epoch_seconds"] = [r["seconds"] for r in log]
        d["sources"] = [str(RUNS_DIR / run["folder"] / f).replace("\\", "/") for f in ("training_log.csv", "history.json", "run_config.json", run["ckpt"])]
        data.append(d)
    return data


# Kurzer Text zum Trainingsbudget
def budget(d):
    if d["key"] == "lp":
        return f"{d['epochs']} Epochen, nur Kopf (Merkmale zwischengespeichert)"
    if d["key"] == "rn":
        return f"{d['epochs']} Epochen, alle Schichten"
    return f"{d['epochs']} Epochen, α/r = {de(d['scaling'])}"


# Diagramm als PNG und PDF speichern
def save(fig, out, name, sources, registry, description):
    fig.savefig(out / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    registry.append(dict(datei=name, beschreibung=description, quellen=sorted(set(sources))))


# Beste Epoche mit einem Stern markieren
def mark_best(ax, d, ykey, factor=1.0):
    row = d["log"][d["best_epoch"] - 1]
    ax.axvline(d["best_epoch"], color=MUTED, lw=1, ls=":", zorder=1)
    ax.plot(d["best_epoch"], row[ykey] * factor, marker="*", ms=12, color=d["color"], mec="white", mew=1.2, zorder=5, ls="none")


# Verlauf von Accuracy oder Loss zeichnen
def fig_curves(data, out, reg, kind):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes = axes.ravel()
    for ax, d in zip(axes, data):
        ep = [r["epoch"] for r in d["log"]]
        if kind == "acc":
            ax.plot(ep, [r["train_accuracy"] for r in d["log"]], color=INK2, lw=1.6, ls="--", marker="o", ms=3.5, label="Training (Mittel über die Epoche)")
            ax.plot(ep, [r["val_accuracy"] for r in d["log"]], color=d["color"], marker=d["marker"], ms=4.5, label="Validierung")
            mark_best(ax, d, "val_accuracy"); ax.set_ylim(90, 100.6); ax.set_ylabel("Accuracy [%]"); de_axis(ax, digits=0)
        else:
            ax.plot(ep, [r["train_loss"] for r in d["log"]], color=INK2, lw=1.6, ls="--", marker="o", ms=3.5, label="Training (Mittel über die Epoche)")
            ax.plot(ep, [r["val_loss"] for r in d["log"]], color=d["color"], marker=d["marker"], ms=4.5, label="Validierung")
            mark_best(ax, d, "val_loss"); ax.set_yscale("log"); ax.set_ylim(3e-5, 0.6); ax.set_ylabel("Cross-Entropy-Loss (log. Skala)")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, max(0, -int(math.floor(math.log10(v))))) if v > 0 else "0"))
        ax.set_xlim(0.5, d["epochs"] + 0.5); ax.set_xticks([1, 5, 10, 15, 20, 25, 30] if d["epochs"] > 10 else range(1, d["epochs"] + 1))
        ax.set_xlabel("Epoche"); ax.set_title(d["label"], loc="left", pad=16)
        ax.text(0.0, 1.02, f"Budget: {budget(d)}", transform=ax.transAxes, fontsize=8, color=INK2, va="bottom", ha="left")
    ax = axes[-1]; ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK2, lw=1.6, ls="--", marker="o", ms=3.5, label="Training (laufendes Mittel über die Epoche)")] +               [Line2D([], [], color=d["color"], marker=d["marker"], ms=4.5, label=f"Validierung {d['short']}") for d in data]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=7.5, ncol=2, columnspacing=1.0)
    ax.text(0, 0.70, "Lesehilfe", fontweight="bold", va="top")
    ax.text(0, 0.62, "★  beste Epoche nach Val-Macro-F1 (gespeicherter Checkpoint)\n"
            "⋯  senkrechte Linie: dieselbe Epoche\n\n"
            "Unterschiedliche Trainingsbudgets:\n"
            "  Linear Probing 30 Epochen, LoRA r = 8 und ResNet18 10,\n"
            "  LoRA r = 4 und r = 16 je 5 Epochen.\n\n"
            "LoRA mit α = 16 für alle Ränge, daher Skalierung\n"
            "  α/r = 4 (r = 4), 2 (r = 8), 1 (r = 16).\n\n"
            "Trainingswerte sind laufende Mittel während der Epoche\n"
            "  (mit Dropout bei LoRA, BatchNorm im Trainingsmodus bei ResNet18),\n"
            "  Validierungswerte nach der Epoche auf 1.233 Bildern.\n"
            "Ein Lauf je Konfiguration (Seed 42).", va="top", fontsize=8, color=INK2, linespacing=1.4)
    title = "Training- und Validierungs-Accuracy je Epoche" if kind == "acc" else "Training- und Validierungs-Loss je Epoche"
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    name = "01_accuracy_verlauf" if kind == "acc" else "02_loss_verlauf"
    save(fig, out, name, [s for d in data for s in d["sources"][:2]], reg, title + "; eigene x-Achse je Lauf entsprechend dem Budget")


# Konfusionsmatrizen zeichnen
def fig_confusion(data, out, reg):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blau", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    short = ["Brit. Shorthair", "Russian Blue", "Persian", "Ragdoll", "Bengal", "Samoyed", "Pomeranian", "Keeshond", "Gr. Pyrenees", "Newfoundland"]

    def draw(ax, d, small=False):
        cm = d["cm"]; rn = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(rn, cmap=cmap, vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6 if small else 7.5, color="white" if rn[i, j] > 0.55 else INK)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(short, rotation=55, ha="right", fontsize=6.5 if small else 8); ax.set_yticklabels(short, fontsize=6.5 if small else 8)
        ax.set_xlabel("Vorhergesagte Klasse"); ax.set_ylabel("Tatsächliche Klasse"); ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"{d['label']}\nbeste Epoche {d['best_epoch']} von {d['epochs']}, Acc. {de(d['best']['val_accuracy'], 2)} %, Macro-F1 {de(d['best']['val_macro_f1'], 4)}", loc="left", fontsize=8.5 if small else 10)
        return im

    for d in data:
        fig, ax = plt.subplots(figsize=(6.6, 6), constrained_layout=True)
        im = draw(ax, d)
        cb = fig.colorbar(im, ax=ax, shrink=0.75); cb.set_label("Anteil der tatsächlichen Klasse (Recall je Zeile)"); cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 1)))
        fig.suptitle("Konfusionsmatrix, Validierung (1.233 Bilder)", x=0.01, ha="left", fontsize=11, fontweight="bold")
        save(fig, out, f"06_konfusionsmatrix_{d['key']}", [d["sources"][3]], reg, f"Konfusionsmatrix {d['label']}, Validierung, bester Checkpoint; Zahlen = Bilder, Farbe = Zeilenanteil")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    for ax, d in zip(axes.ravel(), data):
        im = draw(ax, d, small=True)
    axes.ravel()[-1].axis("off")
    cb = fig.colorbar(im, ax=axes.ravel()[-1], location="left", shrink=0.6); cb.set_label("Anteil der tatsächlichen Klasse (Recall je Zeile)"); cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 1)))
    fig.suptitle("Konfusionsmatrizen der besten Checkpoints, Validierung (1.233 Bilder); Zahlen = Bilder", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, out, "06_konfusionsmatrizen_uebersicht", [d["sources"][3] for d in data], reg, "Alle fünf Konfusionsmatrizen nebeneinander")


# F1 je Klasse zeichnen
def fig_per_class(data, out, reg):
    support = data[0]["cm"].sum(1)
    fig, ax = plt.subplots(figsize=(10, 6.2), constrained_layout=True)
    offs = np.linspace(-0.3, 0.3, len(data))
    ys = np.arange(10)[::-1]
    for k in range(10):
        if k % 2 == 0:
            ax.axhspan(ys[k] - 0.5, ys[k] + 0.5, color="#f4f3f0", zorder=0)
    for o, d in zip(offs, data):
        ax.plot(d["per_class"], ys + o, ls="none", marker=d["marker"], ms=6.5, color=d["color"], mec="white", mew=0.8, label=f"{d['label']} (Macro-F1 {de(d['best']['val_macro_f1'], 4)})", zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([f"{c} (n = {int(n)})" for c, n in zip(CLASS_NAMES, support)])
    ax.set_xlim(0.80, 1.005); ax.set_xticks(np.arange(0.80, 1.001, 0.02)); ax.set_xlabel("F1-Score auf der Validierung (bester Checkpoint)"); de_axis(ax, "x", digits=2)
    ax.grid(axis="y", visible=False); ax.set_ylim(-0.6, 9.6)
    ax.legend(loc="lower left", fontsize=8, bbox_to_anchor=(0, 1.0), ncol=2)
    fig.suptitle("F1-Score je Klasse für die fünf Konfigurationen", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02, "n = Anzahl Validierungsbilder je Klasse. x-Achse beginnt bei 0,80, um Unterschiede sichtbar zu machen. Ein Bild entspricht bei Keeshond (n = 53) bereits rund 1-2 F1-Prozentpunkten.", fontsize=7.5, color=INK2)
    save(fig, out, "07_f1_pro_klasse", [d["sources"][3] for d in data], reg, "F1 je Klasse, Validierung, bester Checkpoint")


# Kennzahlen als CSV speichern
def write_tables(data, out):
    with open(out / "kennzahlen_beste_checkpoints.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h, delimiter=";")
        w.writerow(["konfiguration", "epochen_budget", "beste_epoche", "val_accuracy_prozent", "richtig_von_1233", "wilson95_unten", "wilson95_oben", "val_macro_f1", "val_loss",
                    "trainierbare_parameter", "parameter_gesamt", "lora_rank", "lora_alpha", "alpha_durch_rank", "trainingsdauer_s", "mittel_s_je_epoche"])
        for d in data:
            k = int(np.trace(d["cm"])); lo, hi = wilson(k, 1233)
            w.writerow([d["label"], d["epochs"], d["best_epoch"], round(d["best"]["val_accuracy"], 4), k, round(lo, 2), round(hi, 2), round(d["best"]["val_macro_f1"], 6), round(d["best"]["val_loss"], 6),
                        d["trainable"], d["total"], d.get("rank", ""), d.get("alpha", ""), d["scaling"] if d["scaling"] else "", round(d["seconds_total"], 1), round(float(np.mean(d["epoch_seconds"])), 1)])
    with open(out / "f1_je_klasse.csv", "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h, delimiter=";"); w.writerow(["klasse", "n_validierung"] + [d["label"] for d in data])
        for i, c in enumerate(CLASS_NAMES):
            w.writerow([c, int(data[0]["cm"][i].sum())] + [round(float(d["per_class"][i]), 4) for d in data])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/praktikum1_station1_adaptation/diagramme"); a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    data = load(); reg = []
    fig_curves(data, out, reg, "acc")
    fig_curves(data, out, reg, "loss")
    fig_confusion(data, out, reg)
    fig_per_class(data, out, reg)
    import correct_station1_diagrams as corr  # zeichnet 03, 04, 05, 08 und 08b
    reg += corr.draw_diagrams(out); reg.sort(key=lambda e: e["datei"].split("_")[0])
    write_tables(data, out)
    (out / "datenquellen.json").write_text(json.dumps(dict(hinweis=HINWEIS, diagramme=reg), indent=2, ensure_ascii=False), encoding="utf-8")
    print("Konsistenzprüfung bestanden; erstellt:", len(reg), "Diagramme in", out)


if __name__ == "__main__":
    main()
