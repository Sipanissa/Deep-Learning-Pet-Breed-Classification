"""Abbildungen für bericht/ in Druckgröße (Spalte 84 mm, Satzspiegel 175 mm), nur aus gespeicherten Ergebnissen.

Kein Modell wird gerechnet. Validierung: geprüfte Lesefunktion aus plot_station1_adaptation.py. Oxford-Test: gespeicherte
Dateien in oxford_test/; Kennzahlen, Konfusionsmatrizen und McNemar-Zählungen werden gegen die Vorhersagen je Bild geprüft.
Fotos: Trainingsbilder mit Einzellizenz, gespeicherte Augmentationen, Testbilder der gespeicherten Fehlklassifikationen.
Schreibt nur nach bericht/abbildungen/.
"""
import csv, html, json, math
from pathlib import Path

import numpy as np
from common import ROOT, sha
import plot_station1_adaptation as base
from plot_station1_adaptation import GRID, INK, INK2, MUTED, RUNS, de, de_axis, wilson
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from PIL import Image, ImageOps
from pet_dataset import CLASS_NAMES

OUT = ROOT / "bericht/abbildungen"
S1 = ROOT / "results/praktikum1_station1_adaptation"
TEST = S1 / "oxford_test"
DS = ROOT / "data/dataset_expansion"
VER, AUG = DS / "combined_versions/20261106T110038Z", DS / "augmented_versions/20261106T110038Z_15000"
COL, FULL = 3.317, 6.89  # Breite in Zoll
SHORT = ["Brit. Shorthair", "Russian Blue", "Persian", "Ragdoll", "Bengal", "Samoyed", "Pomeranian", "Keeshond", "Gr. Pyrenees", "Newfoundland"]
NAME = {"lp": "Linear Probing", "r4": "LoRA r = 4", "r8": "LoRA r = 8", "r16": "LoRA r = 16", "rn": "ResNet18"}
KEYS = list(NAME)
STYLE = {r["key"]: r for r in RUNS}
CMAP = LinearSegmentedColormap.from_list("blau", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Beispielbilder aus dem Training mit freier Lizenz und Lage des Ausschnitts
EXAMPLES = [("British Shorthair", "130023.jpg", (.5, .4)), ("Russian Blue", "197110.jpg", (.45, .5)), ("Persian", "169037.jpg", (.75, .5)),
            ("Ragdoll", "141172.jpg", (.5, .5)), ("Bengal", "033236.jpg", (.5, .1)), ("Samoyed", "cda33c11cdd068cd.jpg", (.5, .5)),
            ("Pomeranian", "39986e2e7598a83e845491f1.jpg", (.5, .5)), ("Keeshond", "113e14b92a1a8b7e5b7f5841.jpg", (.5, .5)),
            ("Great Pyrenees", "799b99ac387b9d98.jpg", (.6, .5)), ("Newfoundland", "63fc9a9493adfd51.jpg", (.5, .5))]
AUG_ORIGINAL = "b175760c1048f51181655e67.jpg"

plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "bold", "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "lines.linewidth": 1.3, "lines.markersize": 3.5, "axes.linewidth": .7,
                     "xtick.major.width": .7, "ytick.major.width": .7, "axes.titlepad": 4, "legend.handlelength": 2.0, "axes.unicode_minus": False})
REG = []


# CSV-Datei lesen, Trennzeichen automatisch erkennen
def rd(p):
    with open(p, encoding="utf-8-sig", newline="") as h:
        head = h.readline(); h.seek(0)
        return list(csv.DictReader(h, delimiter=";" if ";" in head else ","))


def res(p):
    return (Path(p) if Path(p).is_absolute() else ROOT / p).resolve()


def src(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def save(fig, name, sources, original=""):
    fig.savefig(OUT / f"{name}.pdf", facecolor="white", dpi=300)
    plt.close(fig)
    REG.append(dict(abbildung=f"{name}.pdf", originaldiagramm=original, quellen=sources))


# Gespeicherte Testergebnisse laden und nachrechnen
def load_test():
    m, pred, ids = {}, {}, None
    for k in KEYS:
        m[k] = json.loads((TEST / f"metriken_{k}.json").read_text(encoding="utf-8"))
        pred[k] = rd(TEST / f"vorhersagen_{k}.csv")
        assert len(pred[k]) == 999 and len({r["image_id"] for r in pred[k]}) == 999
        ids = ids or [r["image_id"] for r in pred[k]]
        assert ids == [r["image_id"] for r in pred[k]]
        cm = np.zeros((10, 10), int)
        for r in pred[k]:
            cm[CLASS_NAMES.index(r["klasse_wahr"]), CLASS_NAMES.index(r["klasse_vorhergesagt"])] += 1
            probs = {c: float(r["p_" + c]) for c in CLASS_NAMES}
            assert probs[r["klasse_vorhergesagt"]] >= max(probs.values()) - 1e-6
        f1, per = base.macro_f1(cm)
        assert cm.tolist() == m[k]["confusion"] and abs(f1 - m[k]["macro_f1"]) < 1e-9 and np.trace(cm) == m[k]["correct"]
        assert abs(100 * np.trace(cm) / 999 - m[k]["accuracy"]) < 1e-9
        m[k]["cm"], m[k]["per"] = cm, per
    allp = rd(TEST / "vorhersagen_alle_modelle.csv")
    for i, r in enumerate(allp):
        assert r["image_id"] == ids[i] and all(r[f"vorhersage_{k}"] == pred[k][i]["klasse_vorhergesagt"] for k in KEYS)
    for row in rd(TEST / "mcnemar_paarweise.csv"):  # McNemar-Werte nachrechnen
        a, b = (next(r["key"] for r in RUNS if r["label"] == row[x]) for x in ("modell_a", "modell_b"))
        ca = [r["richtig"] == "1" for r in pred[a]]; cb = [r["richtig"] == "1" for r in pred[b]]
        n10 = sum(x and not y for x, y in zip(ca, cb)); n01 = sum(y and not x for x, y in zip(ca, cb))
        n = n10 + n01; p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n)
        assert (n10, n01) == (int(row["nur_a_richtig"]), int(row["nur_b_richtig"])) and abs(p - float(row["p_exakt_zweiseitig"])) < 1e-3 * p + 1e-12, row
    return m, pred, allp


# Foto laden und richtig drehen
def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


# Falsch kodierte Zeichen in Titeln reparieren
def clean(s):
    s = html.unescape(s)
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def photo_axes(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(.5); s.set_color(MUTED)


# Beispielbild finden und Lizenz prüfen
def find_example(att, train, breed, name):
    hits = [r for r in train if r["breed"] == breed and Path(r["path"]).name == name]
    assert len(hits) == 1, (breed, name, len(hits))
    p = res(hits[0]["path"]); a = att[str(p).lower()]
    assert a["license"] in ("CC BY 2.0", "CC BY 3.0", "CC0")
    return p, a


# Eintrag für den Bildnachweis
def credit(fig, where, breed, p, a, change):
    return dict(abbildung=fig, bild=where, klasse=breed, split="train", datei=src(p), sha256=sha(p), titel=clean(a["title"]), urheber=clean(a["creator"]) or "nicht angegeben",
                lizenz=a["license"], lizenz_url=a["license_url"], quelle_url=a["page_url"], bearbeitung=change)


# Ein Beispielbild je Rasse
def fig_examples(att, train):
    rows = []
    fig, axes = plt.subplots(2, 5, figsize=(FULL, 3.02))
    fig.subplots_adjust(left=.035, right=.998, top=.998, bottom=.07, wspace=.05, hspace=.2)
    for ax, (breed, name, centre) in zip(axes.ravel(), EXAMPLES):
        p, a = find_example(att, train, breed, name)
        ax.imshow(ImageOps.fit(load_rgb(p), (420, 420), Image.LANCZOS, centering=centre)); photo_axes(ax)
        ax.set_xlabel(breed, fontsize=9, labelpad=3, color=INK)
        rows.append(credit("datensatz_beispiele", breed, breed, p, a, "quadratischer Ausschnitt, verkleinert"))
    axes[0, 0].set_ylabel("Katzen", fontsize=9, fontweight="bold"); axes[1, 0].set_ylabel("Hunde", fontsize=9, fontweight="bold")
    save(fig, "datensatz_beispiele", [r["datei"] for r in rows])
    return rows


# Ein Original und seine gespeicherten Varianten
def fig_augmentation(att, train):
    p, a = find_example(att, train, "Keeshond", AUG_ORIGINAL)
    var = sorted((r for r in rd(AUG / "augmentation_manifest.csv") if res(r["source_path"]) == p), key=lambda r: int(r["variant"]))
    assert len(var) == 5 and sha(p) == var[0]["source_sha256"]
    fig, axes = plt.subplots(2, 3, figsize=(COL, 2.75))
    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.16, wspace=.04, hspace=.42)

    def show(ax, path, title, sub):
        ax.imshow(ImageOps.pad(load_rgb(path), (420, 340), Image.LANCZOS, color="white")); ax.axis("off")
        ax.text(.5, -.03, title, transform=ax.transAxes, ha="center", va="top", fontsize=8, fontweight="bold")
        ax.text(.5, -.16, sub, transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=INK2, linespacing=1.1)

    show(axes[0, 0], p, "Original", "unverändert")
    files = [src(p)]
    for ax, r in zip(axes.ravel()[1:], var):
        q = AUG / r["relative_path"]
        assert sha(q) == r["output_sha256"]
        t = json.loads(r["params"])
        first = f"{'+' if t['rotation'] > 0 else '–'}{de(abs(t['rotation']), 1)}°, {de(100 * t['crop_area'], 0)} %"
        second = ", ".join((["gespiegelt"] if t["flip"] else []) + (["unscharf"] if t["blur_radius"] > 0 else []))
        show(ax, q, f"Variante {int(r['variant']) + 1}", first + ("\n" + second if second else ""))
        files.append(src(q))
    save(fig, "augmentation_beispiel", files + [src(AUG / "augmentation_manifest.csv")])
    return [credit("augmentation_beispiel", "Original und 5 gespeicherte Varianten", "Keeshond", p, a, "Varianten laut augmentation_manifest.csv (Drehung, Zuschnitt, Spiegelung, Farbe, Schärfe); verkleinert")]


# Bilder, die alle fünf Modelle falsch erkennen
def fig_errors(pred, allp):
    sel = [(i, r) for i, r in enumerate(allp) if all(r[f"vorhersage_{k}"] != r["klasse_wahr"] for k in KEYS)]
    sel.sort(key=lambda t: (CLASS_NAMES.index(t[1]["klasse_wahr"]), int(t[1]["image_id"].rsplit("_", 1)[1])))
    assert len(sel) == 8
    test_rows = {Path(r["path"]).stem: r for r in rd(AUG / "test.csv")}
    label = {("lp", "r4", "r8", "r16"): "vier ViT", ("rn",): "ResNet18", ("lp",): "Lin. Probing", ("r4", "r8", "r16", "rn"): "LoRA, ResNet18"}
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 4.2))
    fig.subplots_adjust(left=.004, right=.996, top=.995, bottom=.135, wspace=.05, hspace=.37)
    for letter, ax, (i, r) in zip("abcdefgh", axes.ravel(), sel):
        p = res(test_rows[r["image_id"]]["path"])
        assert sha(p) == r["sha256"] and test_rows[r["image_id"]]["breed"] == r["klasse_wahr"]
        ax.imshow(ImageOps.pad(load_rgb(p), (500, 390), Image.LANCZOS, color="#f4f3f0")); photo_axes(ax)
        groups = {}
        for k in KEYS:
            groups.setdefault(r[f"vorhersage_{k}"], []).append(k)
        lines = [f"alle fünf: {c}" for c in groups] if len(groups) == 1 else [f"{label[tuple(ks)]}: {c}" for c, ks in groups.items()]
        pr = float(pred["r4"][i]["p_" + r["vorhersage_r4"]])
        ax.text(0, -.04, f"({letter}) wahr: {r['klasse_wahr']}", transform=ax.transAxes, va="top", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(0, -.15, "\n".join(lines) + f"\nLoRA r = 4: p = {de(pr, 2)}", transform=ax.transAxes, va="top", fontsize=8, color="#a3290f", linespacing=1.15)
        rows.append(dict(abbildung="fehlklassifikationen", bild=f"({letter}) {r['image_id']}", klasse=r["klasse_wahr"], split="Oxford-Test", datei=src(p), sha256=r["sha256"], titel=r["image_id"],
                         urheber="nicht dokumentiert (Rechte bei den Fotoinhabern)", lizenz="CC BY-SA 4.0 für den Datensatz; Einzelbildrechte ungeklärt", lizenz_url="https://creativecommons.org/licenses/by-sa/4.0/",
                         quelle_url="https://www.robots.ox.ac.uk/~vgg/data/pets/", bearbeitung="verkleinert, auf einheitliches Format aufgefüllt, kein Zuschnitt"))
    save(fig, "fehlklassifikationen", [src(TEST / "vorhersagen_alle_modelle.csv"), src(TEST / "vorhersagen_r4.csv")] + [r["datei"] for r in rows])
    return rows


# Verlauf von Accuracy und Loss
def fig_curves(data):
    fig, axes = plt.subplots(2, 5, figsize=(FULL, 3.35), sharey="row")
    fig.subplots_adjust(left=.092, right=.995, top=.93, bottom=.185, wspace=.1, hspace=.15)
    for j, d in enumerate(data):
        ep = [r["epoch"] for r in d["log"]]; n = d["epochs"]; ms = 2.2 if n > 10 else 3.5
        for i, (tr, va) in enumerate((("train_accuracy", "val_accuracy"), ("train_loss", "val_loss"))):
            ax = axes[i, j]
            ax.plot(ep, [r[tr] for r in d["log"]], color=INK2, ls="--", lw=1.1, marker="o", ms=ms - 1)
            ax.plot(ep, [r[va] for r in d["log"]], color=d["color"], marker=d["marker"], ms=ms)
            ax.axvline(d["best_epoch"], color=MUTED, lw=.8, ls=":", zorder=1)
            ax.plot(d["best_epoch"], d["log"][d["best_epoch"] - 1][va], marker="*", ms=9, color=d["color"], mec="black", mew=.5, ls="none", zorder=5)
            ax.set_xlim(0 if n > 10 else .5, 31 if n > 10 else n + .5); ax.set_xticks([1, 10, 20, 30] if n > 10 else ([1, 4, 7, 10] if n == 10 else [1, 2, 3, 4, 5]))
            if i == 0:
                ax.set_title(NAME[d["key"]]); ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Epoche")
    axes[0, 0].set_ylim(91, 100.4); axes[0, 0].set_ylabel("Accuracy [%]"); de_axis(axes[0, 0], digits=0)
    axes[1, 0].set_yscale("log"); axes[1, 0].set_ylim(4e-5, .6); axes[1, 0].set_ylabel("Loss (log.)"); axes[1, 0].minorticks_off()
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, max(0, -int(round(math.log10(v)))))))
    fig.align_ylabels(axes[:, 0])
    fig.legend(handles=[Line2D([], [], color=INK2, ls="--", lw=1.1, marker="o", ms=2.5, label="Training (Mittel über die Epoche)"),
                        Line2D([], [], color=STYLE["lp"]["color"], marker="o", label="Validierung (Farbe je Modell)"),
                        Line2D([], [], color="white", mfc="white", mec="black", marker="*", ms=9, ls="none", label="ausgewählte Epoche")], loc="lower center", ncol=3, bbox_to_anchor=(.5, -.005))
    save(fig, "verlauf_accuracy_loss", sorted(s for d in data for s in d["sources"][:2]), "01_accuracy_verlauf, 02_loss_verlauf")


# Verlauf der Macro-F1
def fig_val_f1(data):
    fig, ax = plt.subplots(figsize=(COL, 3.1)); fig.subplots_adjust(left=.145, right=.985, top=.985, bottom=.28)
    for d in data:
        ax.plot([r["epoch"] for r in d["log"]], [r["val_macro_f1"] for r in d["log"]], color=d["color"], marker=d["marker"], ms=2.8, ls=d["ls"], label=f"{NAME[d['key']]} ({d['epochs']} Ep.)")
        ax.plot(d["best_epoch"], d["best"]["val_macro_f1"], marker="*", ms=10, color=d["color"], mec="black", mew=.5, ls="none", zorder=5)
    ax.set_xlim(.3, 30.6); ax.set_xticks([1, 5, 10, 15, 20, 25, 30]); ax.set_xlabel("Epoche"); ax.set_ylabel("Macro-F1 (Validierung)"); ax.set_ylim(.91, .981); de_axis(ax, digits=2)
    ins = ax.inset_axes([.47, .2, .52, .5])  # Ausschnitt, weil sich Sterne sonst verdecken
    for d in data[:4]:
        log = [r for r in d["log"] if r["epoch"] <= 11]
        ins.plot([r["epoch"] for r in log], [r["val_macro_f1"] for r in log], color=d["color"], marker=d["marker"], ms=2.4, lw=1)
        ins.plot(d["best_epoch"], d["best"]["val_macro_f1"], marker="*", ms=7.5, color=d["color"], mec="black", mew=.5, ls="none", zorder=5)
    ins.set_xlim(.4, 11.6); ins.set_ylim(.9655, .9785); ins.set_xticks([1, 3, 5, 10]); ins.set_yticks([.97, .975]); de_axis(ins, digits=3); ins.tick_params(labelsize=7, length=2, pad=1.5)
    for s in ins.spines.values():
        s.set_visible(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, columnspacing=1.2, labelspacing=.25, bbox_to_anchor=(.53, -.015))
    save(fig, "val_macro_f1", sorted(d["sources"][0] for d in data), "04_val_macro_f1_vergleich")


# Fünf Konfusionsmatrizen
def fig_cm(name, items, sources, original):
    height = 5.0
    fig = plt.figure(figsize=(FULL, height)); w = .268; h = w * FULL / height; x0 = .122; gx = .025; y1 = .965 - h; y0 = y1 - h - .06
    pos = [(x0 + i * (w + gx), y1) for i in range(3)] + [(x0 + i * (w + gx), y0) for i in range(2)]
    for n, ((title, cm), (x, y)) in enumerate(zip(items, pos)):
        ax = fig.add_axes([x, y, w, h]); rn = cm / cm.sum(1, keepdims=True); im = ax.imshow(rn, cmap=CMAP, vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6.8, color="white" if rn[i, j] > .55 else INK)
        ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.grid(False); ax.tick_params(length=0, pad=2)
        ax.set_xticklabels(SHORT if n >= 2 else [], rotation=90, fontsize=7); ax.set_yticklabels(SHORT if n in (0, 3) else [], fontsize=7); ax.set_title(title, fontsize=8)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRID)
    xc = x0 + 2 * (w + gx)
    cb = fig.colorbar(im, cax=fig.add_axes([xc + .01, .1, w - .02, .022]), orientation="horizontal"); cb.set_label("Anteil der wahren Klasse", fontsize=7.5)
    cb.ax.tick_params(labelsize=7); cb.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 1)))
    fig.text(.008, (y0 + y1 + h) / 2, "Wahre Klasse", rotation=90, va="center", fontsize=8); fig.text(x0 + w + gx / 2, .014, "Vorhergesagte Klasse", ha="center", fontsize=8)
    save(fig, name, sources, original)


# F1 je Klasse
def fig_per_class(data, test):
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 4.1), sharey=True); fig.subplots_adjust(left=.135, right=.985, top=.945, bottom=.165, wspace=.06)
    ys = np.arange(10)[::-1]; offs = np.linspace(.34, -.34, 5)
    for ax, per, title in ((axes[0], {d["key"]: d["per_class"] for d in data}, "Validierung (1.233 Bilder)"), (axes[1], {k: test[k]["per"] for k in KEYS}, "Oxford-Test (999 Bilder)")):
        for k in range(0, 10, 2):
            ax.axhspan(ys[k] - .5, ys[k] + .5, color="#f4f3f0", zorder=0)
        for o, k in zip(offs, KEYS):
            ax.plot(per[k], ys + o, ls="none", marker=STYLE[k]["marker"], ms=4.2, color=STYLE[k]["color"], mec="none", label=NAME[k], zorder=3)
        ax.set_xlim(.825, 1.006); ax.set_xticks([.85, .9, .95, 1]); de_axis(ax, "x", 2); ax.set_title(title); ax.set_xlabel("F1 je Klasse"); ax.grid(axis="y", visible=False); ax.set_ylim(-.55, 9.55)
        assert min(min(v) for v in per.values()) > .825
    axes[0].set_yticks(ys); axes[0].set_yticklabels(CLASS_NAMES)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(.55, -.005))
    save(fig, "f1_je_klasse", sorted(d["sources"][3] for d in data) + [src(TEST / f"metriken_{k}.json") for k in KEYS], "07_f1_pro_klasse, oxford_test/f1_pro_klasse")


# Testergebnisse mit Intervallen
def fig_test(data, test):
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.7)); fig.subplots_adjust(left=.26, right=.975, top=.95, bottom=.15, hspace=.42); ys = np.arange(5)[::-1]
    for ax, key, ci, dg, title in ((axes[0], "accuracy", "accuracy_wilson95", 2, "Accuracy [%]"), (axes[1], "macro_f1", "macro_f1_bootstrap95", 4, "Macro-F1")):
        for y, d in zip(ys, data):
            k = d["key"]; lo, hi = test[k][ci]
            ax.plot([lo, hi], [y, y], color=d["color"], lw=1.8); ax.plot(test[k][key], y, marker=d["marker"], ms=6, color=d["color"], mec="white", mew=.6, zorder=3)
            ax.plot(d["best"]["val_" + key], y, marker="|", ms=9, mew=1.6, color=INK, zorder=4)
            ax.annotate(de(test[k][key], dg), (test[k][key], y), xytext=(0, -4.5), textcoords="offset points", ha="center", va="top", fontsize=7.5)
        ax.set_yticks(ys); ax.set_yticklabels([NAME[k] for k in KEYS]); ax.set_ylim(-.8, 4.5); ax.set_title(title); ax.grid(axis="y", visible=False)
    axes[0].set_xlim(90.8, 98.8); de_axis(axes[0], "x", 0); axes[1].set_xlim(.906, .989); axes[1].set_xticks([.92, .94, .96, .98]); de_axis(axes[1], "x", 2)
    fig.legend(handles=[Line2D([], [], color=INK2, marker="o", lw=1.8, ms=5, label="Test, 95-%-Intervall"), Line2D([], [], color=INK, marker="|", ms=9, mew=1.6, ls="none", label="Validierung")], loc="lower center", ncol=2, bbox_to_anchor=(.56, -.01))
    save(fig, "test_vs_validierung", [src(TEST / f"metriken_{k}.json") for k in KEYS], "oxford_test/test_vs_validierung")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = base.load(); test, pred, allp = load_test()
    for d in data:
        assert abs(test[d["key"]]["val_macro_f1"] - d["best"]["val_macro_f1"]) < 1e-9 and test[d["key"]]["best_epoch"] == d["best_epoch"]
    att = {str(res(r["path"])).lower(): r for r in rd(VER / "attribution.csv")}; train = rd(VER / "train.csv")
    rows = fig_examples(att, train) + fig_augmentation(att, train) + fig_errors(pred, allp)
    fig_curves(data); fig_val_f1(data); fig_per_class(data, test); fig_test(data, test)
    import correct_station1_diagrams as corr  # zeichnet die übrigen drei Abbildungen
    n = len(corr.REG); corr.paper_copies(corr.load(), OUT)
    for e, original in zip(corr.REG[n:], ("05_lora_raenge", "08_dauer_und_komplexitaet", "08b_macro_f1_gegen_parameter")):
        REG.append(dict(abbildung=e["datei"] + ".pdf", originaldiagramm=original, quellen=e["quellen"]))
    fig_cm("konfusion_validierung", [(f"{NAME[d['key']]} (F1 {de(d['best']['val_macro_f1'], 4)})", d["cm"]) for d in data], [d["sources"][3] for d in data], "06_konfusionsmatrizen_uebersicht")
    fig_cm("konfusion_test", [(f"{NAME[k]} (F1 {de(test[k]['macro_f1'], 4)})", test[k]["cm"]) for k in KEYS], [src(TEST / f"metriken_{k}.json") for k in KEYS], "oxford_test/konfusionsmatrizen")
    with open(OUT / "BILDNACHWEIS.csv", "w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), delimiter=";"); w.writeheader(); w.writerows(rows)
    (OUT / "abbildungsquellen.json").write_text(json.dumps(REG, indent=1, ensure_ascii=False), encoding="utf-8")
    wrong = [sum(r[f"vorhersage_{k}"] != r["klasse_wahr"] for k in KEYS) for r in allp]
    print("Prüfungen bestanden.", len(REG), "Abbildungen; Bilder falsch bei n Modellen:", {n: wrong.count(n) for n in range(6)})
    for r in rows[:11]:
        print(r["klasse"], "|", r["urheber"], "|", r["lizenz"], "|", r["titel"], "|", r["quelle_url"])


if __name__ == "__main__":
    main()
