"""Reduzierte, einspaltige Abbildungen für bericht/cvpr2026_final (Spalte 3,25 in, Times, 8 bis 9 pt).

Gleiche gespeicherte Daten und Korrekturen wie correct_station1_diagrams.py (Stand 25.11.2026). Modelle unterscheiden sich durch
Farbe und Marker. Kein Training, kein Oxford-Testlauf. Schreibt nur die sechs unten genannten Dateien.
"""
import json
import math

import numpy as np
import plot_cvpr_final_figures as G
from plot_cvpr_final_figures import COL, INK, INK2, KEYS, MUTED, NAME, OUT, P, corr, de, de_axis
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

REG = []


# Abbildung speichern und Quellen merken
def save(fig, name, sources, original, dpi=None):
    fig.savefig(OUT.parent / f"{name}.pdf", facecolor="white", metadata={"CreationDate": None}, **(dict(dpi=dpi) if dpi else {})); plt.close(fig)
    REG.append(dict(datei=f"{name}.pdf", geprueftes_diagramm=original, quellen=sorted(set(sources))))


# Ein Bild je Rasse
def fig_breeds():
    att = {str(P.res(r["path"])).lower(): r for r in P.rd(P.VER / "attribution.csv")}; train = P.rd(P.VER / "train.csv"); files = []
    fig, axes = plt.subplots(2, 5, figsize=(COL, 1.92)); fig.subplots_adjust(left=.004, right=.996, top=.995, bottom=.135, wspace=.05, hspace=.40)
    two = {"British Shorthair": "British\nShorthair", "Russian Blue": "Russian\nBlue", "Great Pyrenees": "Great\nPyrenees", "Newfoundland": "New-\nfoundland"}
    for ax, (breed, name, centre) in zip(axes.ravel(), P.EXAMPLES):
        p, _ = P.find_example(att, train, breed, name); files.append(P.src(p))
        ax.imshow(P.ImageOps.fit(P.load_rgb(p), (300, 300), P.Image.LANCZOS, centering=centre)); P.photo_axes(ax); ax.set_xlabel(two.get(breed, breed), fontsize=8.5, labelpad=2, color=INK, linespacing=.95)
    save(fig, "bilder/rassen_uebersicht", files, "bericht/abbildungen/datensatz_beispiele.pdf (gleiche zehn Bilder)", 400)


# Verlauf von Accuracy und Loss
def fig_curves(data):
    fig, axes = plt.subplots(5, 2, figsize=(COL, 4.5)); fig.subplots_adjust(left=.142, right=.992, top=.955, bottom=.128, wspace=.31, hspace=.4)
    for (a, b), d in zip(axes, data):
        ep = [r["epoch"] for r in d["log"]]; n = d["epochs"]; ms = 2.2 if n > 10 else 3.3
        for ax, tr, va in ((a, "train_accuracy", "val_accuracy"), (b, "train_loss", "val_loss")):
            ax.plot(ep, [r[tr] for r in d["log"]], color=INK2, ls="--", lw=1, marker="o", ms=ms - .9); ax.plot(ep, [r[va] for r in d["log"]], color=d["color"], lw=1.3, marker=d["marker"], ms=ms)
            ax.axvline(d["best_epoch"], color=MUTED, lw=.8, ls=":", zorder=1); ax.plot(d["best_epoch"], d["log"][d["best_epoch"] - 1][va], marker="*", ms=8, color=d["color"], mec="black", mew=.5, ls="none", zorder=5)
            ax.set_xlim(0 if n > 10 else .6, 31 if n > 10 else n + .4); ax.set_xticks([1, 10, 20, 30] if n > 10 else ([1, 4, 7, 10] if n == 10 else [1, 2, 3, 4, 5])); ax.tick_params(pad=1.5, length=2.5)
        a.set_ylim(91, 100.7); a.set_yticks([92, 96, 100]); de_axis(a, digits=0); a.set_ylabel({"lp": "Lin. Prob."}.get(d["key"], NAME[d["key"]]), fontweight="bold", fontsize=8.5, labelpad=2)
        b.set_yscale("log"); b.set_ylim(4e-5, .6); b.set_yticks([1e-1, 1e-2, 1e-3]); b.minorticks_off(); b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, max(0, -int(round(math.log10(v)))))))
        b.tick_params(axis="y", labelsize=8)
    axes[0, 0].set_title("Accuracy [%]"); axes[0, 1].set_title("Loss (logarithmisch)"); axes[4, 0].set_xlabel("Epoche", labelpad=1); axes[4, 1].set_xlabel("Epoche", labelpad=1); fig.align_ylabels(axes[:, 0])
    fig.legend(handles=[Line2D([], [], color=INK2, ls="--", lw=1, marker="o", ms=2.4, label="Training"), Line2D([], [], color=MUTED, lw=1.3, marker="s", ms=3.3, label="Validierung"),
                        Line2D([], [], color="white", mfc="white", mec="black", marker="*", ms=8, ls="none", label="gewählte Epoche")], loc="lower center", ncol=3, bbox_to_anchor=(.5, -.008), columnspacing=1.1, handlelength=1.9, handletextpad=.5, borderaxespad=.1)
    save(fig, "diagramme/verlauf_spalte", [s for d in data for s in d["sources"][:2]], "diagramme/01_accuracy_verlauf, diagramme/02_loss_verlauf")


# Gemeinsame Beschriftung der Achsen
def rows(ax, title, lim, ticks, digits):
    ax.set_yticks(np.arange(5)[::-1]); ax.set_yticklabels([NAME[k] for k in KEYS]); ax.set_ylim(-.6, 4.6); ax.grid(axis="y", visible=False); ax.set_title(title); ax.set_xlim(*lim); ax.set_xticks(ticks); de_axis(ax, "x", digits)


# Vergleich der fünf Modelle auf der Validierung
def fig_models(data):
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL, 2.6)); fig.subplots_adjust(left=.265, right=.985, top=.93, bottom=.195, hspace=.5)
    for y, d in zip(np.arange(5)[::-1], data):
        a.plot(d["wil"], [y + .13] * 2, color=d["color"], lw=1.8); a.plot(d["best"]["val_accuracy"], y + .13, marker=d["marker"], ms=5.5, color=d["color"], mec="white", mew=.6, ls="none", zorder=3); a.plot(d["tier"], [y - .24] * 2, color=INK2, lw=1, ls="--", marker="|", ms=4.5)
        b.plot(d["f1ci"], [y, y], color=d["color"], lw=1.8); b.plot(d["best"]["val_macro_f1"], y, marker=d["marker"], ms=5.5, color=d["color"], mec="white", mew=.6, ls="none", zorder=3)
    rows(a, "Accuracy [%]", (91.9, 99.1), [92, 94, 96, 98], 0); rows(b, "Macro-F1", (.9165, .9885), [.92, .94, .96, .98], 2)
    fig.legend(handles=[Line2D([], [], color=MUTED, marker="o", lw=1.8, ms=5.5, mec="white", label="Wert, 95 % über Bilder"), Line2D([], [], color=INK2, lw=1, ls="--", marker="|", ms=4.5, label="95 % über geschätzte Tiere")], loc="lower center", ncol=2, bbox_to_anchor=(.5, -.015), columnspacing=.9, handlelength=2.2, handletextpad=.5)
    save(fig, "diagramme/modelle_validierung_spalte", [s for d in data for s in (d["sources"][3],)] + corr.SRC_F1 + corr.SRC_T, "diagramme/03, 04 (rechte Felder), 05")


# Trainierbare und gesamte Parameter
def fig_params(data):
    fig, ax = plt.subplots(figsize=(COL, 1.98)); fig.subplots_adjust(left=.265, right=.985, top=.985, bottom=.28)
    for y, d in zip(np.arange(5)[::-1], data):
        if d["key"] != "rn":
            ax.plot([d["trainable"], d["total"]], [y, y], color="#c9c8c2", lw=1.2, zorder=1); ax.plot(d["total"], y, marker="o", ms=5.5, mfc="white", mec=INK2, mew=1, ls="none", zorder=2)
        else:
            ax.plot(d["total"], y, marker="o", ms=8.5, mfc="white", mec=INK2, mew=1, ls="none", zorder=2)
        ax.plot(d["trainable"], y, marker=d["marker"], ms=5.5, color=d["color"], mec="white", mew=.6, ls="none", zorder=4)
        share = 100 * d["trainable"] / d["total"]; ax.annotate(f"{de(d['trainable'])} ({de(share, 3 if share < .1 else 2 if share < 10 else 0)} %)", (d["trainable"], y), xytext=(0, 6.5 if d["key"] == "rn" else 5), textcoords="offset points", ha="left" if d["key"] == "lp" else "center", fontsize=8.5)
        if d["key"] != "rn":
            ax.annotate(de(d["total"] / 1e6, 1) + " Mio.", (d["total"], y), xytext=(5, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5)
    ax.set_yticks(np.arange(5)[::-1]); ax.set_yticklabels([NAME[k] for k in KEYS]); ax.set_ylim(-.55, 4.8); ax.grid(axis="y", visible=False); ax.set_xscale("log"); ax.set_xlim(3e3, 4e9); ax.set_xticks([1e4, 1e6, 1e8]); ax.minorticks_off()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: corr.LOGPAR.get(v, ""))); ax.set_xlabel("Parameter (logarithmisch)", labelpad=1)
    fig.legend(handles=[Line2D([], [], marker="o", ms=5.5, color=MUTED, mec="white", ls="none", label="trainierbar (Anteil)"), Line2D([], [], marker="o", ms=5.5, mfc="white", mec=INK2, ls="none", label="gesamt")], loc="lower center", ncol=2, bbox_to_anchor=(.6, -.025), columnspacing=1.2, handletextpad=.3)
    save(fig, "diagramme/parameter_spalte", [s for d in data for s in d["sources"][2:3]], "diagramme/08 (Feld Parameter), 08b")


# Konfusionsmatrizen von LoRA r = 4 und ResNet18
def fig_cm(test):
    short = P.SHORT; H = 4.4
    fig, axes = plt.subplots(2, 1, figsize=(COL, H)); fig.subplots_adjust(left=.262, right=.992, top=.962, bottom=.18, hspace=.145)
    for ax, k in zip(axes, ("r4", "rn")):
        cm = np.array(test[k]["confusion"]); rn = cm / cm.sum(1, keepdims=True); ax.imshow(rn, cmap=P.CMAP, vmin=0, vmax=1, aspect="auto")
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8.5, color="white" if rn[i, j] > .55 else INK)
        ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.set_yticklabels(short); ax.set_xticklabels(short if k == "rn" else [], rotation=90); ax.grid(False); ax.tick_params(length=0, pad=2); ax.set_title(NAME[k], pad=3)
        ax.set_xticks(np.arange(-.5, 10), minor=True); ax.set_yticks(np.arange(-.5, 10), minor=True); ax.grid(which="minor", color=P.GRID, lw=.5); ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(P.GRID)
    save(fig, "diagramme/konfusion_spalte", [corr.src(corr.TEST / f"metriken_{k}.json") for k in ("r4", "rn")], "oxford_test/diagramme/konfusionsmatrizen (zwei von fünf Feldern)")


# Ergebnisse auf dem Oxford-Test
def fig_test(data):
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.7)); fig.subplots_adjust(left=.265, right=.985, top=.935, bottom=.245, hspace=.5); src = []
    for ax, key, ci in ((axes[0], "accuracy", "accuracy_wilson95"), (axes[1], "macro_f1", "macro_f1_bootstrap95")):
        for y, d in zip(np.arange(5)[::-1], data):
            k = d["key"]; m = json.loads((corr.TEST / f"metriken_{k}.json").read_text(encoding="utf-8")); assert m["n"] == 999; src.append(corr.src(corr.TEST / f"metriken_{k}.json"))
            ax.plot(m[ci], [y, y], color=d["color"], lw=1.8); ax.plot(m[key], y, marker=d["marker"], ms=5.5, color=d["color"], mec="white", mew=.6, ls="none", zorder=3); ax.plot(d["best"]["val_" + key], y + .34, marker="|", ms=6, mew=1.5, color=INK, ls="none")
    rows(axes[0], "Accuracy [%], 95 % Wilson", (90.9, 98.6), [92, 94, 96, 98], 0); rows(axes[1], "Macro-F1, 95 % Bootstrap über Bilder", (.908, .986), [.92, .94, .96, .98], 2)
    for ax in axes:
        ax.set_ylim(-.6, 4.75)
    fig.legend(handles=[Line2D([], [], color=MUTED, marker="o", lw=1.8, ms=5.5, mec="white", label="Oxford-Test: Wert und 95-%-Intervall"), Line2D([], [], color=INK, marker="|", ms=6.5, mew=1.5, ls="none", label="Validierung: Wert desselben Checkpoints, ohne Intervall")], loc="lower center", ncol=1, bbox_to_anchor=(.52, -.015), labelspacing=.25)
    save(fig, "diagramme/oxford_test_spalte", src + [s for d in data for s in d["sources"][3:4]], "oxford_test/diagramme/test_vs_validierung")


def main():
    data = corr.load(); test, _, _ = P.load_test()
    fig_breeds(); fig_curves(data); fig_models(data); fig_params(data); fig_cm(test); fig_test(data)
    (OUT.parent.parent / "pruefung/diagrammquellen_reduziert.json").write_text(json.dumps(REG, indent=1, ensure_ascii=False), encoding="utf-8"); print(len(REG), "Abbildungen")


if __name__ == "__main__":
    main()
