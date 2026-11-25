"""Druckfassungen der geprüften Diagramme für das CVPR-2026-Layout (Spalte 3,25 in, Satzspiegel 6,875 in).

Gleiche gespeicherte Daten und gleiche Korrekturen wie correct_station1_diagrams.py (03, 04, 05, Oxford test_vs_validierung),
gezeichnet in Originalgröße, damit die Schrift im Druck 7,5 bis 8 pt hat. Modelle unterscheiden sich durch Farbe, Marker und
Linienstil. Kein Training, kein Oxford-Testlauf. Schreibt nur nach bericht/cvpr2026/overleaf_projekt/diagramme.
"""
import json

import numpy as np
import correct_station1_diagrams as corr
import plot_paper_figures as P
from plot_station1_adaptation import INK, INK2, MUTED, de, de_axis
from common import ROOT
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Ausgabeordner und Breiten in Zoll
OUT = ROOT / "bericht/cvpr2026/overleaf_projekt/diagramme"
FULL, COL = 6.875, 3.25
NAME = P.NAME
KEYS = P.KEYS
# Linienart je Modell, damit man nicht nur auf Farben angewiesen ist
LS = {"lp": "-", "r4": (0, (5, 1.6)), "r8": (0, (5, 1.3, 1, 1.3)), "r16": (0, (1, 1.1)), "rn": (0, (3, 1, 1, 1, 1, 1))}
REG = []
plt.rcParams["lines.solid_capstyle"] = "butt"; plt.rcParams["lines.dash_capstyle"] = "butt"  # Linien enden genau am Wert


# Abbildung speichern und Quellen merken
def save(fig, name, sources, original):
    fig.savefig(OUT / f"{name}.pdf", facecolor="white", metadata={"CreationDate": None}); plt.close(fig)
    REG.append(dict(datei=f"{name}.pdf", geprueftes_diagramm=original, quellen=sorted(set(sources))))


# Macro-F1 auf der Validierung
def fig_macro_f1(data):
    fig = plt.figure(figsize=(FULL, 2.4)); ax = fig.add_axes([.072, .262, .515, .66]); bx = fig.add_axes([.725, .262, .265, .66])
    for d in data:
        k = d["key"]
        ax.plot([r["epoch"] for r in d["log"]], [r["val_macro_f1"] for r in d["log"]], color=d["color"], ls=LS[k], lw=1.3, marker=d["marker"], ms=3.4, label=NAME[k])
        ax.plot(d["best_epoch"], d["best"]["val_macro_f1"], marker="*", ms=9, color=d["color"], mec="black", mew=.5, ls="none", zorder=5)
    ax.set_xlim(.3, 30.6); ax.set_xticks([1, 5, 10, 15, 20, 25, 30]); ax.set_ylim(.91, .981); de_axis(ax, digits=2); ax.set_xlabel("Epoche", labelpad=1.5); ax.set_ylabel("Macro-F1 (Validierung)")
    ax.set_title("Verlauf (Stern: gewählte Epoche)", loc="left")
    ins = ax.inset_axes([.47, .13, .51, .52])
    for d in data[1:4]:
        log = [r for r in d["log"] if r["epoch"] <= 5]
        ins.plot([r["epoch"] for r in log], [r["val_macro_f1"] for r in log], color=d["color"], ls=LS[d["key"]], lw=1.1, marker=d["marker"], ms=3.2)
        ins.plot(d["best_epoch"], d["best"]["val_macro_f1"], marker="*", ms=7.5, color=d["color"], mec="black", mew=.5, ls="none", zorder=5)
    ins.set_xlim(.6, 5.4); ins.set_ylim(.9645, .9788); ins.set_xticks([1, 2, 3, 4, 5]); ins.yaxis.set_major_locator(MultipleLocator(.005)); de_axis(ins, digits=3); ins.tick_params(labelsize=7.5, length=2, pad=1.5)
    ins.set_title("LoRA, Epochen 1 bis 5", loc="left", fontsize=7.5, fontweight="normal", pad=2)
    for s in ins.spines.values():
        s.set_visible(True)
    ys = np.arange(5)[::-1]
    for y, d in zip(ys, data):
        bx.plot(d["f1ci"], [y, y], color=d["color"], lw=1.6); bx.plot(d["best"]["val_macro_f1"], y, marker=d["marker"], ms=5.5, color=d["color"], mec="white", mew=.6, ls="none", zorder=3)
        bx.annotate(de(d["best"]["val_macro_f1"], 4), (d["best"]["val_macro_f1"], y), xytext=(0, 4.6), textcoords="offset points", ha="center", fontsize=7.5)
    bx.set_yticks(ys); bx.set_yticklabels([NAME[d["key"]] for d in data]); bx.set_ylim(-.5, 4.95); bx.grid(axis="y", visible=False); bx.set_xlim(.916, .989); bx.set_xticks([.92, .94, .96, .98]); de_axis(bx, "x", 2)
    bx.set_xlabel("Macro-F1 (Validierung)", labelpad=1.5); bx.set_title("Gewählter Checkpoint, 95-%-Intervall", fontsize=7.8, x=.40)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(.5, -.015), columnspacing=1.5, handlelength=3.4)
    save(fig, "04_val_macro_f1_druck", [s for d in data for s in (d["sources"][0], d["sources"][3])] + corr.SRC_F1, "diagramme/04_val_macro_f1_vergleich")


# Vergleich der LoRA-Ränge
def fig_lora(data):
    lora = data[1:4]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL, 2.45)); fig.subplots_adjust(left=.085, right=.995, top=.925, bottom=.355, wspace=.28)
    note = lambda ax, i, t, y, dy=3: ax.annotate(t, (i, y), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=7.5, bbox=dict(facecolor="white", edgecolor="none", pad=.2))
    for i, d in enumerate(lora):
        a.plot([i + .2] * 2, d["tier"], color=INK2, lw=1, ls="--", marker="_", ms=5); a.plot([i, i], d["wil"], color=d["color"], lw=1.8); a.plot(i, d["best"]["val_accuracy"], marker=d["marker"], ms=6, color=d["color"], mec="white", ls="none", zorder=3); note(a, i, de(d["best"]["val_accuracy"], 2), d["tier"][1])
        b.plot([i, i], d["f1ci"], color=d["color"], lw=1.8); b.plot(i, d["best"]["val_macro_f1"], marker=d["marker"], ms=6, color=d["color"], mec="white", ls="none", zorder=3); note(b, i, de(d["best"]["val_macro_f1"], 4), d["f1ci"][1])
    a.set_title("Accuracy [%]"); a.set_ylim(94, 100); a.set_yticks([94, 96, 98, 100]); de_axis(a, digits=0)
    b.set_title("Macro-F1"); b.set_ylim(.94, 1); b.set_yticks([.94, .96, .98, 1]); de_axis(b, digits=2)
    for ax in (a, b):
        ax.set_xlim(-.55, 2.55); ax.set_xticks(range(3)); ax.grid(axis="x", visible=False); ax.set_xticklabels([f"r = {d['rank']}\nα/r = {d['scaling']:.0f}\n{d['epochs']} Ep." for d in lora])
    fig.legend(handles=[Line2D([], [], color=MUTED, lw=1.8, label="95 % über Bilder"), Line2D([], [], color=INK2, lw=1, ls="--", marker="_", ms=5, label="95 % über geschätzte Tiere")], loc="lower center", ncol=2, bbox_to_anchor=(.53, -.018), columnspacing=1.2, handlelength=2.4)
    save(fig, "05_lora_raenge_druck", [s for d in lora for s in d["sources"]] + corr.SRC_T + corr.SRC_F1[1:4], "diagramme/05_lora_raenge (ohne Felder Parameter und Dauer; Werte im Text und in Tabelle 2)")


# Ergebnisse auf dem Oxford-Test
def fig_test(data):
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.05)); fig.subplots_adjust(left=.265, right=.985, top=.943, bottom=.19, hspace=.42); ys = np.arange(5)[::-1]; src = []
    for ax, key, ci, dg, title in ((axes[0], "accuracy", "accuracy_wilson95", 2, "Accuracy [%], 95 % Wilson"), (axes[1], "macro_f1", "macro_f1_bootstrap95", 4, "Macro-F1, 95 % Bootstrap über Bilder")):
        for y, d in zip(ys, data):
            k = d["key"]; m = json.loads((corr.TEST / f"metriken_{k}.json").read_text(encoding="utf-8")); assert m["n"] == 999; src.append(corr.src(corr.TEST / f"metriken_{k}.json"))
            ax.plot(m[ci], [y, y], color=d["color"], lw=1.8); ax.plot(m[key], y, marker=d["marker"], ms=5.5, color=d["color"], mec="white", mew=.6, ls="none", zorder=3); ax.plot(d["best"]["val_" + key], y + .34, marker="|", ms=6, mew=1.5, color=INK, ls="none")
            left = k != "rn"; ax.annotate(de(m[key], dg), (m[ci][0 if left else 1], y), xytext=(-4 if left else 4, 0), textcoords="offset points", ha="right" if left else "left", va="center", fontsize=7.5)
        ax.set_yticks(ys); ax.set_yticklabels([NAME[k] for k in KEYS]); ax.set_ylim(-.6, 4.75); ax.set_title(title); ax.grid(axis="y", visible=False)
    axes[0].set_xlim(90.9, 98.6); de_axis(axes[0], "x", 0); axes[1].set_xlim(.908, .986); axes[1].set_xticks([.92, .94, .96, .98]); de_axis(axes[1], "x", 2)
    fig.legend(handles=[Line2D([], [], color=MUTED, marker="o", lw=1.8, ms=5.5, mec="white", label="Oxford-Test: Wert und 95-%-Intervall"), Line2D([], [], color=INK, marker="|", ms=6.5, mew=1.5, ls="none", label="Validierung: Wert desselben Checkpoints, ohne Intervall")],
               loc="lower center", ncol=1, bbox_to_anchor=(.52, -.015), labelspacing=.25)
    save(fig, "oxford_test_vs_validierung_druck", src + [s for d in data for s in d["sources"][3:4]], "oxford_test/diagramme/test_vs_validierung")


# Konfusionsmatrizen von LoRA r = 4 und ResNet18
def fig_cm(test):
    H = 3.1; fig = plt.figure(figsize=(FULL, H)); w = .306; h = w * FULL / H; y = .941 - h
    for n, (x, k) in enumerate(((.15, "r4"), (.15 + w + .035, "rn"))):
        cm = np.array(test[k]["confusion"]); ax = fig.add_axes([x, y, w, h]); rn = cm / cm.sum(1, keepdims=True); im = ax.imshow(rn, cmap=P.CMAP, vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                if cm[i, j]:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.5, color="white" if rn[i, j] > .55 else INK)
        ax.set_xticks(range(10)); ax.set_yticks(range(10)); ax.grid(False); ax.tick_params(length=0, pad=2); ax.set_xticklabels([s.replace("Brit. Shorthair", "Brit. Shorth.").replace("Newfoundland", "Newfoundl.").replace("Gr. Pyrenees", "Gr. Pyren.") for s in P.SHORT], rotation=90, fontsize=7.5); ax.set_yticklabels(P.SHORT if n == 0 else [], fontsize=7.5)
        ax.set_title(f"{NAME[k]}: Acc. {de(test[k]['accuracy'], 2)} %, Macro-F1 {de(test[k]['macro_f1'], 4)}", fontsize=8)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(P.GRID)
    cb = fig.colorbar(im, cax=fig.add_axes([.86, y, .016, h])); cb.set_label("Anteil der wahren Klasse", fontsize=7.5); cb.ax.tick_params(labelsize=7.5); cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(v, 1)))
    fig.text(.03, y + h / 2, "Wahre Klasse", rotation=90, va="center", fontsize=8); fig.text(.15 + w + .0175, .006, "Vorhergesagte Klasse", ha="center", fontsize=8)
    save(fig, "oxford_konfusion_druck", [corr.src(corr.TEST / f"metriken_{k}.json") for k in ("r4", "rn")], "oxford_test/diagramme/konfusionsmatrizen (zwei von fünf Feldern)")


def fig_errors(pred, allp):
    """Wie plot_paper_figures.fig_errors (gleiche acht Bilder, gleiche Prüfungen), flacheres Format."""
    sel = [(i, r) for i, r in enumerate(allp) if all(r[f"vorhersage_{k}"] != r["klasse_wahr"] for k in KEYS)]
    sel.sort(key=lambda t: (P.CLASS_NAMES.index(t[1]["klasse_wahr"]), int(t[1]["image_id"].rsplit("_", 1)[1]))); assert len(sel) == 8
    rows = {P.Path(r["path"]).stem: r for r in P.rd(P.AUG / "test.csv")}
    label = {("lp", "r4", "r8", "r16"): "vier ViT", ("rn",): "ResNet18", ("lp",): "Lin. Probing", ("r4", "r8", "r16", "rn"): "LoRA, ResNet18"}; files = []
    fig, axes = plt.subplots(2, 4, figsize=(FULL, 3.16)); fig.subplots_adjust(left=.002, right=.998, top=.998, bottom=.155, wspace=.04, hspace=.49)
    for letter, ax, (i, r) in zip("abcdefgh", axes.ravel(), sel):
        f = P.res(rows[r["image_id"]]["path"]); assert P.sha(f) == r["sha256"] and rows[r["image_id"]]["breed"] == r["klasse_wahr"]; files.append(P.src(f))
        ax.imshow(P.ImageOps.pad(P.load_rgb(f), (520, 310), P.Image.LANCZOS, color="#f4f3f0"), aspect="equal"); P.photo_axes(ax); groups = {}
        for k in KEYS:
            groups.setdefault(r[f"vorhersage_{k}"], []).append(k)
        lines = [f"alle fünf: {c}" for c in groups] if len(groups) == 1 else [f"{label[tuple(ks)]}: {c}" for c, ks in groups.items()]
        pr = float(pred["r4"][i]["p_" + r["vorhersage_r4"]])
        ax.text(0, -.04, f"({letter}) wahr: {r['klasse_wahr']}", transform=ax.transAxes, va="top", fontsize=8, fontweight="bold", color=INK)
        ax.text(0, -.165, "\n".join(lines) + f"\nLoRA r = 4: p = {de(pr, 2)}", transform=ax.transAxes, va="top", fontsize=7.5, color="#a3290f", linespacing=1.1)
    fig.savefig(OUT.parent / "bilder/fehlklassifikationen.pdf", facecolor="white", dpi=300, metadata={"CreationDate": None}); plt.close(fig)
    REG.append(dict(datei="../bilder/fehlklassifikationen.pdf", geprueftes_diagramm="bericht/abbildungen/fehlklassifikationen.pdf (gleicher Inhalt)", quellen=[P.src(P.TEST / "vorhersagen_alle_modelle.csv"), P.src(P.TEST / "vorhersagen_r4.csv")] + files))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = corr.load(); test, pred, allp = P.load_test()
    fig_macro_f1(data); fig_lora(data); fig_test(data); fig_cm(test); fig_errors(pred, allp)

    # Verlaufskurven in der passenden Größe speichern
    def save_curves(fig, name, sources, original):
        fig.set_size_inches(FULL, 2.85); fig.subplots_adjust(left=.085, right=.995, top=.92, bottom=.205, wspace=.1, hspace=.17)
        fig.axes[0].set_yticks([92, 94, 96, 98, 100])
        fig.legends[0].texts[1].set_text("Validierung (Farbe und Marker je Modell)"); save(fig, name, sources, "diagramme/01_accuracy_verlauf, diagramme/02_loss_verlauf")
    P.FULL, P.save = FULL, save_curves; P.fig_curves(data)
    (OUT.parent.parent / "pruefung").mkdir(exist_ok=True); (OUT.parent.parent / "pruefung/diagrammquellen.json").write_text(json.dumps(REG, indent=1, ensure_ascii=False), encoding="utf-8")
    print(len(REG), "Druckfassungen in", OUT)


if __name__ == "__main__":
    main()
