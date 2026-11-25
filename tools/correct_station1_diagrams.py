"""Korrigierte Diagramme der Station 1 (gültige Fassung), nur aus gespeicherten Ergebnissen. Kein Training, kein Oxford-Test.

Zeichnet 03, 04, 05, 08, 08b, oxford_test/f1_pro_klasse, oxford_test/test_vs_validierung, validierungspruefung/test_vs_validierung
und die Kopien lora_raenge, dauer_und_komplexitaet, f1_gegen_parameter in bericht/abbildungen. plot_station1_adaptation.py und
plot_paper_figures.py rufen diese Funktionen auf; die alten Zeichenfunktionen gibt es dort nicht mehr. Die Ausgabe ist
reproduzierbar (kein Zeitstempel in den Dateien).
  ohne Option      zeichnet alle elf Diagramme an ihren Platz
  --pruefen        zeichnet in einen temporären Ordner und vergleicht die Dateien bytegenau mit den gespeicherten
Originale und Prüfberichte: diagramme_korrektur_20261125/ und diagramme_qualitaetskontrolle_20261125/.
"""
import argparse, json, sys, tempfile
from pathlib import Path

import numpy as np
from common import ROOT
import plot_station1_adaptation as base
from plot_station1_adaptation import INK, INK2, MUTED, RUNS, de, de_axis, wilson
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from pet_dataset import CLASS_NAMES

BASE_RC = plt.rcParams.copy()
S1 = ROOT / "results/praktikum1_station1_adaptation"
KOR = S1 / "diagramme_korrektur_20261125"
VALP = S1 / "oxford_test_vorbereitung/validierungspruefung"
TEST = S1 / "oxford_test"
BER = ROOT / "bericht/abbildungen"
KEYS = [r["key"] for r in RUNS]
STYLE = {r["key"]: r for r in RUNS}
LOGPAR = {1e3: "1 Tsd.", 1e4: "10 Tsd.", 1e5: "100 Tsd.", 1e6: "1 Mio.", 1e7: "10 Mio.", 1e8: "100 Mio."}
REG = []
# Hinweistexte unter den Diagrammen
NOTE_ACC = ("Farbige Linie: 95-%-Wilson-Intervall, nimmt 1.233 unabhängige Bilder an. Graue Linie: 95-%-Bootstrap über geschätzte Tiere (nicht verifiziert): 744 Catza-Bilder wurden anhand des\n"
            "Namens im Bildtitel 234 Katzen zugeordnet. Für die übrigen 489 Bilder gibt es keine Tierangabe, jedes zählt einzeln. Das graue Intervall ist daher eine Näherung.")
NOTE_F1 = "Macro-F1: gespeichertes 95-%-Bootstrap-Intervall über einzelne Bilder (2.000 Ziehungen), nimmt unabhängige Bilder an. Zeigen mehrere Fotos dasselbe Tier, ist es vermutlich zu schmal."
NOTE_SEL = "Die Validierung diente auch zur Wahl der Epoche (30, 5, 10, 5, 10 Kandidaten); die Werte können dadurch zu günstig ausfallen."
NOTE_RUN = ("Alle Intervalle zeigen die Unsicherheit durch die begrenzte Bildstichprobe bei festem Checkpoint. Schwankungen zwischen Trainingsläufen sind nicht enthalten\n"
            "und wurden nicht gemessen (ein Lauf je Konfiguration, Seed 42).")
NOTE_SPAN = "Min. bis Max. ist die Spanne der Epochendauern innerhalb eines Laufs, kein Unsicherheitsintervall und keine Streuung zwischen Läufen."
OUTROOT = ROOT
WHITE = dict(facecolor="white", edgecolor="none", pad=.5)


# Diagramm ohne Zeitstempel speichern, damit es genau wiederholbar ist
def save(fig, path, sources, text, png=True):
    path = OUTROOT / path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (("png", "pdf") if png else ("pdf",)):
        fig.savefig(path.with_suffix("." + ext), bbox_inches="tight" if png else None, facecolor="white", metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    REG.append(dict(datei=path.name, beschreibung=text, quellen=sorted(set(sources))))


def budget(d):
    return f"{d['epochs']} Ep., " + {"lp": "nur Kopf", "rn": "alle Schichten"}.get(d["key"], f"α/r = {de(d['scaling'], 0)}" if d["scaling"] else "")


def src(p):
    return p.relative_to(ROOT).as_posix()


# Ergebnisse laden und Unsicherheitsintervalle ergänzen
def load():
    data = base.load()
    boot = json.loads((KOR / "tierbasierter_bootstrap/ergebnis.json").read_text(encoding="utf-8"))
    for d in data:
        m = json.loads((VALP / f"metriken_{d['key']}.json").read_text(encoding="utf-8"))
        assert m["confusion"] == d["cm"].tolist() and abs(m["macro_f1"] - d["best"]["val_macro_f1"]) < 1e-9
        assert abs(boot["accuracy"][d["key"]] - d["best"]["val_accuracy"]) < 1e-9
        d["f1ci"], d["wil"], d["tier"] = m["macro_f1_bootstrap95"], wilson(int(np.trace(d["cm"])), 1233), boot["B_bereinigt"]["accuracy95"][d["key"]]
    return data


SRC_F1 = [src(VALP / f"metriken_{k}.json") for k in KEYS]
SRC_T = [src(KOR / "tierbasierter_bootstrap/ergebnis.json")]


# Diagramme 03 und 04: Vergleich der fünf Modelle
def fig_compare(data, key, out):
    acc = key == "val_accuracy"
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[1.8, 1]), constrained_layout=True)
    for d in data:
        ax.plot([r["epoch"] for r in d["log"]], [r[key] for r in d["log"]], color=d["color"], marker=d["marker"], ms=4.5, ls=d["ls"], label=f"{d['label']} ({budget(d)})")
        ax.plot(d["best_epoch"], d["log"][d["best_epoch"] - 1][key], marker="*", ms=12, color=d["color"], mec=INK, mew=.7, ls="none", zorder=5)
    for e in (5, 10):
        ax.axvline(e + 0.5, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
        ax.text(e + 0.6, 1.0, "Budgetende\n" + ("r4, r16" if e == 5 else "r8, ResNet18"), transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va="top")
    ylabel = "Val-Accuracy [%]" if acc else "Val-Macro-F1"
    ax.set_xlim(0.5, 30.5); ax.set_xlabel("Epoche"); ax.set_ylabel(ylabel); ax.set_xticks([1, 5, 10, 15, 20, 25, 30]); de_axis(ax, digits=0 if acc else 2)
    ax.set_title("Verlauf über die Epochen (★ = gespeicherter bester Checkpoint)", loc="left"); ax.legend(loc="lower right", fontsize=7.5)
    ins = ax.inset_axes([.43, .36, .54, .33])  # Ausschnitt, weil sich Sterne sonst verdecken
    for d in data[1:4]:
        log = [r for r in d["log"] if r["epoch"] <= 5]
        ins.plot([r["epoch"] for r in log], [r[key] for r in log], color=d["color"], marker=d["marker"], ms=4, lw=1.3)
        if d["best_epoch"] <= 5:
            ins.plot(d["best_epoch"], d["best"][key], marker="*", ms=8, color=d["color"], mec=INK, mew=.5, ls="none", zorder=5)
    ins.set_xlim(.6, 5.4); ins.set_xticks([1, 2, 3, 4, 5]); ins.set_ylim(*((95.8, 97.5) if acc else (.9645, .9788))); ins.tick_params(labelsize=7, length=2)
    ins.yaxis.set_major_locator(MultipleLocator(.5 if acc else .005)); de_axis(ins, digits=1 if acc else 3)
    ins.set_title("Ausschnitt: Epochen 1 bis 5, LoRA", loc="left", fontsize=7.5, fontweight="normal", pad=3)
    for s in ins.spines.values():
        s.set_visible(True)
    ys = np.arange(len(data))[::-1]
    for y, d in zip(ys, data):
        v = d["best"][key]
        if acc:
            bx.plot(d["tier"], [y - .2] * 2, color=INK2, lw=1.4, marker="|", ms=6, mew=1.2)
        bx.plot(d["wil"] if acc else d["f1ci"], [y, y], color=d["color"], lw=2, solid_capstyle="round")
        bx.plot(v, y, marker=d["marker"], ms=8, color=d["color"], mec="white", mew=1, ls="none", zorder=3)
        bx.annotate(de(v, 2) + " %" if acc else de(v, 4), (v, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    bx.set_yticks(ys); bx.set_yticklabels([f"{d['short']}\nEp. {d['best_epoch']}/{d['epochs']}" for d in data]); bx.grid(axis="y", visible=False)
    bx.set_xlabel(ylabel); bx.set_ylim(-0.7, len(data) - (0 if acc else .3)); bx.set_title("Bester Checkpoint, 95-%-Intervalle" if acc else "Bester Checkpoint, 95-%-Bootstrap über Bilder", loc="left", fontsize=9.5)
    if acc:
        bx.set_xlim(91.8, 99.2); de_axis(bx, "x", digits=0)
        bx.legend(handles=[Line2D([], [], color=MUTED, lw=2, marker="o", ms=6, mec="white", label="Wert, Wilson (Farbe je Modell)"), Line2D([], [], color=INK2, lw=1.4, marker="|", ms=6, label="Bootstrap über geschätzte Tiere")], loc="upper left", fontsize=7.5, borderaxespad=.2)
    else:
        bx.set_xlim(.915, .9895); de_axis(bx, "x", digits=2); bx.xaxis.set_major_locator(MultipleLocator(.02))
    fig.suptitle("Validierungs-Accuracy der fünf Konfigurationen" if acc else "Validierungs-Macro-F1 der fünf Konfigurationen", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.01, (NOTE_ACC if acc else NOTE_F1) + "\n" + NOTE_RUN + " " + NOTE_SEL + "\nAchsen beginnen nicht bei null.", fontsize=7.5, color=INK2, va="top")
    name = "03_val_accuracy_vergleich" if acc else "04_val_macro_f1_vergleich"
    save(fig, out / name, [s for d in data for s in (d["sources"][0], d["sources"][3])] + (SRC_T if acc else SRC_F1),
         "Val-Accuracy-Verläufe und bester Checkpoint mit 95-%-Wilson-Intervall und 95-%-Bootstrap über geschätzte Tiere (Name im Bildtitel, nicht verifiziert)" if acc else "Val-Macro-F1-Verläufe und bester Checkpoint mit gespeichertem 95-%-Bootstrap-Intervall über einzelne Bilder")


# Diagramm 05: LoRA-Ränge
def fig_lora(data, out):
    lora = [d for d in data if d["key"] in ("r4", "r8", "r16")]; x = np.arange(3)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), constrained_layout=True)
    a, b, c, e = axes
    for i, d in enumerate(lora):
        a.plot([i + .13] * 2, d["tier"], color=INK2, lw=1.4, marker="_", ms=6, mew=1.2)
        a.plot([i, i], d["wil"], color=d["color"], lw=2, solid_capstyle="round"); a.plot(i, d["best"]["val_accuracy"], marker=d["marker"], ms=9, color=d["color"], mec="white", ls="none", zorder=3)
        a.annotate(f"{de(d['best']['val_accuracy'], 2)} %\n({de(int(np.trace(d['cm'])))}/1.233)", (i, d["tier"][1]), xytext=(4, 4), textcoords="offset points", ha="center", fontsize=7.5, bbox=WHITE)
        b.plot([i, i], d["f1ci"], color=d["color"], lw=2, solid_capstyle="round"); b.plot(i, d["best"]["val_macro_f1"], marker=d["marker"], ms=9, color=d["color"], mec="white", ls="none", zorder=3)
        b.annotate(de(d["best"]["val_macro_f1"], 4), (i, d["f1ci"][1]), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5)
        c.bar(i, d["trainable"], color=d["color"], width=.6, edgecolor="white", linewidth=2); c.annotate(de(d["trainable"]), (i, d["trainable"]), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
        s = np.array(d["epoch_seconds"]) / 60; e.bar(i, s.mean(), color=d["color"], width=.6, edgecolor="white", linewidth=2)
        e.plot([i, i], [s.min(), s.max()], color=INK, lw=1.2, marker="_", ms=8, mew=1.2)
        e.annotate(f"{de(s.mean(), 1)} min/Ep.\ngesamt {de(d['seconds_total'] / 60, 0)} min", (i, s.max()), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5, bbox=WHITE)
    a.set_title("Val-Accuracy, 95-%-Intervalle", loc="left", fontsize=9.5); a.set_ylabel("Accuracy [%]"); a.set_ylim(94, 100); de_axis(a, digits=0)
    a.legend(handles=[Line2D([], [], color=MUTED, lw=2, label="Wilson"), Line2D([], [], color=INK2, lw=1.4, marker="_", ms=6, label="Bootstrap über geschätzte Tiere")], loc="lower left", fontsize=7.5)
    b.set_title("Val-Macro-F1, 95-%-Bootstrap über Bilder", loc="left", fontsize=9.5); b.set_ylabel("Macro-F1"); b.set_ylim(.94, 1.0); de_axis(b, digits=2)
    c.set_title("Trainierbare Parameter", loc="left", fontsize=9.5); c.set_ylabel("Parameter"); c.set_ylim(0, 680000); c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: de(int(v))))
    e.set_title("Dauer je Epoche: Mittel, Min. bis Max.", loc="left", fontsize=9.5); e.set_ylabel("Minuten (Training + Validierung)"); e.set_ylim(0, 22); e.yaxis.set_major_locator(MultipleLocator(2)); de_axis(e, digits=0)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([f"r = {d['rank']}\nα/r = {de(d['scaling'], 0)}\n{d['epochs']} Ep.\n(beste: {d['best_epoch']})" for d in lora], fontsize=8); ax.set_xlim(-.6, 2.6); ax.grid(axis="x", visible=False)
    fig.suptitle("LoRA-Ränge im Vergleich (α = 16 fest, daher unterschiedliche Skalierung α/r; Budget 5, 10, 5 Epochen)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02, NOTE_ACC + "\n" + NOTE_F1 + "\n" + NOTE_RUN + " " + NOTE_SEL + "\nRang, Skalierung und Budget ändern sich gemeinsam, je Rang gibt es nur einen Lauf: Unterschiede, auch in der Dauer (Wanduhrzeit, AMD RX 6600 mit DirectML, Batch 32), lassen sich keinem Rang-Effekt zuordnen.\n" + NOTE_SPAN + "\n"
             "Die Achsen der Accuracy und der Macro-F1 umfassen denselben Bereich (6 Prozentpunkte bzw. 0,06) und beginnen nicht bei null.", fontsize=7.5, color=INK2, va="top")
    save(fig, out / "05_lora_raenge", [s for d in lora for s in d["sources"]] + SRC_T + SRC_F1[1:4], "LoRA-Ränge: Accuracy (Wilson, Bootstrap über geschätzte Tiere), Macro-F1 (Bootstrap über Bilder), trainierbare Parameter, Dauer je Epoche mit Min. bis Max. innerhalb des Laufs")


# Diagramme 08 und 08b: Parameter und Dauer
def cost_panels(axes, data, fs, ms, short):
    ys = np.arange(5)[::-1]; a, b, c = axes
    for y, d in zip(ys, data):
        col = d["color"]
        if d["key"] == "rn":  # hier sind alle Parameter trainierbar
            a.plot(d["total"], y, marker="o", ms=ms + 5, mfc="white", mec=INK2, mew=1.2, ls="none", zorder=2)
        else:
            a.plot([d["trainable"], d["total"]], [y, y], color="#c9c8c2", lw=1.2, zorder=1); a.plot(d["total"], y, marker="o", ms=ms, mfc="white", mec=INK2, mew=1.2, ls="none", zorder=3)
            a.annotate(de(d["total"] / 1e6, 1) + " Mio.", (d["total"], y), xytext=(0, ms * .8), textcoords="offset points", ha="center", fontsize=fs)
        a.plot(d["trainable"], y, marker=d["marker"], ms=ms, color=col, mec="white", mew=.8, ls="none", zorder=4)
        t = d["trainable"]; a.annotate((de(t) if not short else (de(t / 1e6, 1) + " Mio." if t >= 1e6 else de(t / 1e3, 1 if t < 1e4 else 0) + " Tsd.")), (t, y), xytext=(0, ms * (1.1 if d["key"] == "rn" else .8)), textcoords="offset points", ha="center", fontsize=fs)
        m = d["seconds_total"] / 60; b.plot(m, y, marker=d["marker"], ms=ms, color=col, mec="white", mew=.8, ls="none", zorder=3)
        right = m < 100
        text = f"{de(m, 1)} min" + (f" ({d['epochs']} Ep.)" if not short else "")
        if d["key"] == "lp":
            text += "" if short else f"\nMerkmale {de(d['feature_seconds'] / 60, 1)} min (GPU) + Kopf {de(d['head_seconds'], 1)} s (CPU)"
        b.annotate(text, (m, y), xytext=(ms * .9 if right else -ms * .9, 0), textcoords="offset points", ha="left" if right else "right", va="center", fontsize=fs)
        s = np.array(d["epoch_seconds"]) / 60
        if d["key"] == "lp":
            c.annotate(f"nicht vergleichbar:\nnur Kopf (CPU), {de(s.mean() * 60, 2)} s", (0, y), xytext=(3, 0), textcoords="offset points", va="center", fontsize=fs, color=INK2)
        else:
            c.plot([s.min(), s.max()], [y, y], color=col, lw=1.8, solid_capstyle="round"); c.plot(s.mean(), y, marker=d["marker"], ms=ms, color=col, mec="white", mew=.8, ls="none", zorder=3)
            left = s.mean() < 5
            c.annotate(de(s.mean(), 1) + ("" if short else " min") + f" ({de(s.min(), 1)}–{de(s.max(), 1)})", (s.max() if left else s.mean(), y), xytext=(ms, 0) if left else (0, ms * .8), textcoords="offset points",
                       ha="left" if left else "center", va="center" if left else "baseline", fontsize=fs)
    for ax in axes:
        ax.set_ylim(-.6, 4.7); ax.set_yticks(ys); ax.grid(axis="y", visible=False)
    a.set_xscale("log"); a.set_xlim(1.3e3, 6e8); a.minorticks_off(); a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: LOGPAR.get(v, "")))
    b.set_xlim(0, 145); c.set_xlim(0, 20); c.set_xticks([0, 5, 10, 15, 20])


def fig_cost(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True, sharey=True)
    cost_panels(axes, data, 7.5, 8, False); a, b, c = axes
    a.set_yticklabels([f"{d['short']}\n{d['epochs']} Ep." for d in data])
    a.set_title("Parameter (log. Skala)", loc="left"); a.set_xlabel("Parameter"); b.set_title("Gesamte Dauer des Laufs", loc="left"); b.set_xlabel("Minuten (gemessene Wanduhrzeit)"); c.set_title("Dauer je Epoche: Mittel (Min.–Max. im Lauf)", loc="left"); c.set_xlabel("Minuten je Epoche (Training + Validierung)")
    a.legend(handles=[Line2D([], [], marker="o", ms=8, mfc="white", mec=INK2, mew=1.2, ls="none", label="gesamt"), Line2D([], [], marker="o", ms=8, color=MUTED, mec="white", ls="none", label="davon trainierbar (Farbe je Modell)")], loc="lower left", fontsize=7.5)
    fig.suptitle("Trainingsdauer und Modellkomplexität", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02, "Hardware: LoRA und ResNet18 auf AMD RX 6600 (DirectML). Linear Probing: Merkmale des eingefrorenen ViT einmal auf AMD RX 6600 (DirectML) berechnet und gespeichert, der Kopf danach auf der CPU trainiert. Batch 32.\n"
             "Eingeschränkt vergleichbar: unterschiedliche Budgets (30 / 5 / 10 / 5 / 10 Epochen), unterschiedliche Rechenwege, je Konfiguration ein Lauf mit Wanduhrzeit ohne kontrollierte Systemlast. Die Gesamtdauern sind nicht auf gleiche\n"
             "Epochenzahl normiert. Die Dauer je Epoche ist nur zwischen LoRA und ResNet18 sinnvoll zu vergleichen, nicht mit Linear Probing. ResNet18: alle Parameter trainierbar.\n" + NOTE_SPAN, fontsize=7.5, color=INK2, va="top")
    save(fig, out / "08_dauer_und_komplexitaet", [s for d in data for s in d["sources"][:3]] + [base.LP_LOG.as_posix()], "Trainierbare und gesamte Parameter (log. Punktdiagramm), Gesamtdauer des Laufs, Dauer je Epoche mit Min. bis Max. innerhalb des Laufs; Linear Probing nicht im Epochenvergleich")


def f1_param(ax, data, fs, ms, names, off):
    for d in data:
        v = d["best"]["val_macro_f1"]
        ax.plot([d["trainable"]] * 2, d["f1ci"], color=d["color"], lw=1.6, solid_capstyle="round", zorder=2)
        if d["key"] != "rn":
            ax.plot([d["trainable"], d["total"]], [v, v], color=d["color"], lw=.8, ls=(0, (1, 2)), zorder=1); ax.plot(d["total"], v, marker=d["marker"], ms=ms * .8, mfc="none", mec=d["color"], mew=1.1, ls="none", zorder=3)
        ax.plot(d["trainable"], v, marker=d["marker"], ms=ms, color=d["color"], mec="white", mew=.8, ls="none", zorder=4)
        (dx, dy), ha, va = off[d["key"]]; ax.annotate(names(d), (d["trainable"], v if d["key"] == "rn" else d["f1ci"][0]), xytext=(dx, dy), textcoords="offset points", fontsize=fs, ha=ha, va=va, arrowprops=dict(arrowstyle="-", color=MUTED, lw=.6, shrinkA=0, shrinkB=1) if d["key"] == "r8" else None)
    ax.set_xscale("log"); ax.set_xlim(3e3, 4e8); ax.minorticks_off(); ax.set_ylim(.91, .99); de_axis(ax, digits=2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: LOGPAR.get(v, "")))


def fig_param(data, out):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    off = {"lp": ((0, -4), "center", "top"), "r4": ((-12, -4), "center", "top"), "r8": ((0, -30), "center", "top"), "r16": ((14, -4), "center", "top"), "rn": ((-10, 0), "right", "center")}
    f1_param(ax, data, 7.5, 9, lambda d: f"{d['label']}\n{de(d['best']['val_macro_f1'], 4)} | {de(d['trainable'])}", off)
    ax.set_xlabel("Parameter (log. Skala)"); ax.set_ylabel("Val-Macro-F1 (bester Checkpoint)")
    vit = [d["total"] / 1e6 for d in data[:4]]; ax.annotate(f"ViT gesamt:\n{de(min(vit), 1)} bis {de(max(vit), 1)} Mio.\n(liegen fast übereinander)", (9e7, .971), xytext=(0, -6), textcoords="offset points", ha="center", va="top", fontsize=7.5)
    ax.annotate("11.181.642 gesamt,\nalle trainierbar", (data[4]["total"], data[4]["best"]["val_macro_f1"]), xytext=(10, 0), textcoords="offset points", va="center", fontsize=7.5)
    ax.legend(handles=[Line2D([], [], marker="o", ms=9, color=MUTED, mec="white", ls="none", label="trainierbare Parameter"), Line2D([], [], marker="o", ms=8, mfc="white", mec=MUTED, mew=1.1, ls="none", label="Gesamtparameter desselben Modells"),
                       Line2D([], [], color=MUTED, lw=1.6, label="95-%-Bootstrap über Bilder (Macro-F1)")], loc="lower left", fontsize=8)
    fig.suptitle("Val-Macro-F1 gegen trainierbare und gesamte Parameter", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02, NOTE_F1.replace(", nimmt", ",\nnimmt") + "\n" + NOTE_RUN.replace("Checkpoint. Schw", "Checkpoint.\nSchw").replace("enthalten\nund", "enthalten und") + "\n" + NOTE_SEL + "\nArchitektur, Vortraining und Lernraten unterscheiden sich zugleich; ein ursächlicher Einfluss der Parameterzahl lässt sich nicht ableiten. y-Achse ist ein Ausschnitt.", fontsize=7.5, color=INK2, va="top")
    save(fig, out / "08b_macro_f1_gegen_parameter", [s for d in data for s in (d["sources"][2], d["sources"][3])] + SRC_F1, "Val-Macro-F1 mit 95-%-Bootstrap über Bilder gegen trainierbare (gefüllt) und gesamte Parameter (hohl)")


# Ergebnisse von Oxford-Test und Validierung
def fig_results(folder, data, validation):
    boot = {d["key"]: d["tier"] for d in data}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True); ys = np.arange(5)[::-1]; sources = []
    for ax, key, ci, lab, dg in ((axes[0], "accuracy", "accuracy_wilson95", "Accuracy [%]", 2), (axes[1], "macro_f1", "macro_f1_bootstrap95", "Macro-F1", 4)):
        for y, k in zip(ys, KEYS):
            m = json.loads((folder / f"metriken_{k}.json").read_text(encoding="utf-8")); assert m["n"] == (1233 if validation else 999); st = STYLE[k]; sources.append(src(folder / f"metriken_{k}.json"))
            if validation and key == "accuracy":
                ax.plot(boot[k], [y + .2] * 2, color=INK2, lw=1.4, marker="|", ms=6, mew=1.2)
            ax.plot(m[ci], [y, y], color=st["color"], lw=2, solid_capstyle="round"); ax.plot(m[key], y, marker=st["marker"], ms=9, color=st["color"], mec="white", ls="none", zorder=3)
            if not validation:
                ax.plot(m["val_" + key], y + .28, marker="|", ms=12, mew=2, color=INK2, ls="none")
            ax.annotate(de(m[key], dg), (m[key], y), xytext=(0, -8), textcoords="offset points", ha="center", va="top", fontsize=7.5)
        ax.set_yticks(ys); ax.set_yticklabels([STYLE[k]["label"].replace("ViT-B/16 ", "ViT ") for k in KEYS]); ax.set_ylim(-.75, 4.6); ax.grid(axis="y", visible=False); ax.set_xlabel(lab); de_axis(ax, "x", digits=0 if key == "accuracy" else 2)
        ax.set_title(("Accuracy, 95-%-Intervalle (farbig: Wilson)" if validation else "Accuracy, 95-%-Wilson-Intervall") if key == "accuracy" else "Macro-F1, 95-%-Bootstrap über Bilder", loc="left"); ax.xaxis.set_major_locator(MultipleLocator(1 if key == "accuracy" else .01))
    point = Line2D([], [], color=MUTED, marker="o", ms=8, mec="white", lw=2, label=("Validierung" if validation else "Oxford-Test") + ": Wert und 95-%-Intervall (Farbe und Form je Modell)")
    second = Line2D([], [], color=INK2, lw=1.4, marker="|", ms=6, label="95-%-Bootstrap über geschätzte Tiere (nur Accuracy)") if validation else Line2D([], [], color=INK2, marker="|", ms=12, mew=2, ls="none", label="Validierung: Wert desselben Checkpoints, ohne Intervall")
    fig.legend(handles=[point, second], loc="upper center", bbox_to_anchor=(.5, 0), ncol=2, fontsize=8.5)
    if validation:
        fig.suptitle("Ergebnisse auf der Validierung (1.233 Bilder, bester Checkpoint nach Val-Macro-F1)", x=.01, ha="left", fontsize=12, fontweight="bold"); note = NOTE_ACC + "\n" + NOTE_F1 + "\n" + NOTE_RUN + " " + NOTE_SEL + "\nDie x-Achsen sind Ausschnitte."; sources += SRC_T
    else:
        fig.suptitle("Ergebnisse auf dem Oxford-Test (999 Bilder, bester Checkpoint nach Val-Macro-F1)", x=.01, ha="left", fontsize=12, fontweight="bold")
        note = ("Die Intervalle zeigen die Unsicherheit durch die begrenzte Stichprobe von 999 Testbildern bei festem Checkpoint. Schwankungen zwischen Trainingsläufen sind nicht enthalten und\n"
                "wurden nicht gemessen (ein Lauf je Konfiguration, Seed 42). Die Intervalle nehmen unabhängige Bilder an. Tieridentitäten sind für den Test nicht dokumentiert; sollten mehrere Testbilder\n"
                "dasselbe Tier zeigen, könnten die Intervalle zu schmal sein. Die Validierung (1.233 Bilder) diente zur Wahl der Epoche und stammt überwiegend aus anderen Quellen als der Test (1.051 Bilder;\n"
                "182 stammen wie der Test aus Oxford-IIIT Pet). Der Abstand zwischen Strich und Punkt ist deshalb kein reiner Generalisierungseffekt. Die x-Achsen sind Ausschnitte.")
    fig.text(.01, -.08, note, fontsize=7.5, color=INK2, va="top")
    save(fig, folder / "diagramme/test_vs_validierung", sources, "Validierung: Accuracy und Macro-F1 mit Intervallen" if validation else "Oxford-Test: Accuracy und Macro-F1 mit Intervallen, Validierungswert als Strich")


# F1 je Klasse auf dem Oxford-Test
def fig_test_f1():
    res = {k: json.loads((TEST / f"metriken_{k}.json").read_text(encoding="utf-8")) for k in KEYS}
    fig, ax = plt.subplots(figsize=(10, 7.8), constrained_layout=True); offs = np.linspace(.36, -.36, 5); ys = np.arange(10)[::-1]
    for k in range(0, 10, 2):
        ax.axhspan(ys[k] - 0.5, ys[k] + 0.5, color="#f4f3f0", zorder=0)
    for o, k in zip(offs, KEYS):
        ax.plot([res[k]["per_class"][c]["f1"] for c in CLASS_NAMES], ys + o, ls="none", marker=STYLE[k]["marker"], ms=6, color=STYLE[k]["color"], mec="none", label=f"{STYLE[k]['label']} (Macro-F1 {de(res[k]['macro_f1'], 4)})", zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([f"{c} (n = {res['lp']['per_class'][c]['support']})" for c in CLASS_NAMES]); ax.grid(axis="y", visible=False)
    lo = min(res[k]["per_class"][c]["f1"] for k in KEYS for c in CLASS_NAMES)
    assert lo > .8; ax.set_xlim(.795, 1.005); ax.set_ylim(-.6, 9.6); ax.xaxis.set_major_locator(MultipleLocator(.025)); ax.set_xlabel("F1-Score (Oxford-Test)"); de_axis(ax, "x", digits=3)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    fig.suptitle("F1-Score je Klasse, Oxford-Test", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(.01, -.01, "n = Testbilder je Klasse. Punktwerte ohne Intervall: Ein anders klassifiziertes Bild ändert den F1-Wert einer Klasse um rund 0,005. Je Modell ein Trainingslauf (Seed 42),\n"
             "Schwankungen zwischen Läufen wurden nicht gemessen; kleine Unterschiede zwischen den Modellen sind nicht abgesichert. Die x-Achse ist ein Ausschnitt.", fontsize=7.5, color=INK2, va="top")
    save(fig, TEST / "diagramme/f1_pro_klasse", [src(TEST / f"metriken_{k}.json") for k in KEYS], "Oxford-Test: F1 je Klasse (Punktwerte ohne Intervall)")


# Kleinere Fassungen in Druckgröße
def paper_copies(data, out=BER):
    import plot_paper_figures as P
    lora = [d for d in data if d["key"] in ("r4", "r8", "r16")]; x = np.arange(3)
    fig, axes = plt.subplots(2, 2, figsize=(P.COL, 3.75)); fig.subplots_adjust(left=.135, right=.99, top=.95, bottom=.225, wspace=.36, hspace=.33); a, b, c, e = axes.ravel()
    for i, d in enumerate(lora):
        note = lambda ax, t, y, dy=2: ax.annotate(t, (i, y), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=7.5, bbox=dict(WHITE, pad=0))
        a.plot([i + .17] * 2, d["tier"], color=INK2, lw=1, marker="_", ms=4); a.plot([i, i], d["wil"], color=d["color"], lw=1.6); a.plot(i, d["best"]["val_accuracy"], marker=d["marker"], ms=6, color=d["color"], mec="white", ls="none", zorder=3); note(a, de(d["best"]["val_accuracy"], 2), d["tier"][1], 3)
        b.plot([i, i], d["f1ci"], color=d["color"], lw=1.6); b.plot(i, d["best"]["val_macro_f1"], marker=d["marker"], ms=6, color=d["color"], mec="white", ls="none", zorder=3); note(b, de(d["best"]["val_macro_f1"], 4), d["f1ci"][1], 3)
        c.bar(i, d["trainable"] / 1e3, color=d["color"], width=.62); note(c, de(d["trainable"] / 1e3, 0), d["trainable"] / 1e3)
        s = np.array(d["epoch_seconds"]) / 60; e.bar(i, s.mean(), color=d["color"], width=.62); e.plot([i, i], [s.min(), s.max()], color=INK, lw=.9, marker="_", ms=5); note(e, de(s.mean(), 1), s.max(), 3)
    a.set_title("Accuracy [%]"); a.set_ylim(94, 100); de_axis(a, digits=0); a.set_yticks([94, 96, 98, 100]); b.set_title("Macro-F1"); b.set_ylim(.94, 1); de_axis(b, digits=2); b.set_yticks([.94, .96, .98, 1])
    c.set_title("Parameter [Tsd.]"); c.set_ylim(0, 720); e.set_title("Minuten je Epoche"); e.set_ylim(0, 22); e.set_yticks([0, 5, 10, 15, 20])
    for ax in (a, b, c, e):
        ax.set_xlim(-.55, 2.55); ax.set_xticks(x); ax.grid(axis="x", visible=False)
        ax.set_xticklabels([f"r = {d['rank']}" for d in lora] if ax in (a, b) else [f"r = {d['rank']}\nα/r {d['scaling']:.0f}\n{d['epochs']} Ep." for d in lora])
    fig.legend(handles=[Line2D([], [], color=MUTED, lw=1.6, label="95 % je Bild"), Line2D([], [], color=INK2, lw=1, marker="_", ms=4, label="95 % je Tier (Tier geschätzt)"), Line2D([], [], color=INK, lw=.9, marker="_", ms=5, label="Epochen: Min.–Max.")], loc="lower center", ncol=2, bbox_to_anchor=(.52, -.015), columnspacing=.8, handlelength=1.6, labelspacing=.25)
    save(fig, out / "lora_raenge", [s for d in lora for s in d["sources"]] + SRC_T + SRC_F1[1:4], "wie 05", png=False)

    fig, axes = plt.subplots(1, 3, figsize=(P.FULL, 2.6), sharey=True); fig.subplots_adjust(left=.085, right=.99, top=.91, bottom=.215, wspace=.08)
    cost_panels(axes, data, 7.2, 5.5, True); a, b, c = axes
    a.set_yticklabels(["ViT-LP", "r = 4", "r = 8", "r = 16", "ResNet18"]); a.set_title("Parameter (log.)"); b.set_title("Gesamtdauer [min]"); c.set_title("Minuten je Epoche (Min.–Max.)"); b.set_xticks([0, 40, 80, 120]); a.set_xticks([1e4, 1e6, 1e8])
    for t in b.texts:
        t.set_text(t.get_text().replace(" min", ""))
    fig.legend(handles=[Line2D([], [], marker="o", ms=5.5, mfc="white", mec=INK2, ls="none", label="gesamt"), Line2D([], [], marker="o", ms=5.5, color=MUTED, mec="white", ls="none", label="trainierbar")], loc="lower left", ncol=2, bbox_to_anchor=(.07, -.015))
    fig.text(.995, .02, "Budgets, Rechenwege (LP-Kopf auf CPU) und Systemlast verschieden: eingeschränkt vergleichbar", ha="right", fontsize=7.2, color=INK2)
    save(fig, out / "dauer_und_komplexitaet", [s for d in data for s in d["sources"][:3]] + [base.LP_LOG.as_posix()], "wie 08", png=False)

    fig, ax = plt.subplots(figsize=(P.COL, 2.75)); fig.subplots_adjust(left=.15, right=.985, top=.975, bottom=.28)
    off = {"lp": ((0, -3), "center", "top"), "r4": ((-8, -3), "center", "top"), "r8": ((4, -12), "center", "top"), "r16": ((14, -3), "center", "top"), "rn": ((-6, 0), "right", "center")}
    f1_param(ax, data, 7.5, 6, lambda d: {"lp": "LP", "r4": "r = 4", "r8": "r = 8", "r16": "r = 16", "rn": "ResNet18"}[d["key"]], off)
    ax.set_xticks([1e4, 1e6, 1e8]); ax.set_xlabel("Parameter (log.)"); ax.set_ylabel("Macro-F1 (Validierung)")
    fig.legend(handles=[Line2D([], [], marker="o", ms=5.5, color=MUTED, mec="white", ls="none", label="trainierbar"), Line2D([], [], marker="o", ms=5, mfc="white", mec=MUTED, ls="none", label="gesamt"), Line2D([], [], color=MUTED, lw=1.6, label="95 % je Bild")], loc="lower center", ncol=3, bbox_to_anchor=(.55, -.015), columnspacing=1, handlelength=1.4)
    save(fig, out / "f1_gegen_parameter", [s for d in data for s in d["sources"][2:4]] + SRC_F1, "wie 08b", png=False)


def draw_diagrams(out):
    """03, 04, 05, 08, 08b nach out; Rückgabe: Einträge für datenquellen.json. Aufruf auch aus plot_station1_adaptation.py."""
    data = load(); n = len(REG)
    with plt.rc_context(BASE_RC):
        fig_compare(data, "val_accuracy", out); fig_compare(data, "val_macro_f1", out); fig_lora(data, out); fig_cost(data, out); fig_param(data, out)
    return REG[n:]


# Alle elf Diagramme zeichnen
def draw_all():
    draw_diagrams(S1 / "diagramme"); data = load()
    with plt.rc_context(BASE_RC):
        fig_test_f1(); fig_results(TEST, data, False); fig_results(VALP, data, True)
    paper_copies(data)


def main():
    global OUTROOT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0]); ap.add_argument("--pruefen", action="store_true", help="nichts ändern, nur bytegenau vergleichen"); a = ap.parse_args()
    if not a.pruefen:
        draw_all(); print(len(REG), "Diagramme gezeichnet"); return
    # Prüfmodus: neu zeichnen und mit den gespeicherten Dateien vergleichen
    with tempfile.TemporaryDirectory() as tmp:
        OUTROOT = Path(tmp); draw_all(); files = sorted(p for p in Path(tmp).rglob("*") if p.is_file())
        bad = [q.relative_to(ROOT).as_posix() for p in files for q in [ROOT / p.relative_to(tmp)] if not q.exists() or q.read_bytes() != p.read_bytes()]
    print(len(files), "Dateien verglichen,", len(bad), "abweichend", *bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
