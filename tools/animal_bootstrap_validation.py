"""Bootstrap der Validierungs-Accuracy über geschätzte Tiere, nur aus gespeicherten Vorhersagen (kein Modell, kein Oxford-Test).

Geschätzte Tieridentifikation, nicht verifiziert: Catza-Bilder mit demselben Namen im Bildtitel (attribution.csv) bilden eine Gruppe. Bilder anderer Quellen
haben keine Tierangabe und zählen einzeln; Abhängigkeiten dort bleiben unerkannt, die Intervalle sind daher eher zu schmal.
Gezogen werden Gruppen mit Zurücklegen; je Ziehung ist die Accuracy richtige Bilder / gezogene Bilder. 4.000 Ziehungen, Seed 1.
Regel A entspricht dem Prüfbericht vom 25.11.2026, Regel B entfernt zusätzlich Titel wie "FIFe CH" und Länderkürzel wie "SE*".
"""
import csv, json, re
from pathlib import Path

import numpy as np
from common import ROOT, sha

S1 = ROOT / "results/praktikum1_station1_adaptation"
OUT = S1 / "diagramme_korrektur_20261125/tierbasierter_bootstrap"
VER = ROOT / "data/dataset_expansion/combined_versions/20261106T110038Z"
PRED = S1 / "oxford_test_vorbereitung/validierungspruefung/vorhersagen_alle_modelle.csv"
KEYS = ["lp", "r4", "r8", "r16", "rn"]
REPS, SEED = 4000, 1


# CSV-Datei lesen
def rd(p):
    with open(p, encoding="utf-8-sig", newline="") as h:
        return list(csv.DictReader(h))


# Katzenname aus dem Bildtitel, Regel A
def name_a(title):
    parts = [x.strip() for x in title.split(" . ")]
    s = re.sub(r"\s*[\[(].*", "", parts[1] if len(parts) > 1 else parts[0])
    s = re.sub(r",?\s*(DSM|DVM|JW|DM)\b.*$", "", s)
    return re.sub(r"^(?:(?:[A-Z]{1,4}\d*\*?|S\*)[ ,])+", "", s).strip()


# Regel B entfernt zusätzlich Titel und Länderkürzel
def name_b(title):
    s = re.sub(r"^(?:(?:FIFe|CFA|TICA|WCF|[A-Z]{1,4}\d*)[ ,]+)+", "", name_a(title))
    return re.sub(r"^[A-Z]{1,2}\*", "", s).strip()


# Geschätzte Tiere statt einzelner Bilder zufällig ziehen,
# weil Fotos desselben Tieres nicht unabhängig sind
def bootstrap(groups, correct):
    ids = {c: i for i, c in enumerate(dict.fromkeys(groups))}
    g = np.array([ids[c] for c in groups]); n = len(ids); rng = np.random.default_rng(SEED)
    weights = [np.bincount(rng.integers(0, n, n), minlength=n)[g] for _ in range(REPS)]
    return n, {k: [float(x) for x in np.percentile([100 * (ok * w).sum() / w.sum() for w in weights], [2.5, 97.5])] for k, ok in correct.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    titles = {Path(r["path"]).stem: r["title"] for r in rd(VER / "attribution.csv") if r["quelle"] == "Catza"}
    val = {Path(r["path"]).stem: r for r in rd(VER / "val.csv")}
    rows = rd(PRED)
    assert len(rows) == len(val) == 1233 and {r["image_id"] for r in rows} == set(val)
    # Nur Catza-Bilder haben Namen, andere Bilder zählen einzeln
    catza = [val[r["image_id"]]["source"].startswith("Catza") for r in rows]
    assert sum(catza) == 744 and all(r["image_id"] in titles for r, c in zip(rows, catza) if c)
    correct = {k: np.array([float(r["klasse_wahr"] == r["vorhersage_" + k]) for r in rows]) for k in KEYS}
    result = {}
    groups = {}
    # Beide Regeln rechnen
    for rule, fn in (("A_pruefbericht", name_a), ("B_bereinigt", name_b)):
        groups[rule] = [fn(titles[r["image_id"]]) if c else "bild:" + r["image_id"] for r, c in zip(rows, catza)]
        breeds = {}
        for r, c, g in zip(rows, catza, groups[rule]):
            breeds.setdefault(g, set()).add(r["klasse_wahr"])
        assert all(len(v) == 1 for v in breeds.values())
        n, ci = bootstrap(groups[rule], correct)
        result[rule] = dict(gruppen=n, catza_tiere=len({g for g, c in zip(groups[rule], catza) if c}), accuracy95=ci)
    result["accuracy"] = {k: 100 * float(v.mean()) for k, v in correct.items()}
    result["verfahren"] = dict(ziehungen=REPS, seed=SEED, bilder=1233, catza_bilder=744, perzentile=[2.5, 97.5], code="tools/animal_bootstrap_validation.py",
                               quellen={p.relative_to(ROOT).as_posix(): sha(p) for p in (PRED, VER / "attribution.csv", VER / "val.csv")},
                               einschraenkung="Geschaetzte Tieridentifikation nach Namen im Bildtitel (Catza), nicht verifiziert; andere Quellen ohne Tierangabe, jedes Bild zaehlt einzeln")
    # Ergebnis und Zuordnung je Bild speichern
    (OUT / "ergebnis.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "gruppen_je_bild.csv", "w", encoding="utf-8-sig", newline="") as h:
        w = csv.writer(h, delimiter=";"); w.writerow(["image_id", "klasse", "quelle", "bildtitel", "gruppe_regel_a", "gruppe_regel_b"])
        for r, c, a, b in zip(rows, catza, groups["A_pruefbericht"], groups["B_bereinigt"]):
            w.writerow([r["image_id"], r["klasse_wahr"], val[r["image_id"]]["source"].split(" / ")[0], titles[r["image_id"]] if c else "", a, b])
    for rule in ("A_pruefbericht", "B_bereinigt"):
        print(rule, result[rule]["gruppen"], result[rule]["catza_tiere"], {k: [round(x, 2) for x in v] for k, v in result[rule]["accuracy95"].items()})


if __name__ == "__main__":
    main()
