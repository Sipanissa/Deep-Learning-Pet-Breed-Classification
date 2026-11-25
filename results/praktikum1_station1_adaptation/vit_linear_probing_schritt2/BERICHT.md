# Station 1, Schritt 2: Linear-Probing-Training vorbereitet

**Ergebnis:** Die Trainingsroutine ist fertig, alle technischen Tests sind bestanden. Ein vollständiges Training lief nicht.

## A. Integritätsprüfung

- Keine aktive Datei braucht `train_cnn.py`.
- Zwei Werkzeuge nannten die Datei noch:
  - `kandidaten_bereinigen.py` nutzte sie als Kennzeichen für den Hauptordner, jetzt ist es `pet_dataset.py`.
  - `check_training_environment.py` nahm sie in den optionalen Probelauf der Augmentation auf, jetzt läuft dort nur noch ResNet18.
- Entfernt wurden nur der Schutzeintrag in `protected_hashes.json` und der zugehörige Ausnahmeeintrag in `protected_script_updates.json`. Beide stehen mit Prüfsumme in `protected_removed.json`.
- `tools/check_project_integrity.py`: **ALLES BESTANDEN**. Geprüft sind 4.152 geschützte Dateien, die 999 Oxford-Testbilder einzeln, die zehn Klassen, die Lizenzen und die ImageNet-Zustimmung. Protokoll: `../integritaet_20261107.log`.

## B. Trainingsroutine

Skript: `train_vit_linear_probe.py`. Bausteine: `vit_linear_probe.py`. Test: `tools/check_vit_linear_probe_training.py`.

| Punkt | Umsetzung |
|---|---|
| Modell | ViT-B/16, `IMAGENET1K_V1`, Backbone eingefroren, Kopf `Linear(768, 10)` mit 7.690 Parametern |
| Daten | `20261106T110038Z_15000`: 12.768 Training, 1.233 Validierung |
| Backbone | DirectML in `set_probe_mode()`, ohne Rückfall auf die CPU |
| Kopf | CPU, AdamW (Weight Decay 0,01), CrossEntropyLoss |
| Kennzahlen | Loss, Accuracy, Macro-F1, F1 je Klasse, Konfusionsmatrix der besten Epoche |
| Auswahl | Beste Validierungs-Macro-F1 |
| Reproduzierbarkeit | Seed für Python, NumPy, Torch und Batch-Reihenfolge; deterministische Algorithmen |
| Ausgaben | `best_head.pt`, `last_checkpoint.pt` mit Optimizer, `training_log.csv`, `history.json`, `run_config.json` mit Prüfsummen der Listen |

### Zwischenspeichern der Merkmale

**Sinnvoll und exakt.** Die Augmentation liegt fest auf der Platte, die Vorverarbeitung ist deterministisch und der Backbone hat keinen Dropout. Jede Epoche sähe also dieselben Merkmale.

- Der Test findet keine Abweichung zwischen Cache und direkter Berechnung, ebenso zwischen „Merkmale plus Kopf“ und dem vollen Forward.
- Der Cache gilt nur für dieselbe Liste, dieselben Gewichte und dieselbe Vorverarbeitung, sonst wird er neu berechnet.
- Mit `--online-flip on` ist der Cache gesperrt. `--features online` rechnet jede Epoche neu.

## Technischer Test

320 Trainings- und 100 Validierungsbilder, geschichtet, 2 Epochen mit je 10 Batches.

| Prüfung | Ergebnis |
|---|---|
| `--check-only` | Bestanden |
| Optimierungsschritt und Validierung | Laufen; Val-Macro-F1 0,919 und 0,939 (nur Funktionsnachweis, keine Aussage über die Güte) |
| Zwei Läufe mit gleichem Seed | Identische Werte, auch bei neu berechneten Merkmalen |
| Erneuter Lauf mit Cache | Liest den Cache, identische Werte |
| Nur der Kopf wird aktualisiert | Optimizer verwaltet nur 10 × 768 und 10; Kopf verändert; Prüfsumme des Backbones vor und nach dem Lauf gleich |
| Modus ohne Cache | Läuft |
| Oxford-Test | Von 452 geöffneten Bilddateien keine aus dem Test |
| Datensatzdateien | Bytegleich zu vorher |

Protokoll: `technischer_test.log`, Werte: `technischer_test/test_ergebnis.json`.

## Startbefehl

```
.venv\Scripts\python.exe train_vit_linear_probe.py --epochs 30 --batch-size 32 --learning-rate 0.001 --seed 42
```

Datensatz und Cache sind voreingestellt. Ausgabe nach `results/20261106T110038Z_15000/seed_42/vit_b16_linear_probe/`.

**Dauer:** Die Merkmale werden einmalig berechnet, rund 11 Minuten auf der RX 6600. Danach dauert jede Epoche nur wenige Sekunden. Weitere Läufe, etwa mit anderen Seeds oder Lernraten, nutzen den Cache aus `results/vit_feature_cache/` (rund 43 MB).

## Hinweise

- Die Validierung dient zur Auswahl der besten Epoche. Eine unabhängige Schätzung der Güte liefert erst der Oxford-Test.
- 30 Epochen ohne Lernraten-Plan sind eine Startkonfiguration. Der Verlauf in `training_log.csv` zeigt, ob mehr oder weniger sinnvoll ist.
