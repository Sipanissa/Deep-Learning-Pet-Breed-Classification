# Station 1: Erster vollständiger Lauf ViT-B/16 Linear Probing

**Ergebnis:** Der Lauf ist vollständig abgeschlossen, alle 30 Epochen sind durchgelaufen (`EXIT 0`). Die beste Epoche ist **11**: **Validierungs-Macro-F1 0,9735**, Accuracy 96,67 %, Val-Loss 0,139. Der Backbone ist unverändert. Oxford-Testbilder wurden nicht geladen und nicht ausgewertet.

## Konfiguration

| Punkt | Wert |
|---|---|
| Datensatz | `20261106T110038Z_15000`: 12.768 Training, 1.233 Validierung |
| Modell | ViT-B/16, `IMAGENET1K_V1`, Backbone eingefroren, 7.690 trainierbare Parameter |
| Training | 30 Epochen, Batch 32, Learning Rate 0,001, AdamW mit Weight Decay 0,01, CrossEntropyLoss, Seed 42 |
| Merkmale | Einmal berechnet: Training 573 s, Validierung 57 s; danach 0,2 bis 0,3 s je Epoche |
| Auswahl | Beste Validierungs-Macro-F1 |

Ausgabe in `results/20261106T110038Z_15000/seed_42/vit_b16_linear_probe/`: `best_head.pt`, `last_checkpoint.pt`, `training_log.csv`, `history.json`, `run_config.json` (mit Prüfsummen von `train.csv`, `val.csv`, `test.csv` und des Backbones). Merkmals-Cache in `results/vit_feature_cache/20261106T110038Z_15000/`.

## Verlauf

| Epoche | Train-Loss | Val-Loss | Val-Acc | Val-Macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 0,189 | 0,152 | 95,13 % | 0,9591 |
| 3 | 0,039 | **0,130** | 96,11 % | 0,9677 |
| 11 | 0,007 | 0,139 | **96,67 %** | **0,9735** |
| 22 | 0,001 | 0,172 | 96,67 % | 0,9725 |
| 30 | 0,0005 | 0,204 | 96,43 % | 0,9709 |

Alle Epochen stehen in `training_log.csv`.

## Beste Epoche (11): F1 je Klasse

| Klasse | Bilder | Richtig | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| British Shorthair | 119 | 98 | 0,952 | 0,824 | **0,883** |
| Russian Blue | 113 | 110 | 0,991 | 0,973 | 0,982 |
| Persian | 243 | 238 | 0,960 | 0,979 | 0,969 |
| Ragdoll | 258 | 249 | 0,919 | 0,965 | 0,941 |
| Bengal | 117 | 117 | 1,000 | 1,000 | 1,000 |
| Samoyed | 78 | 76 | 0,987 | 0,974 | 0,981 |
| Pomeranian | 123 | 123 | 1,000 | 1,000 | 1,000 |
| Keeshond | 53 | 53 | 1,000 | 1,000 | 1,000 |
| Great Pyrenees | 68 | 67 | 0,971 | 0,985 | 0,978 |
| Newfoundland | 61 | 61 | 1,000 | 1,000 | 1,000 |

## Konfusionsmatrix (Zeile: wahre Klasse, Spalte: Vorhersage)

| | BSH | RUS | PER | RAG | BEN | SAM | POM | KEE | GP | NEW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| British Shorthair | **98** | 1 | 2 | 18 | 0 | 0 | 0 | 0 | 0 | 0 |
| Russian Blue | 3 | **110** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Persian | 1 | 0 | **238** | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| Ragdoll | 1 | 0 | 8 | **249** | 0 | 0 | 0 | 0 | 0 | 0 |
| Bengal | 0 | 0 | 0 | 0 | **117** | 0 | 0 | 0 | 0 | 0 |
| Samoyed | 0 | 0 | 0 | 0 | 0 | **76** | 0 | 0 | 2 | 0 |
| Pomeranian | 0 | 0 | 0 | 0 | 0 | 0 | **123** | 0 | 0 | 0 |
| Keeshond | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **53** | 0 | 0 |
| Great Pyrenees | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | **67** | 0 |
| Newfoundland | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **61** |

41 Fehler insgesamt, alle innerhalb der Katzen beziehungsweise innerhalb der Hunde. Allein 18 entfallen auf British Shorthair, die als Ragdoll erkannt wurden. Dateien: `konfusionsmatrix_beste_epoche.csv`, `f1_je_klasse.csv`.

## Nachprüfung

- **Backbone:** Frisch geladene `IMAGENET1K_V1`-Gewichte und das aus `best_head.pt` gebaute Modell haben dieselbe Prüfsumme wie zu Laufbeginn.
- **Kopf:** Validierungswerte und Konfusionsmatrix aus `best_head.pt` neu berechnet, identisch.
- **Datensatzlisten:** Bytegleich zu vorher.

Protokolle: `training.log`, `nachpruefung.log`.

## Einordnung

- **Überanpassung:** Der Val-Loss ist am niedrigsten in Epoche 3 und steigt danach, während der Train-Loss gegen 0 geht. Der Kopf passt sich also zunehmend an die Trainingsmerkmale an.
- **Stabilität:** Die Macro-F1 bleibt ab Epoche 3 zwischen 0,966 und 0,974. Das entspricht einer Handvoll Validierungsbilder. Epoche 11 ist deshalb kaum sicher besser als ihre Nachbarn.
- **Auswahlverzerrung:** Die beste Epoche wurde auf der Validierung gewählt, ihre Werte sind daher leicht optimistisch. Eine unabhängige Schätzung liefert erst der Oxford-Test.
- **Auffällig:** Die 18 Verwechslungen British Shorthair → Ragdoll. Es könnte sich lohnen, diese Bilder anzusehen; das ist nicht geschehen.
