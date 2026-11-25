# Station 1: ResNet18 als CNN-Vergleichsmodell, Vorbereitung

**Ergebnis:** `train_resnet18_adaptation.py` ist fertig und technisch getestet, einschließlich bitgenauer Wiederaufnahme. Ein vollständiges Training wurde nicht gestartet.

## Prüfung des vorhandenen `train_resnet18.py`

| Punkt | Befund |
|---|---|
| Verfahren | Backbone eingefroren, nur der Kopf (5.130 Parameter) trainiert |
| Auswahl | Bestes Modell nach Val-Accuracy, nicht nach Macro-F1 |
| Kennzahlen | Keine Macro-F1, keine F1 je Klasse |
| Checkpoint | Nur bestes Modell, kein Optimizer- oder Zufallszustand, nicht fortsetzbar |
| Optimizer | `torch.optim.AdamW`: `lerp` fällt bei DirectML auf die CPU zurück |
| Voreinstellung | Datensatz `original`, Batch 16 |
| Vorverarbeitung | Passend (`weights.transforms()`) |

Die Datei ist in `protected_script_updates.json` mit Prüfsumme geschützt und bleibt **unverändert**. Die neue Routine steht daneben in `train_resnet18_adaptation.py`.

## Neue Routine `train_resnet18_adaptation.py`

| Punkt | Umsetzung |
|---|---|
| Modell | ResNet18, `IMAGENET1K_V1`, neuer Kopf `Linear(512, 10)` |
| Verfahren | `--mode finetune` (Voreinstellung): alle Schichten trainieren, BatchNorm im Trainingsmodus. `--mode frozen`: nur Kopf, Backbone und BatchNorm eingefroren wie im alten Skript |
| Daten | `20261106T110038Z_15000`: 12.768 Training, 1.233 Validierung; Test nur über die Prüfsumme von `test.csv` |
| Klassen | `CLASS_NAMES` aus `pet_dataset.py`, gegen die feste Liste der zehn Klassen geprüft |
| Vorverarbeitung | Die der Gewichte: Resize 256 bilinear, Center Crop 224, ImageNet-Normalisierung. Keine Online-Augmentation |
| Ladereihenfolge | Gleicher Generator mit Seed 42 wie bei den ViT-Läufen |
| Optimizer | `AdamWNoLerp`, zwei Gruppen: Backbone 0,0001, Kopf 0,001; Weight Decay 0,01 |
| Je Epoche | Train-Loss, Train-Accuracy, Val-Loss, Val-Accuracy, Val-Macro-F1, F1 je Klasse, Dauer; dieselbe Auswertefunktion wie bei LoRA |
| Auswahl | `best_resnet18.pt` nach Val-Macro-F1, mit Konfusionsmatrix |
| `last_checkpoint.pt` | Nach jeder Epoche: ganzes Modell samt BatchNorm-Statistik, Optimizer, Epoche, Verlauf, Zufallszustände von Python, NumPy, Torch, DirectML und Ladereihenfolge |
| Protokolle | `run_config.json`, `training_log.csv`, `history.json` |
| Ausgabe | `results/20261106T110038Z_15000/seed_42/resnet18_finetune/` |

## Parameter und Einstellungen

| | finetune (empfohlen) | frozen |
|---|---|---|
| Trainierbare Parameter | 11.181.642 (alle) | 5.130 (Kopf) |
| Batch Size | 32 | 32 |
| Lernrate | Backbone 0,0001, Kopf 0,001 | Kopf 0,001 |
| Epochen | 10 | 10 |

Empfohlen ist Fine-Tuning: Linear Probing (nur Kopf) und LoRA (Teile des Backbones) passen beim ViT das Netz unterschiedlich stark an. Ein voll angepasstes CNN ist der übliche Vergleich dazu. Mit `--mode frozen` steht zusätzlich ein CNN-Gegenstück zum Linear Probing bereit.

**Trainingsdauer:** 10 Epochen wie bei LoRA. Bei rund 2,5 Minuten je Epoche sind das etwa **25 Minuten**. Die Auswahl nach Val-Macro-F1 fängt eine Überanpassung in späten Epochen ab.

## DirectML auf der RX 6600

- **Gradienten korrekt:** Die Gradienten des ganzen Modells stimmen mit der CPU überein (relative Abweichung 1·10⁻⁴). DirectML liegt sogar näher an der float64-Rechnung als die CPU in float32. Die laufende BatchNorm-Statistik weicht nur um 5·10⁻⁷ ab. Siehe `directml_probe.json`.
- **CPU-Rückfall:** Nur `lerp` in `torch.optim.AdamW`, behoben mit `AdamWNoLerp`. Im Trainingstest trat kein Rückfall auf. Die LoRA-Umgehungen für LayerNorm und Attention sind beim ResNet nicht nötig.
- **Speicher:** Batch 32 belegt 3,3 GB dediziert und 0,08 GB gemeinsam. Bei LoRA waren es 6,8 GB und 1,3 GB, beim ResNet gibt es also reichlich Reserve.
- **Tempo:** Ein Schritt dauert auf der GPU 0,12 s. Eine Epoche dauert mit Bildladen rund 2,5 Minuten, wobei das Laden der Bilder überwiegt. Mit 2 Ladeprozessen waren es 6,2 Minuten, weil der Prozessstart unter Windows Zeit kostet. Deshalb bleibt die Voreinstellung 0.

## Technischer Test

Test mit 96 Trainings- und 64 Validierungsbildern über 2 Epochen, jeder Lauf in einem eigenen Prozess (`tools/check_resnet18_training.py`).

| Prüfung | Ergebnis |
|---|---|
| `--check-only` beide Modi | Bestanden; Klassenverteilung geprüft, Gradienten genau bei den trainierbaren Parametern |
| Kurzer Lauf | Val-Macro-F1 0,41 → 0,68 (nur Funktionsnachweis) |
| Abbruch nach Epoche 1, Fortsetzen in neuem Prozess | **Bitgleich** zum ununterbrochenen Lauf: Verlauf, Modell samt BatchNorm-Statistik, Optimizer |
| finetune | Alle 62 Parametertensoren verändert, BatchNorm-Statistik in allen 20 Schichten aktualisiert; `best_resnet18.pt` ladbar |
| frozen | Nur `fc.weight` und `fc.bias` verändert, Rest unverändert |
| Oxford-Test | Kein Testbild geöffnet |
| Unverändert | ViT-Ergebnisse und -Checkpoints (110 Dateien per Prüfsumme), Datensatzdateien, `train_resnet18.py`; Integritätsprüfung bestanden |

Die Modelldateien des Tests wurden danach gelöscht (770 MB). Protokolle und `test_ergebnis.json` liegen in `technischer_test/`.

## Startbefehl für das vollständige Training

```powershell
cd C:\Users\Sipan\DeepLearning
.\.venv\Scripts\python.exe -X utf8 train_resnet18_adaptation.py --mode finetune --epochs 10 --batch-size 32 --learning-rate 0.0001 --head-learning-rate 0.001 --weight-decay 0.01 --seed 42 2>&1 | Tee-Object -FilePath results\praktikum1_station1_adaptation\resnet18_finetune_training.log
```

Nach einem Abbruch denselben Befehl mit `--resume auto` starten und bei `Tee-Object` `-Append` ergänzen.
