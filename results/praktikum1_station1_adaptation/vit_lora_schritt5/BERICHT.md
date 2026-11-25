# Station 1, Schritt 5: LoRA-Trainingsroutine

**Ergebnis:** `train_vit_lora.py` ist fertig und funktioniert vollständig, einschließlich einer bitgenauen Wiederaufnahme nach einer Unterbrechung. Ein vollständiger Lauf wurde nicht gestartet.

## Routine

| Punkt | Umsetzung |
|---|---|
| Modell | ViT-B/16, `IMAGENET1K_V1`, bestehende `vit_lora.py`: LoRA auf Query und Value in allen 12 Schichten |
| Rang | `--rank 4`, `8` oder `16` |
| Voreinstellung | Batch 32, Lernrate LoRA 0,0005, Lernrate Kopf 0,001, Alpha 16, Dropout 0,1, Weight Decay 0,01, Seed 42, 10 Epochen |
| Optimizer | `AdamWNoLerp` mit zwei Parametergruppen, rechnet wie `torch.optim.AdamW`, aber ohne CPU-Rückfall |
| Daten | 12.768 Training, 1.233 Validierung; Vorverarbeitung der Gewichte; Test nur über die Prüfsumme von `test.csv` |
| Je Epoche | Train-Loss, Train-Accuracy, Val-Loss, Val-Accuracy, Val-Macro-F1, F1 je Klasse, Dauer; Fortschrittszeile alle 50 Batches |
| `last_checkpoint.pt` | Nach jeder Epoche: LoRA- und Kopfgewichte, Optimizer-Zustand, Epoche, Verlauf, beste Epoche, Trainingszeit, Zufallszustände von Python, NumPy, Torch, DirectML und Ladereihenfolge |
| `best_lora.pt` | Beste Epoche nach Val-Macro-F1, mit Konfusionsmatrix |
| `--resume auto` | Prüft vorher, ob Einstellungen, Datensatz-Prüfsummen und eingefrorene Gewichte zum Checkpoint passen |
| `run_config.json`, `history.json`, `training_log.csv` | Vollständige Konfiguration, Prüfsummen, Ergebnisse, Trainingsdauer |
| Ausgabeordner | `results/20261106T110038Z_15000/seed_42/vit_b16_lora_r{rang}/`; der Linear-Probing-Ordner bleibt unberührt |

Mit Alpha 16 für alle Ränge ist die Skalierung α/r bei Rank 4 gleich 4, bei Rank 8 gleich 2 und bei Rank 16 gleich 1. Alpha lässt sich mit `--lora-alpha` ändern.

## Technischer Test

Rank 8, 96 Trainings- und 64 Validierungsbilder, 2 Epochen mit je 3 Batches. Jeder Lauf in einem eigenen Prozess.

| Prüfung | Ergebnis |
|---|---|
| `--check-only` | Bestanden; Gradienten nur bei LoRA und Kopf |
| Lauf U: 2 Epochen ohne Unterbrechung | Läuft; Val-Macro-F1 0,340 → 0,729 (nur Funktionsnachweis) |
| Lauf I: Abbruch nach Epoche 1 | Checkpoint enthält alles, was die Wiederaufnahme braucht |
| Lauf R: neuer Prozess, `--resume auto` | **Bitgleich zu Lauf U:** Verlauf, alle Gewichte, Optimizer-Zustand |
| Trainierbar | 50 Tensoren, 302.602 Parameter, nur LoRA und Kopf; alle 50 verändert |
| Eingefroren | Prüfsumme am Laufende gleich; eingefrorene Gewichte im besten Modell gleich dem Ausgangsmodell |
| Oxford-Test | Kein Testbild geöffnet |
| Unverändert | Linear-Probing-Ergebnisse und Datensatzdateien |

## DirectML

- **Keine neuen Probleme.** Die bekannten Fehler sind in `vit_lora.py` umgangen: Absturz beim Backward von LayerNorm mit eingefrorenen Gewichten, CPU-Rückfall von `lerp` in AdamW, CPU-Rückfall der Attention. Es trat kein CPU-Rückfall auf.
- **Tempo:** Batch 32 mit Bildladen ergibt rund **15 Minuten pro Epoche** samt Validierung, also etwa 2,5 Stunden für 10 Epochen.
- **Ladeprozesse:** Mit 2 Ladeprozessen war der kurze Testlauf langsamer (24 statt 15 Minuten hochgerechnet). Unter Windows kostet das Starten der Prozesse Zeit, und gebremst wird die GPU, nicht das Laden. Deshalb bleibt es bei 0.
- **Speicher:** Die 8 GB der RX 6600 sind mit Batch 32 ausgeschöpft, DirectML lagert rund 1,3 GB in den Arbeitsspeicher aus. Kurze Läufe sind stabil; über Stunden ist das noch nicht erprobt. Bei einem Absturz lässt sich der Lauf mit `--resume auto` nach der letzten vollständigen Epoche fortsetzen.

## Startbefehl für den ersten vollständigen Lauf (Rank 8)

Im VS-Code-Terminal (PowerShell) im Projektordner:

```powershell
cd C:\Users\Sipan\DeepLearning
.\.venv\Scripts\python.exe -X utf8 train_vit_lora.py --rank 8 --epochs 10 --batch-size 32 --learning-rate 0.0005 --head-learning-rate 0.001 --lora-alpha 16 --lora-dropout 0.1 --weight-decay 0.01 --seed 42 2>&1 | Tee-Object -FilePath results\praktikum1_station1_adaptation\vit_lora_r8_training.log
```

Nach einem Abbruch mit denselben Parametern fortsetzen und `--resume auto` anhängen. `Tee-Object` überschreibt die Logdatei; für die Fortsetzung `-Append` ergänzen.
