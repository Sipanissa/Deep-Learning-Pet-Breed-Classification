# Augmentation und Modellvergleich: Plan (06.11.2026)

**Status: Augmentation am 06.11.2026 ausgeführt, Ergebnis in `augmented_versions/20261106T110038Z_15000` (siehe `UEBERSICHT.md` dort). Es wurde nicht trainiert.**

Grundlage ist die aktive Version `20261106T110038Z` (4.898 Train, 1.233 Val, 999 Test).

## 1. Zielgröße

| Teil | Bilder |
|---|---:|
| Train, Originale | 4.898 |
| Train, augmentiert | 7.870 |
| Val, unverändert | 1.233 |
| Test, unverändert | 999 |
| **Gesamt** | **15.000** |

Davon sind 7.130 Originalbilder. Die 7.870 augmentierten Bilder sind keine neuen Originale und zählen nicht zum Ziel von 20.000 Originalbildern. In der Arbeit sollten beide Zahlen getrennt genannt werden.

## 2. Verteilung auf die Klassen

Alle Klassen werden im Training auf eine gemeinsame Größe aufgefüllt. So erhalten unterrepräsentierte Klassen die meisten zusätzlichen Bilder.

| Klasse | Train original | Zusätzlich | Train gesamt | Augmentierte je Original | Max. Varianten je Original | Val | Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| British Shorthair | 494 | 783 | 1.277 | 1,59 | 2 | 119 | 100 |
| Russian Blue | 627 | 650 | 1.277 | 1,04 | 2 | 113 | 100 |
| Persian | 940 | 336 | 1.276 | 0,36 | 1 | 243 | 100 |
| Ragdoll | 912 | 364 | 1.276 | 0,40 | 1 | 258 | 100 |
| Bengal | 446 | 831 | 1.277 | 1,86 | 2 | 117 | 100 |
| Samoyed | 303 | 974 | 1.277 | 3,21 | 4 | 78 | 100 |
| Pomeranian | 443 | 834 | 1.277 | 1,88 | 2 | 123 | 100 |
| Keeshond | 217 | 1.060 | 1.277 | 4,88 | 5 | 53 | 99 |
| Great Pyrenees | 275 | 1.002 | 1.277 | 3,64 | 4 | 68 | 100 |
| Newfoundland | 241 | 1.036 | 1.277 | 4,30 | 5 | 61 | 100 |
| **Gesamt** | **4.898** | **7.870** | **12.768** | | | **1.233** | **999** |

Risiko: Keeshond und Newfoundland stammen zu über 80 % aus wiederverwendeten Originalen. Die Vielfalt der Klasse steigt dadurch nicht wie bei echten neuen Bildern, Überanpassung bleibt möglich. Val und Test bleiben ungleich verteilt.

## 3. Operationen

Pro Bild werden die Schritte in dieser Reihenfolge angewendet.

| Schritt | Parameter |
|---|---|
| Horizontal spiegeln | Wahrscheinlichkeit 0,5 |
| Drehen | ±10°, Ausschnitt ohne leere Ecken |
| Zufälliger Ausschnitt | Fläche 85–100 %, Position zur Mitte hin (25–75 %), Seitenverhältnis 0,9–1,1 (geändert nach Sichtkontrolle; erste Fassung in `augmentation_plan_v1_ersetzt.csv`) |
| Helligkeit | Faktor 0,85–1,15 |
| Kontrast | Faktor 0,85–1,15 |
| Sättigung | Faktor 0,90–1,10 |
| Leichte Unschärfe | Wahrscheinlichkeit 0,1, Radius 0,3–1,0 |
| Speichern | JPEG Qualität 95, 4:4:4, kurze Seite mindestens 224 Pixel |

**Bewusst nicht verwendet:** Farbton, Graustufen, vertikales Spiegeln, Erasing, Mixup und CutMix. Die Fellfarbe unterscheidet mehrere Klassen, etwa Russian Blue und Samoyed. Sie darf daher nicht stark verändert werden.

## 4. Reproduzierbarkeit und Herkunft

- Grund-Seed: 20261106. Der Seed jedes Bildes ergibt sich aus Grund-Seed, Prüfsumme des Originals und Variantennummer, also unabhängig von der Reihenfolge.
- `augmentation_plan.csv` enthält schon jetzt für alle 7.870 Bilder: ID, Klasse, Split, Pfad und Prüfsumme des Originals, Quelle, Lizenz, Variante, Seed und die exakten Parameter.
- Beim Generieren kommen Ausgabepfad, Prüfsumme, pHash und Bildgröße hinzu (`augmentation_manifest.csv`).
- `augmentation_config.json` hält Bereiche, Prüfsummen der Splits und die Versionen von Python und Pillow fest. Bei einer anderen Pillow-Version verweigert das Skript die Erzeugung.
- Alle Varianten eines Originals bleiben in Train. Val und Test werden bytegleich kopiert.

## 5. Herkunft der geplanten Bilder und Rechte

| Quelle des Originals | Geplante Bilder |
|---|---:|
| Stanford Dogs | 2.713 |
| Catza (CC BY 3.0) | 2.500 |
| Oxford-IIIT Pet | 1.717 |
| ImageNet | 469 |
| Openverse | 259 |
| Open Images | 209 |
| FeliBreed | 3 |

Augmentierte Bilder sind Bearbeitungen und erben die Bedingungen ihres Originals. Das ist eine Einschätzung, keine Rechtsberatung.

- **ImageNet:** nur nichtkommerzielle Forschung und Lehre; Zugang nur für Personen, die den Bedingungen vorher zugestimmt haben.
- **CC BY und CC BY-SA:** Bei einer Weitergabe sind Urheber und Änderungen anzugeben. 50 Bilder gehen auf CC BY-SA zurück und müssten dann unter denselben Bedingungen stehen.
- **Empfehlung:** Den augmentierten Datensatz nicht veröffentlichen. Stattdessen Plan, Skript und Parameter dokumentieren.

## 6. Vergleichbare Bedingungen

| Einstellung | Festlegung |
|---|---|
| Daten | Dieselben CSV-Dateien für alle Modelle, Prüfsummen in `run_config.json` |
| Epochen, Batchgröße, Lernrate | Für alle gleich, Vorschlag 20 / 16 / 0,001 |
| Seeds | 42, 43, 44; Ergebnis als Mittelwert und Streuung |
| Spiegeln beim Training | `--online-flip off` für alle |
| Modellauswahl | Nur nach Val |
| Test | Einmal am Ende mit den ausgewählten Modellen |
| Kennzahlen | Accuracy, Macro-F1, Kennzahlen je Klasse, Konfusionsmatrix |
| Varianten | Ohne Augmentation (`--dataset active`) und mit Augmentation (Ordner der neuen Version) |

Beispiel, nur zur Prüfung ohne Training:

```
.venv\Scripts\python.exe train_cnn.py --dataset active --epochs 20 --seed 42 --online-flip off --check-only
```

Ohne Parameter arbeiten die Skripte wie bisher mit `data/final_splits`. Läufe auf anderen Daten schreiben nach `results/<Version>/seed_<Seed>/<Modell>`, die bisherigen Ergebnisse bleiben also erhalten.

## 7. Bekannte Unterschiede und offene Punkte

1. **Drittes Modell fehlt:** Im Projekt gibt es nur `train_cnn.py` und `train_resnet18.py`.
2. **Testauswertung fehlt:** `test_trained_cnn.py` und `test_resnet18.py` nutzen `val.csv`. Für den Vergleich braucht es ein gemeinsames Auswertungsskript für den Oxford-Test.
3. **Vorverarbeitung:** Die CNN nutzt `Resize(224,224)` und Normalisierung 0,5; ResNet18 nutzt die ImageNet-Vorverarbeitung mit Zuschnitt. Das ist modellbedingt und bleibt so, muss aber dokumentiert werden.
4. **Vortrainierte Gewichte:** ResNet18 wurde auf ImageNet vortrainiert. Stanford Dogs, die ImageNet-Bilder und mindestens 14 Motive mit Bezug zum Oxford-Test könnten dem Backbone also schon bekannt sein. Das kann den Vorsprung von ResNet18 überschätzen und gehört in die Diskussion der Arbeit.
5. **Genauigkeit der Wiederholung:** Die Bilderzeugung ist in Bytes wiederholbar. Das Training auf DirectML ist es nicht garantiert, darum die drei Seeds.
6. **Leakage-Prüfung:** `generate` prüft jedes augmentierte Bild gegen Val und Test (pHash in 8 Ausrichtungen). Bei einem Treffer zum Test oder zu einem Val-Bild derselben Klasse wird die Variante mit einem abgeleiteten Seed neu gezogen und in `augmentation_replacements.csv` festgehalten.

## 8. Geprüft

| Prüfung | Ergebnis |
|---|---|
| `train_cnn.py` und `train_resnet18.py` mit `--check-only`, Datensatz `original` | 2.510 / 629 / 999, ein Batch auf DirectML durchlaufen |
| Dieselben Skripte, Datensatz `active` | 4.898 / 1.233 / 999, ein Batch auf DirectML durchlaufen |
| Zehn Klassen | `pet_dataset.py` und `dataset_expansion/config.py` identisch, alle Klassen in allen Splits |
| Überschneidung Train und Val | Keine Pfade |
| Stichprobe Augmentation (70 Bilder, nur im Speicher) | Bytegleich wiederholbar, RGB, mindestens 224 Pixel, keine leeren Ecken |
| `generate` ohne `--confirm` | Verweigert, kein Ordner angelegt |
| Vollständiger Ladetest (`check_training_environment.py`) | 4.898 / 1.233 / 999, jedes Bild mit beiden Vorverarbeitungen geladen |
| Labels gegen Dateipfade | 7.130 Bilder, 0 Widersprüche |
| Kleiner Durchlauf von `generate` im temporären Ordner | 20 Bilder (2 je Klasse), Leakage-Prüfung 0 Treffer, Val und Test bytegleich, beide Skripte laden das Ergebnis; Ordner danach gelöscht |
| `data/final_splits/*.csv`, `pet_dataset.py` | Prüfsummen unverändert |
