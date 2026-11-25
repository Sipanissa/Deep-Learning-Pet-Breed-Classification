# Station 1: Vorbereitung der einmaligen Oxford-Testauswertung

**Ergebnis:** `evaluate_station1_oxford_test.py` ist technisch bereit. Selbsttest und Validierungsprüfung sind bestanden. Kein Oxford-Testbild wurde geladen oder ausgewertet, und der Ergebnisordner `oxford_test/` existiert noch nicht.

## Das Skript

| Modus | Zweck | Oxford-Test |
|---|---|---|
| `--manifest-erstellen` | Friert Prüfsummen, beste Epochen, Validierungswerte und Vorverarbeitung der fünf Checkpoints ein. Berechnet den pHash der ImageNet-Bilder nahe am Test. Einmalig, überschreibt nichts | nicht geöffnet |
| `--selbsttest --out …` | Ganze Auswertung auf 50 synthetischen Bildern in einem temporären Ordner | nicht geöffnet |
| `--validierung --out …` | Wertet jeden Checkpoint auf der Validierung aus; die Konfusionsmatrix muss exakt der gespeicherten entsprechen | nicht geöffnet |
| `--oxford-test --freigabe OXFORD-TEST-EINMALIG` | Die einmalige finale Auswertung | 999 Bilder |

### Prüfungen in jedem Lauf
- **Checkpoints:** sha256 jeder Checkpoint-Datei, von `run_config.json` und `training_log.csv` gleich dem Manifest; die Prüfung läuft vor und nach der Auswertung.
- **Beste Epoche:** Epoche im Checkpoint gleich der Epoche mit der höchsten Val-Macro-F1 im Protokoll (LP 11, r4 1, r8 3, r16 3, ResNet18 10).
- **Klassenreihenfolge:** `CLASS_NAMES`, die Klassen im Checkpoint und das Manifest stimmen mit der festen Liste der zehn Klassen überein.
- **Vorverarbeitung:** die Transformation der Gewichte, gleich dem Eintrag in `run_config.json`. Resize 256 bilinear, Center Crop 224, ImageNet-Normalisierung; bei allen fünf Modellen gleich.
- **Keine Gewichtsänderung:** Prüfsumme des gesamten Modellzustands vor und nach der Auswertung gleich dem Manifest. Alle Parameter haben `requires_grad=False`.
- **Kein Training:** Während der Auswertung sind `torch.optim.Optimizer` und `backward` gesperrt; ein Aufruf bricht ab. Gerechnet wird unter `no_grad`, die Modelle laufen im Auswertungsmodus. Linear Probing rechnet wie beim Training: Merkmale auf DirectML, Kopf auf der CPU.

### Schutz der einmaligen Auswertung
- Ohne `--freigabe OXFORD-TEST-EINMALIG` startet die Auswertung nicht.
- Existiert `results/praktikum1_station1_adaptation/oxford_test/` bereits, bricht das Skript ab; nichts wird überschrieben.
- Vorher prüft es `test.csv` gegen die bekannte Prüfsumme, die Zahl von 999 Bildern und jede Bilddatei gegen `protected_hashes.json`.

### Ausgaben der finalen Auswertung (`oxford_test/`)
- **Vorhersagen je Modell (`vorhersagen_<modell>.csv`):** stabile Bild-ID (Oxford-Dateiname, z. B. `Bengal_1`), sha256 der Bilddatei, wahre und vorhergesagte Klasse, Wahrscheinlichkeit je Klasse.
- **Alle Vorhersagen nebeneinander (`vorhersagen_alle_modelle.csv`),** mit dem pHash-Abstand zu ImageNet; die Grundlage für paarweise Vergleiche.
- **Kennzahlen je Modell (`metriken_<modell>.json`):**
  - Accuracy mit 95-%-Wilson-Intervall;
  - Macro-F1 mit 95-%-Bootstrap-Intervall (2.000 Wiederholungen, Seed 42);
  - Macro-Precision und Macro-Recall;
  - Precision, Recall, F1 und Bildzahl je Klasse;
  - Konfusionsmatrix und Vergleichswerte der Validierung.
- **Gemeinsame Ergebnistabelle:** `ergebnisse_tabelle.csv` und `.md`.
- **Kennzahlen je Klasse für alle Modelle:** `klassenmetriken.csv`.
- **Paarweise Vergleiche (`mcnemar_paarweise.csv`):** exakter McNemar-Test für alle 10 Modellpaare mit Holm-Korrektur.
- **Mögliche ImageNet-Überschneidung:** `imagenet_naehe_je_bild.csv` (je Bild); in der Ergebnistabelle stehen zusätzlich die Kennzahlen ohne diese Bilder.
- **Diagramme in `diagramme/`, jeweils PNG und PDF:**
  - Test gegen Validierung (Accuracy und Macro-F1 mit Intervallen);
  - F1 je Klasse;
  - Konfusionsmatrizen.
- **`lauf_protokoll.json`:** Zeitpunkt, Versionen, Checkpoint- und Modellzustands-Prüfsummen.

## Technische Prüfung ohne Oxford-Test

| Prüfung | Ergebnis |
|---|---|
| Selbsttest, 50 synthetische Bilder, alle fünf Modelle | Bestanden: alle Ausgaben und Diagramme erzeugt, Bild-IDs stabil, Gewichte unverändert |
| Trainingssperre | Optimizer und `backward` werden abgewiesen |
| Freigabe | Ohne oder mit falscher Freigabe startet `--oxford-test` nicht |
| Geöffnete Dateien | Selbsttest nur die 50 synthetischen Bilder; Validierungsprüfung nur 1.233 Validierungsbilder; kein Testbild |
| Validierungsprüfung | **Alle fünf Checkpoints reproduzieren ihre gespeicherte Validierungs-Konfusionsmatrix exakt.** Laden, Vorverarbeitung, Klassenreihenfolge und Vorwärtsrechnung entsprechen damit dem Training |
| Bestehende Ergebnisse | Alle 153 früheren Dateien per Prüfsumme unverändert; Integritätsprüfung bestanden |

Nebenergebnis der Validierungsprüfung: Die Vorhersagen je Bild erlauben jetzt einen McNemar-Test auf der Validierung (`validierungspruefung/mcnemar_paarweise.csv`). Linear Probing gegen LoRA ergibt p = 0,21 bis 0,31, zwischen den Rängen p = 1. ViT gegen ResNet18 ergibt p < 0,001, auch nach Holm-Korrektur. Die Validierung diente allerdings zur Auswahl der Epoche.

## Einschränkungen für den Bericht

### Vortraining auf ImageNet-1k
- **Bekannte Klassen:** Beide Backbones (ViT-B/16 und ResNet18, je `IMAGENET1K_V1`) wurden überwacht auf ImageNet-1k trainiert. Sechs der zehn Klassen sind dort eigene Klassen: Persian, Samoyed, Pomeranian, Keeshond, Great Pyrenees und Newfoundland. Die Backbones kennen diese Rassen also schon mit Label. British Shorthair, Russian Blue, Ragdoll und Bengal gibt es in ImageNet-1k nicht als Klassen; dort gibt es nur allgemeine Katzenklassen wie tabby, tiger cat und Egyptian cat. Gute Werte bei den Hunden sind deshalb kein reiner Anpassungserfolg.
- **Dieselben Fotos im Vortraining:** Laut ImageNet-Prüfbericht der Datenphase liegen 59 ImageNet-Bilder mit pHash-Abstand ≤ 10 nahe an Oxford-Testbildern. 56 davon stammen aus dem ILSVRC-2012-**Trainingssplit**, also aus den Vortrainingsdaten. 14 haben den Abstand 0 (12 Keeshond, 1 Newfoundland, 1 Persian); die angesehenen Paare zeigen dasselbe Foto. Diese Testbilder kann das Vortraining gesehen haben.
- **Umgang damit:**
  - Die ImageNet-Bilder wurden in der Datenphase aus dem eigenen Training ausgeschlossen.
  - Die finale Auswertung berechnet für jedes Testbild den kleinsten pHash-Abstand zu diesen 59 Bildern und gibt die Kennzahlen zusätzlich ohne Bilder mit Abstand ≤ 10 an.
  - Andere Fotos derselben Tiere findet pHash nicht.
- **Stanford Dogs:** Die Bilder stammen aus ImageNet und liegen in Training und Validierung. Validierungswerte der Hundeklassen können deshalb durch das Vortraining begünstigt sein.

### Überschneidungen zwischen den Splits
- In der Datenphase geprüft: Kein Trainings- oder Validierungsbild hat einen pHash-Abstand ≤ 10 zu einem Testbild (8 Ausrichtungen), und es gibt keine exakten Duplikate.
- **Dasselbe Tier auf anderen Fotos:** Training und Test enthalten beide Oxford-IIIT-Pet-Bilder. Der Oxford-Split trennt nach Bild, nicht nach Tier. Ein Tier kann daher auf verschiedenen Fotos in Training und Test vorkommen; außer bei Catza lässt sich das nicht ausschließen.

### Statistik
- **Größe des Tests:** etwa 100 Bilder je Klasse. Das 95-%-Intervall der Accuracy ist rund ±1,1 bis 1,5 Prozentpunkte breit. Die Unterschiede zwischen den LoRA-Rängen auf der Validierung (1 Bild) liegen weit darunter.
- **Nur ein Seed je Konfiguration:** Streuung über Trainingsläufe ist nicht erfasst. Der McNemar-Test vergleicht nur die Vorhersagen der gespeicherten Modelle.
- **Unterschiedliche Budgets und Lernraten** (30, 5, 10, 5, 10 Epochen) bleiben eine Einschränkung des Vergleichs.

## Start der finalen Auswertung

Erst starten, wenn alle Trainings endgültig abgeschlossen sind; danach dürfen keine Einstellungen mehr nach dem Testergebnis angepasst werden.

```powershell
cd C:\Users\Sipan\DeepLearning
.\.venv\Scripts\python.exe -X utf8 evaluate_station1_oxford_test.py --oxford-test --freigabe OXFORD-TEST-EINMALIG 2>&1 | Tee-Object -FilePath results\praktikum1_station1_adaptation\oxford_test_auswertung.log
```

Geschätzte Dauer: unter 10 Minuten, davon etwa 1 bis 2 Minuten je ViT-Modell und unter 1 Minute für ResNet18, dazu die Prüfsummen und den pHash der 999 Bilder. Bricht der Lauf ab, bevor `oxford_test/` vollständig ist, muss der unvollständige Ordner vor einem neuen Start bewusst gelöscht werden.

## Dateien

| Datei | Inhalt |
|---|---|
| `evaluate_station1_oxford_test.py` | Auswertungsskript im Projektordner |
| `oxford_test_vorbereitung/checkpoint_manifest.json` | Eingefrorene Prüfsummen, Epochen, Validierungswerte, Vorverarbeitung |
| `oxford_test_vorbereitung/imagenet_testnaehe_phash.csv` | 59 ImageNet-Bilder mit Split und pHash (bleibt lokal; ImageNet-Bedingungen) |
| `oxford_test_vorbereitung/selbsttest/`, `selbsttest.log` | Ausgaben des Selbsttests auf synthetischen Bildern |
| `oxford_test_vorbereitung/validierungspruefung/`, `validierungspruefung.log` | Auswertung auf der Validierung, mit Vorhersagen je Bild |
