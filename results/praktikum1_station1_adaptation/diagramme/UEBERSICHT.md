# Diagramme Praktikum 1, Station 1: Adaptation

Erzeugt mit `tools/plot_station1_adaptation.py` (Matplotlib 3.11, 300 dpi PNG und PDF mit eingebetteten Schriften). 03, 04, 05, 08 und 08b zeichnet `tools/correct_station1_diagrams.py` in der am 25.11.2026 korrigierten Fassung; `plot_station1_adaptation.py` ruft dieses Skript auf. Die früheren Zeichenfunktionen sind entfernt.

```
.venv\Scripts\python.exe -X utf8 tools\plot_station1_adaptation.py              (alle Diagramme dieses Ordners)
.venv\Scripts\python.exe -X utf8 tools\correct_station1_diagrams.py             (nur die elf korrigierten Diagramme)
.venv\Scripts\python.exe -X utf8 tools\correct_station1_diagrams.py --pruefen   (nichts ändern, bytegenau vergleichen)
```

Frühere Fassungen: `diagramme_korrektur_20261125/sicherung_originale/`. Prüfbericht: `diagramme_qualitaetskontrolle_20261125/PRUEFBERICHT.md`.

## Datengrundlage

- **Läufe:** `results/20261106T110038Z_15000/seed_42/<lauf>/`. Gelesen werden `training_log.csv`, `history.json`, `run_config.json` und der beste Checkpoint (`best_head.pt`, `best_lora.pt`, `best_resnet18.pt`).
- **Merkmalsberechnung beim Linear Probing:** 573 s und 57 s aus `vit_linear_probing_schritt3/training.log`.
- **Nur Validierung:** Alle Werte stammen aus dem Validierungssplit mit 1.233 Bildern. Für die Diagramme dieses Ordners wurde der Oxford-Test weder geladen noch ausgewertet; seine Diagramme liegen in `oxford_test/diagramme/`.
- **Konsistenzprüfung vor dem Zeichnen:** Das Skript bricht bei Abweichungen ab. Es prüft:
  - `training_log.csv` gleich `history.json` für jede Epoche;
  - die Epoche des Checkpoints gleich der Epoche mit der höchsten Val-Macro-F1 im Protokoll;
  - Accuracy, Macro-F1 und F1 je Klasse neu aus der gespeicherten Konfusionsmatrix berechnet und gleich den gespeicherten Werten.
- **Macro-F1-Intervalle:** gespeicherter 95-%-Bootstrap über einzelne Bilder (2.000 Ziehungen) aus `oxford_test_vorbereitung/validierungspruefung/metriken_*.json`.
- **Selbst berechnete Werte:** nur die 95-%-Wilson-Intervalle der Accuracy (aus der Zahl richtiger Bilder), der 95-%-Bootstrap der Accuracy über geschätzte Tiere (siehe „Unsicherheit“) und die Gesamtparameterzahl bei LoRA (ViT-B/16 mit Kopf plus LoRA-Parameter aus `run_config.json`). Nichts ist interpoliert. Linien verbinden nur gemessene Epochen.
- **Unverändert (Stand 14.11.2026):** Alle 153 Dateien unter `results/20261106T110038Z_15000/`, `results/praktikum1_station1_adaptation/` und im Merkmals-Cache sind per Prüfsumme vor und nach dem Lauf identisch.
- **Quellen je Diagramm:** in `datenquellen.json`. Die Zahlen hinter den Diagrammen stehen in `kennzahlen_beste_checkpoints.csv` und `f1_je_klasse.csv`.

## Unsicherheit

- **Was die Intervalle zeigen:** die Unsicherheit durch die begrenzte Bildstichprobe, bei festem Checkpoint.
- **Was sie nicht zeigen:** Schwankungen zwischen Trainingsläufen. Es gibt einen Lauf je Konfiguration (Seed 42), diese Schwankung wurde nicht gemessen.
- **Wilson und Bootstrap über Bilder** nehmen unabhängige Bilder an. Zeigen mehrere Fotos dasselbe Tier, sind sie vermutlich zu schmal.
- **Bootstrap über geschätzte Tiere** (nur Accuracy; 03, 05): **geschätzte Tieridentifikation, nicht verifiziert.** 744 Catza-Bilder wurden anhand des Namens im Bildtitel 234 Katzen zugeordnet. Die übrigen 489 Validierungsbilder haben keine Tierangabe, jedes zählt einzeln. Gleiche Namen können verschiedene Katzen bezeichnen, verschiedene Schreibweisen dieselbe. Die Intervalle sind eine Näherung.
- **Min. bis Max. der Epochendauer** (05, 08) ist die Spanne innerhalb eines Laufs, kein Unsicherheitsintervall.

## Konfigurationen

| Konfiguration | Budget | Beste Epoche | Val-Acc. | Richtig | Val-Macro-F1 | Trainierbar | Dauer |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViT-B/16 Linear Probing | 30 Ep. | 11 | 96,67 % | 1.192 | 0,9735 | 7.690 | 10,6 min |
| ViT-B/16 LoRA r = 4, α/r = 4 | 5 Ep. | 1 | 97,16 % | 1.198 | 0,9773 | 155.146 | 83 min |
| ViT-B/16 LoRA r = 8, α/r = 2 | 10 Ep. | 3 | 97,16 % | 1.198 | 0,9760 | 302.602 | 130 min |
| ViT-B/16 LoRA r = 16, α/r = 1 | 5 Ep. | 3 | 97,24 % | 1.199 | 0,9766 | 597.514 | 79 min |
| ResNet18 Fine-Tuning | 10 Ep. | 10 | 94,16 % | 1.161 | 0,9359 | 11.181.642 | 20 min |

Die Linear-Probing-Dauer setzt sich aus der Merkmalsberechnung (10,5 min) und dem Training des Kopfes (7 s für 30 Epochen) zusammen.

## Diagramme

| Datei | Inhalt |
|---|---|
| `01_accuracy_verlauf` | Train- und Val-Accuracy je Epoche, ein Feld je Konfiguration. Jede x-Achse zeigt das eigene Budget, die beste Epoche ist markiert. Lesehilfe mit Budgets und α/r |
| `02_loss_verlauf` | Train- und Val-Loss je Epoche, logarithmische y-Achse, sonst wie 01 |
| `03_val_accuracy_vergleich` | Links alle Val-Accuracy-Verläufe auf einer Achse von 1 bis 30 mit markiertem Budgetende nach 5 und 10 Epochen und einem Ausschnitt der LoRA-Läufe. Rechts der beste Checkpoint mit 95-%-Wilson-Intervall (farbig) und 95-%-Bootstrap über geschätzte Tiere (grau) |
| `04_val_macro_f1_vergleich` | Wie 03 für die Val-Macro-F1, rechts mit gespeichertem 95-%-Bootstrap über Bilder |
| `05_lora_raenge` | Rang 4, 8, 16: Val-Accuracy (Wilson, Bootstrap über geschätzte Tiere), Val-Macro-F1 (Bootstrap über Bilder), trainierbare Parameter, mittlere Dauer je Epoche mit Min. bis Max. im Lauf. Je Rang sind α/r, Budget und beste Epoche angegeben |
| `06_konfusionsmatrix_<lauf>` (lp, r4, r8, r16, rn) und `06_konfusionsmatrizen_uebersicht` | Konfusionsmatrix des besten Checkpoints. Zahlen sind Bilder, die Farbe zeigt den Zeilenanteil (Recall) |
| `07_f1_pro_klasse` | F1 je Klasse für alle fünf Konfigurationen, mit Bildzahl je Klasse |
| `08_dauer_und_komplexitaet` | Parameter gesamt und trainierbar (Punkte, logarithmisch), Gesamtdauer des Laufs, mittlere Dauer je Epoche mit Min. bis Max. im Lauf (ohne Linear Probing). Hardware je Schritt und eingeschränkte Vergleichbarkeit sind angegeben |
| `08b_macro_f1_gegen_parameter` | Val-Macro-F1 mit 95-%-Bootstrap über Bilder gegen trainierbare (gefüllt) und gesamte Parameter (hohl) |

Farben und Marker sind über alle Diagramme gleich. Zusätzlich hat jede Konfiguration eine eigene Markerform, ResNet18 eine gestrichelte Linie; so bleiben die Diagramme auch in Schwarz-Weiß und bei Farbsehschwäche lesbar.

## Was sich aus den Daten ableiten lässt

1. **Die vier ViT-Konfigurationen liegen auf der Validierung deutlich vor ResNet18.**
   - ViT: 1.192 bis 1.199 richtige Bilder; ResNet18: 1.161, also 31 bis 38 Bilder weniger.
   - Macro-F1 0,973 bis 0,977 gegen 0,936.
   - Die 95-%-Wilson-Intervalle von ResNet18 (92,7 bis 95,3 %) und den ViT-Läufen (ab 95,5 %) überlappen nicht; sie nehmen unabhängige Bilder an.
   - Die Intervalle über geschätzte Tiere überlappen dagegen: ResNet18 92,4 bis 95,7 %, Linear Probing ab 95,0 %, LoRA ab 95,4 bis 95,6 %. Überlappende Intervalle belegen nicht, dass kein Unterschied besteht; dafür sind gepaarte Vergleiche nötig (`oxford_test/mcnemar_paarweise.csv`).
   - Alle Intervalle gelten für den jeweiligen Checkpoint. Ob der Abstand bei anderen Seeds gleich ausfällt, ist nicht gemessen.
   - Belegt ist dies nur für genau diese Einstellungen. ResNet18 nutzt eine andere Lernrate, und seine beste Epoche ist die letzte (Diagramme 03 und 04).
2. **Zwischen den LoRA-Rängen gibt es keinen belastbaren Unterschied.**
   - Die drei Ränge liegen 1 Bild auseinander (1.198, 1.198, 1.199). Bei der Macro-F1 beträgt die Spanne 0,0013.
   - Rang, Skalierung α/r und Budget ändern sich gleichzeitig, ihr Einfluss lässt sich also nicht trennen. Eine Aussage „größerer Rang ist besser“ oder „schlechter“ geben die Daten nicht her (Diagramm 05).
3. **LoRA liegt knapp vor Linear Probing, der Unterschied ist aber nicht abgesichert.**
   - LoRA erkennt 6 bis 7 Bilder mehr (+0,5 Prozentpunkte Accuracy) und erreicht eine um 0,0025 bis 0,0038 höhere Macro-F1.
   - Die Intervalle überlappen, und es gibt nur einen Lauf je Konfiguration.
   - Die Vorhersagen je Bild sind seit der Validierungsprüfung gespeichert; gepaarte Vergleiche stehen in `oxford_test_vorbereitung/validierungspruefung/mcnemar_paarweise.csv` und `oxford_test/mcnemar_paarweise.csv`. Im Oxford-Test ist der Unterschied zwischen Linear Probing und LoRA nicht signifikant (p = 0,19 bis 0,56). Das ist eine Tendenz, kein Nachweis.
4. **Die ViT-Läufe erreichen ihr Bestes früh, danach passen sie sich den Trainingsdaten an.**
   - Der Train-Loss fällt auf 10⁻³ bis 10⁻⁵, der Val-Loss steigt nach seinem Minimum: bei Linear Probing ab Epoche 3, bei LoRA ab Epoche 1 (Diagramm 02).
   - LoRA erreicht seine beste Macro-F1 in Epoche 1 oder 3. Bei Rank 8 bringen die Epochen 4 bis 10 keine Verbesserung.
   - Ein längeres Budget hätte bei LoRA nach diesen Daten wenig gebracht. Für Rank 4 und 16 lässt sich das nur bis Epoche 5 sagen.
5. **ResNet18 schwankt zwischen den Epochen stärker und ist am Budgetende vermutlich nicht ausgereizt.**
   - Die Val-Accuracy springt zwischen 91,6 % und 94,2 %. Die beste Epoche ist die letzte; ob mehr Epochen helfen, ist offen.
   - Der Abstand zwischen Training (99,4 %) und Validierung (94,2 %) ist größer als bei den ViT-Läufen.
6. **British Shorthair ist für alle Modelle die schwierigste Klasse.**
   - F1 0,86 bis 0,89, häufigster Fehler ist die Verwechslung mit Ragdoll: 18, 20, 18 und 18 Bilder bei den ViT-Läufen, 13 bei ResNet18 (Diagramme 06 und 07).
   - ResNet18 hat zusätzliche Schwächen bei den Hunden: Samoyed wird als Great Pyrenees erkannt (7 Bilder), Pomeranian als Samoyed (5). Außerdem Russian Blue als British Shorthair (9).
   - Die ViT-Läufe erkennen Bengal, Pomeranian, Keeshond und Newfoundland nahezu fehlerfrei.
7. **Trainierbare Parameter und Ergebnis**
   - Linear Probing trainiert 0,009 % der Parameter des Modells, LoRA 0,18 bis 0,69 %. Beide erreichen auf dieser Aufgabe eine höhere Macro-F1 als ResNet18 mit 100 % trainierbaren Parametern (Diagramm 08b).
   - Das spricht für die Stärke der vortrainierten ViT-Merkmale. Es ist aber kein allgemeiner Vergleich zwischen CNN und ViT, weil die Modelle sehr unterschiedlich groß sind (11,2 gegen 86 Mio. Parameter).
8. **Rechenaufwand**
   - Eine LoRA-Epoche dauert 12 bis 18 Minuten, eine ResNet18-Epoche 2 Minuten. Linear Probing braucht nach der einmaligen Merkmalsberechnung (10,5 min) nur Sekunden.
   - Die Dauer je Epoche steigt nicht mit dem LoRA-Rang (r8 13,0 min, r16 15,9 min, r4 16,7 min). Woher die Unterschiede kommen, lässt sich nicht sagen: Je Rang gibt es nur einen Lauf, und die Systemlast wurde nicht kontrolliert. Einem Rang-Effekt lassen sie sich nicht zuordnen.

## Was sich nicht ableiten lässt

- **Leistung auf neuen Daten:** Alle Werte dieses Ordners sind Validierungswerte, und die Validierung wurde auch zur Auswahl der besten Epoche benutzt. Sie können deshalb zu günstig ausfallen. Testwerte stehen in `oxford_test/`.
- **Streuung zwischen Trainingsläufen:** nicht gemessen (ein Lauf je Konfiguration, Seed 42). Die Intervalle ersetzen sie nicht.
- **Abhängige Testbilder:** Tieridentitäten sind für den Oxford-Test nicht dokumentiert. Ob mehrere Testbilder dasselbe Tier zeigen, ist offen; in diesem Fall könnten die Testintervalle zu schmal sein.
- **Wirkung des Rangs allein:** α ist fest, und die Budgets sind verschieden.
- **Fairer Budgetvergleich:** Linear Probing lief 30 Epochen, LoRA 5 oder 10, ResNet18 10, mit jeweils eigener Lernrate. Die Kurven zeigen, was jedes Budget erreicht hat, nicht was bei gleichem Budget möglich wäre.
