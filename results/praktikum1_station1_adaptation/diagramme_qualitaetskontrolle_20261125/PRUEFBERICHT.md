# Wissenschaftlicher Prüfbericht: Qualitätskontrolle der korrigierten Diagramme, Station 1 (25.11.2026)

**Ergebnis: bestanden.** Alle elf korrigierten Diagramme stimmen mit den Originalmessdaten überein (502 gezeichnete Werte, 116 gedruckte Beschriftungen, 0 Abweichungen) und lassen sich bytegenau neu erzeugen. Es gab kein Training und keine Oxford-Testauswertung. Paper, LaTeX-Dateien, Modelle, Datensätze und Trainingsprotokolle sind unverändert. Die Diagrammbearbeitung ist abgeschlossen.

## 1. Sichtprüfung

Alle elf Diagramme wurden vor und nach der Überarbeitung gerendert angesehen. Die angesehenen Fassungen sind bytegleich mit den gespeicherten.

| Diagramm | Befund | Behebung |
|---|---|---|
| 03, 04 | Legende lag auf der Linie „Budgetende“; Stern von r = 8 von r = 16 verdeckt; doppelte Klammer in der Legende | Legende gekürzt, Ausschnitt „Epochen 1 bis 5, LoRA“, einheitlicher Sternrand |
| 03 | Achse endete knapp am Intervallende | x-Achse bis 99,2 |
| 05 | Titel ragte über das Feld; Beschriftung lag auf Gitterlinien | Titel gekürzt, Beschriftung hinterlegt |
| 08 | Min.–Max. von r = 16 und ResNet18 vom Punkt verdeckt | Werte stehen als Zahl dabei, z. B. „15,9 min (15,8–16,0)“ |
| 08b | Hohle Punkte bei 86 Mio. verdeckten sich; Beschriftung von r = 8 ohne Bezug | Punkte ohne Füllung, Hinweis, Verbindungslinie |
| Oxford `f1_pro_klasse` | Punkte überdeckten sich; Reihenfolge umgekehrt zur Legende; leerer Bereich unter 0,80 | Zeilen höher, gleiche Reihenfolge, x-Achse ab 0,795 |
| Validierungsprüfung `test_vs_validierung` | Titel nannte nur Wilson, gezeigt sind zwei Intervalle | Titel berichtigt |
| `lora_raenge` (Kopie) | Wert auf Gitterlinie; Legende ohne „geschätzt“ | behoben |
| `dauer_und_komplexitaet` (Kopie) | Achsenzahlen „150“ und „0“ stießen aneinander; „7,7 Tsd.“ berührte die Achse | Achsen angepasst |
| `f1_gegen_parameter` (Kopie) | wie 08b | wie 08b |
| Oxford `test_vs_validierung` | keine Layoutmängel | nur Hinweistext (siehe 3) |

Nach der Überarbeitung: keine störenden Überlappungen, keine abgeschnittenen Elemente (auch rechnerisch geprüft: jeder Text liegt innerhalb der Seite). Die Sterne von r = 8 und r = 16 liegen im Hauptfeld von 03 und 04 weiterhin teilweise übereinander, weil die Werte fast gleich sind; der Ausschnitt trennt sie. Achsen, die nicht bei null beginnen, sind gekennzeichnet.

## 2. Geschätzte Tieridentifikation

- Alle Legenden nennen „Bootstrap über geschätzte Tiere“, die Hinweistexte zusätzlich „nicht verifiziert“, in der Kopie `lora_raenge` „95 % je Tier (Tier geschätzt)“.
- 744 Catza-Bilder wurden anhand des Namens im Bildtitel 234 Katzen zugeordnet. Die übrigen 489 Validierungsbilder haben keine Tierangabe, jedes zählt einzeln.
- Gleiche Namen können verschiedene Katzen bezeichnen, verschiedene Schreibweisen dieselbe. Die Intervalle sind eine Näherung.
- Kennzeichnung auch in `animal_bootstrap_validation.py` und `ergebnis.json` (nur das Textfeld `einschraenkung`). Alle Zahlen und die Zuordnung je Bild sind nach erneutem Rechnen identisch.

## 3. Mögliche Abhängigkeiten der Oxford-Testbilder

- **Neue Formulierung:** „Tieridentitäten sind für den Test nicht dokumentiert; sollten mehrere Testbilder dasselbe Tier zeigen, könnten die Intervalle zu schmal sein.“ Die frühere Formulierung setzte solche Fälle voraus.
- **Fehler berichtigt:** Bisher stand im Diagramm, die Validierung habe andere Bildquellen als der Test. Tatsächlich stammen 182 der 1.233 Validierungsbilder wie der Test aus Oxford-IIIT Pet, 1.051 aus anderen Quellen (`val.csv`, `test.csv`). Der Hinweis nennt jetzt beide Zahlen.

## 4. Kennzeichnung der Unsicherheit

| Angabe | Bedeutung | Diagramme |
|---|---|---|
| 95-%-Wilson | Bildstichprobe, fester Checkpoint, unabhängige Bilder angenommen | 03, 05, beide `test_vs_validierung`, `lora_raenge` |
| 95-%-Bootstrap über Bilder | wie oben, Macro-F1, gespeicherte Werte (2.000 Ziehungen) | 04, 05, 08b, beide `test_vs_validierung`, `lora_raenge`, `f1_gegen_parameter` |
| 95-%-Bootstrap über geschätzte Tiere | wie oben, Accuracy, Näherung | 03, 05, Validierungsprüfung, `lora_raenge` |
| Min. bis Max. | Spanne der Epochendauer innerhalb eines Laufs, kein Intervall | 05, 08, `lora_raenge`, `dauer_und_komplexitaet` |
| Punktwerte ohne Intervall | ein anders klassifiziertes Bild ändert F1 um rund 0,005 | Oxford `f1_pro_klasse` |

Schwankungen zwischen Trainingsläufen wurden nicht gemessen (ein Lauf je Konfiguration, Seed 42). Die sechs Diagramme mit Intervallen in den Ergebnisordnern und Oxford `f1_pro_klasse` sagen das ausdrücklich; 05 und 08 stellen für Min.–Max. klar, dass es keine Streuung zwischen Läufen ist. Die drei kleinen Kopien enthalten dafür keinen Platz; dort gehört der Hinweis in die Bildunterschrift.

## 5. Reproduzierbarkeit

- `correct_station1_diagrams.py` ist die einzige Quelle der elf Diagramme. PDF-Dateien enthalten keinen Zeitstempel mehr.
- `correct_station1_diagrams.py --pruefen` zeichnet alles neu in einen temporären Ordner und vergleicht: 19 Dateien, 0 abweichend (Matplotlib 3.11.2; andere Versionen können andere Bytes erzeugen).
- **Alte Fassungen verhindert:** In `plot_station1_adaptation.py` und `plot_paper_figures.py` sind die alten Zeichenfunktionen entfernt; beide rufen die korrigierten auf. In Testordnern ausgeführt, liefern beide bytegleiche korrigierte Diagramme. Die stichprobenartig verglichenen unveränderten Diagramme (01, 02, 06-Übersicht, 07 als PNG) und beide CSV-Tabellen sind ebenfalls bytegleich.
- **`evaluate_station1_oxford_test.py`:** Zwei Zeilen in der Zeichenfunktion `make_plots` ergänzt. Vorhandene Diagramme werden nicht mehr überschrieben. Modelle, Auswertung und Kennzahlen sind nicht berührt. Geprüft mit gespeicherten Kennzahlen in einem Testordner, ohne Modelllauf. Grenze: In einem neuen, leeren Ausgabeordner würde das Skript seine ursprünglichen Diagramme zeichnen.

## 6. Dokumentation

- `diagramme/UEBERSICHT.md`: Erzeugung, Diagrammbeschreibungen, neuer Abschnitt „Unsicherheit“. Widerlegte oder überholte Aussagen zu den Diagrammen berichtigt: „Intervalle überlappen nicht“ (gilt nur für Wilson), Schwankung zwischen Läufen als Ursache der Dauer, „Oxford-Test nicht ausgewertet“, „McNemar nicht möglich“.
- `diagramme/datenquellen.json`: Beschreibungen und Quellen von 03, 04, 05, 08, 08b, Hinweis zu Intervallen. Einträge der übrigen Diagramme inhaltlich unverändert.

## 7. Vergleich mit den Originalmessdaten

Werte aus den Vektordaten der PDFs zurückgerechnet, an den Gitterlinien kalibriert, unabhängig vom Zeichenskript mit `training_log.csv`, `run_config.json`, Checkpoint-Matrizen, `metriken_*.json` und dem Merkmalsprotokoll verglichen. Verlaufskurven jetzt absolut an der Achse, nicht nur der Form nach.

| Diagramm | Werte | Abweichungen |
|---|---:|---:|
| 03 (davon Ausschnitt 15) | 100 | 0 |
| 04 (davon Ausschnitt 15) | 90 | 0 |
| 05 | 36 | 0 |
| 08 | 27 | 0 |
| 08b | 28 | 0 |
| Oxford `test_vs_validierung` | 40 | 0 |
| Oxford `f1_pro_klasse` | 50 | 0 |
| Validierungsprüfung `test_vs_validierung` | 40 | 0 |
| `lora_raenge` | 36 | 0 |
| `dauer_und_komplexitaet` | 27 | 0 |
| `f1_gegen_parameter` | 28 | 0 |
| **Summe** | **502** | **0** |

- 116 gedruckte Beschriftungen stimmen mit den Originalwerten überein, ebenso die Zahlen der Hinweistexte (1.233, 182, 1.051, 744, 489, 234, 999).
- Gegenprobe: Mit absichtlich verschobenen Sollwerten (Accuracy +0,05 Prozentpunkte, Macro-F1 +0,0005) meldet die Prüfung 103 Abweichungen.
- Ein erster Lauf meldete eine Abweichung bei `f1_pro_klasse`. Ursache war ein zu enger Positionsfilter im Prüfskript (oberster Punkt lag höher als früher); nach der Anpassung 0.
- Projektvergleich vorher/nachher: 27 Dateien geändert (19 Bilddateien, 5 Skripte, `UEBERSICHT.md`, `datenquellen.json`, Textfeld in `ergebnis.json`), keine gelöscht. Inhalt per SHA-256, der Ordner `data/` per Größe und Änderungszeit. Integritätsprüfung: ALLES BESTANDEN.

Sicherung des Stands vor dieser Kontrolle: `sicherung_vorher/` mit `PRUEFSUMMEN.sha256`. Nachprüfung: `nachpruefung/`.

## Hinweise außerhalb dieses Auftrags (nicht geändert)

1. **7-Seiten-Paper:** enthält noch die alten Fassungen von 04, 05 und Oxford `test_vs_validierung`. Beim Austausch den Zuschnitt (`trim`) von 05 und die Bildunterschriften anpassen.
2. **7-Seiten-Paper, `main.tex` Zeile 268:** „hat andere Quellen als der Test“ ist ungenau (182 Validierungsbilder stammen aus Oxford-IIIT Pet). „Enthält mehrere Fotos derselben Tiere“ beruht auf geschätzten Tieren.
3. **10-Seiten-Fassung:** bindet die drei Kopien aus `bericht/abbildungen/` ein. Seitengrößen sind gleich geblieben; die Bildunterschriften sind noch anzupassen und sollten den Hinweis auf den einen Trainingslauf enthalten.
4. `ABSCHLUSSBERICHT.md` der Korrektur beschreibt den Stand davor; seine offenen Punkte 3 und 4 sind erledigt.
