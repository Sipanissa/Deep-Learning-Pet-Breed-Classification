# Abschlussbericht: Korrektur der Diagramme, Station 1 (25.11.2026)

Grundlage: wissenschaftlicher Prüfbericht vom 25.11.2026. Kein Training, kein Oxford-Testlauf, keine geänderten Messwerte, Checkpoints oder Datensätze. Das 7-Seiten-Paper, sein Overleaf-Projekt und alle LaTeX-Dateien sind unverändert.

## Geänderte Diagramme

Pfade ab `results/praktikum1_station1_adaptation/`, jeweils PNG und PDF; `bericht/abbildungen/` nur PDF.

| Datei | Korrektur |
|---|---|
| `oxford_test/diagramme/f1_pro_klasse` | Achsenzahlen mit drei Nachkommastellen (0,800 bis 1,000 in Schritten von 0,025) |
| `oxford_test/diagramme/test_vs_validierung` | Werte liegen nicht mehr auf der Achse, Legende für Test und Validierung, Hinweise zu Unabhängigkeit und anderen Bildquellen |
| `diagramme/03_val_accuracy_vergleich` | Tierbasiertes Intervall neben Wilson, Legende, Einschränkungen |
| `diagramme/04_val_macro_f1_vergleich` | Gespeicherte Bootstrap-Intervalle der Macro-F1, Einschränkungen, dunkler Sternrand |
| `diagramme/05_lora_raenge` | Unbelegte Aussage zur Laufschwankung entfernt. Macro-F1-Intervalle und tierbasierte Accuracy-Intervalle ergänzt. Accuracy- und Macro-F1-Achse gleich breit (6 Prozentpunkte, 0,06). Dauer mit Min. bis Max., Zahlenformat vereinheitlicht |
| `diagramme/08_dauer_und_komplexitaet` | Punktdiagramme statt Balken auf Log-Achse, Hardware je Schritt, Linear Probing nicht mehr im Epochenvergleich, eingeschränkte Vergleichbarkeit benannt, Textüberdeckung behoben |
| `diagramme/08b_macro_f1_gegen_parameter` | Trainierbare (gefüllt) und gesamte Parameter (hohl) unterschieden, Macro-F1-Intervalle, Hinweis zur Vermischung |
| `oxford_test_vorbereitung/validierungspruefung/diagramme/test_vs_validierung` | Sinnloser Selbstvergleich entfernt, Legende, tierbasiertes Intervall, Einschränkungen |
| `bericht/abbildungen/lora_raenge`, `dauer_und_komplexitaet`, `f1_gegen_parameter` | Dieselben Korrekturen wie 05, 08 und 08b in Druckgröße |

Neue Skripte: `tools/correct_station1_diagrams.py` (Diagramme), `tools/animal_bootstrap_validation.py` (Tier-Bootstrap). Quellen je Diagramm: `datenquellen_korrektur.json`.

## Sicherung

`sicherung_originale/` enthält alle Dateien der vier Diagrammordner vor der Korrektur, mit Prüfsummen in `PRUEFSUMMEN.sha256`.

## Unsicherheitsintervalle

- **Macro-F1:** gespeicherte Bootstrap-Intervalle aus `validierungspruefung/metriken_*.json` (2.000 Ziehungen über Bilder). Sie nehmen unabhängige Bilder an und sind vermutlich zu schmal. Ein tierbasiertes Macro-F1-Intervall gab es nicht und wurde nicht ergänzt.
- **Accuracy, tierbasiert:** Die Werte standen bisher nur im Prüfbericht, ohne dokumentierte Datenbasis. Regel, Zuordnung je Bild, Quellen mit Prüfsummen und Ergebnis liegen jetzt in `tierbasierter_bootstrap/`. Die berichteten Werte sind exakt reproduziert (Regel A). Regel A zählte einige Katzen doppelt, weil Titel wie `FIFe CH` oder `SE*` im Namen blieben. Verwendet wird die bereinigte Regel B mit 234 statt 240 Katzen; die Grenzen ändern sich um höchstens 0,1 Prozentpunkte.
- **Grenze:** Nur Catza-Bilder haben Tiernamen. Bei anderen Quellen zählt jedes Bild einzeln.

| Modell | Wilson | Tier-Bootstrap, Prüfbericht (A) | Tier-Bootstrap, verwendet (B) |
|---|---|---|---|
| Linear Probing | 95,52–97,54 | 94,95–98,13 | 94,96–98,13 |
| LoRA r = 4 | 96,08–97,95 | 95,31–98,67 | 95,37–98,63 |
| LoRA r = 8 | 96,08–97,95 | 95,41–98,62 | 95,44–98,62 |
| LoRA r = 16 | 96,17–98,02 | 95,53–98,63 | 95,59–98,65 |
| ResNet18 | 92,71–95,34 | 92,30–95,82 | 92,35–95,73 |

## Nachprüfung

Die Werte wurden aus den Vektordaten der neuen PDFs zurückgerechnet, anhand der Gitterlinien kalibriert und unabhängig vom Plot-Code mit `training_log.csv`, `run_config.json`, Checkpoint-Matrizen, `metriken_*.json` und dem Merkmalsprotokoll verglichen (`nachpruefung/`).

| Diagramm | Geprüfte Werte | Abweichungen |
|---|---:|---:|
| 03 | 85 | 0 |
| 04 | 75 | 0 |
| 05 | 36 | 0 |
| 08 | 27 | 0 |
| 08b | 28 | 0 |
| Oxford `test_vs_validierung` | 40 | 0 |
| Oxford `f1_pro_klasse` | 50 | 0 |
| Validierungsprüfung `test_vs_validierung` | 40 | 0 |
| bericht `lora_raenge` | 36 | 0 |
| bericht `dauer_und_komplexitaet` | 27 | 0 |
| bericht `f1_gegen_parameter` | 28 | 0 |
| **Summe** | **472** | **0** |

Verlaufskurven in 03 und 04 wurden in der Form verglichen (bis auf eine lineare Achsenabbildung). Die Texte der Beschriftungen wurden in der Ansicht geprüft, die PNG-Dateien stammen aus derselben Zeichnung wie die PDFs. Alle übrigen Dateien des Projekts sind per Prüfsumme unverändert, die Integritätsprüfung ist bestanden.

## Kopien in anderen Ordnern

| Ort | Ergebnis |
|---|---|
| `bericht/abbildungen/lora_raenge`, `dauer_und_komplexitaet`, `f1_gegen_parameter` | korrigiert |
| `bericht/abbildungen/` (übrige) | korrekt, unverändert |
| `bericht/overleaf_projekt/diagramme/` und Overleaf-ZIP: `04`, `05`, `oxford_test_vs_validierung` | alte Fassung, müssten ersetzt werden; gehören zum 7-Seiten-Paper, daher unverändert |
| `oxford_test_vorbereitung/selbsttest*/` | synthetischer Technik-Nachweis, unverändert; nicht als Ergebnis verwenden |
| `01`, `02`, `06`, `07`, Oxford-`konfusionsmatrizen` | korrekt, unverändert |

## Offen

1. 7-Seiten-Paper: `04`, `05` und `oxford_test_vs_validierung` austauschen. Der Zuschnitt (`trim`) von `05` in `main.tex` passt nicht zur neuen Fassung und muss neu bestimmt werden; Bildunterschriften prüfen.
2. 10-Seiten-Fassung: Bildunterschriften zu `lora_raenge`, `dauer_und_komplexitaet` und `f1_gegen_parameter` anpassen.
3. `diagramme/UEBERSICHT.md`, `diagramme/datenquellen.json`: beschreiben 03, 04, 05, 08 und 08b noch in der alten Form. Aussage 1 in `UEBERSICHT.md` („überlappen nicht“) gilt nur für Wilson.
4. `plot_station1_adaptation.py` und `plot_paper_figures.py` erzeugen weiterhin die alten Fassungen. Nach einem Lauf dieser Skripte muss `correct_station1_diagrams.py` erneut laufen.
