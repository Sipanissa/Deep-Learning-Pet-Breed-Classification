# Vorbereitete Datensatzvariante `20261106T110038Z_15000`

**15.000 Bildbeispiele: 7.130 Originalbilder und 7.870 augmentierte Trainingsbilder.** Erzeugt am 06.11.2026 aus der aktiven Version `20261106T110038Z`.

Der Versionszeiger `LATEST_COMBINED_VERSION.txt` bleibt auf `20261106T110038Z`. Diese Variante wird im Training gezielt mit `--dataset` gewählt. Nicht gelaufen: Modelltraining, Hyperparameter-Suche, Oxford-Testauswertung.

## Verteilung

| Klasse | Train original | Train augmentiert | Train gesamt | Val | Test | Gesamt |
|---|---:|---:|---:|---:|---:|---:|
| British Shorthair | 494 | 783 | 1.277 | 119 | 100 | 1.496 |
| Russian Blue | 627 | 650 | 1.277 | 113 | 100 | 1.490 |
| Persian | 940 | 336 | 1.276 | 243 | 100 | 1.619 |
| Ragdoll | 912 | 364 | 1.276 | 258 | 100 | 1.634 |
| Bengal | 446 | 831 | 1.277 | 117 | 100 | 1.494 |
| Samoyed | 303 | 974 | 1.277 | 78 | 100 | 1.455 |
| Pomeranian | 443 | 834 | 1.277 | 123 | 100 | 1.500 |
| Keeshond | 217 | 1.060 | 1.277 | 53 | 99 | 1.429 |
| Great Pyrenees | 275 | 1.002 | 1.277 | 68 | 100 | 1.445 |
| Newfoundland | 241 | 1.036 | 1.277 | 61 | 100 | 1.438 |
| **Gesamt** | **4.898** | **7.870** | **12.768** | **1.233** | **999** | **15.000** |

Bis zum Ziel von 20.000 Originalbildern fehlen weiterhin 12.870. Augmentierte Bilder zählen dafür nicht.

## Struktur

| Teil | Datei | Bilder liegen |
|---|---|---|
| Train original | `train_original.csv` | Unverändert am bisherigen Ort, nicht kopiert |
| Train augmentiert | `train_augmented.csv` | `images/<klasse>/` (rund 900 MB) |
| Train gesamt | `train.csv` | Original und augmentiert |
| Val | `val.csv` | Unverändert am bisherigen Ort |
| Test | `test.csv` | Unverändert am bisherigen Ort |

Herkunft: `augmentation_manifest.csv` (jedes Bild mit Original, Prüfsummen, Seed, Parametern, Versuch, Größe, pHash).

Weitere Dateien: `allocation.csv`, `augmentation_config.json`, `augmentation_replacements.csv`, `augmentation_leakage_check.csv`, `pruefung_vollstaendig.log`, `pruefung_val_andere_klasse.csv`, `pruefung_nahe_paare_verschiedene_originale.csv`.

## Augmentationsregeln

| Schritt | Parameter |
|---|---|
| Horizontal spiegeln | Wahrscheinlichkeit 0,5 |
| Drehen | ±10°, Ausschnitt ohne leere Ecken |
| Zuschnitt | Fläche 85–100 %, Position 25–75 % des Spielraums (zur Mitte hin), Seitenverhältnis 0,9–1,1 |
| Helligkeit, Kontrast | Faktor 0,85–1,15 |
| Sättigung | Faktor 0,90–1,10 |
| Leichte Unschärfe | Wahrscheinlichkeit 0,1, Radius 0,3–1,0 |
| Speichern | JPEG Qualität 95, 4:4:4, kurze Seite mindestens 224 Pixel |

Nicht verwendet: Farbton, Graustufen, vertikales Spiegeln, Erasing, Mixup, CutMix.

## Reproduzierbarkeit

- Grund-Seed 20261106. Bild-Seed aus Grund-Seed, Prüfsumme des Originals und Variantennummer.
- War eine Variante zu nah an Val oder Test, wurde sie mit einem aus denselben Werten und der Versuchsnummer abgeleiteten Seed neu gezogen. Das betraf 22 Bilder, alle in `augmentation_replacements.csv` belegt.
- Stichprobe: 200 zufällige Bilder wurden aus dem Manifest bytegleich neu berechnet.
- Python 3.11.9, Pillow 12.3.0. Aufruf: `prepare_augmentation.py plan`, `generate --confirm`, `verify_augmented_version.py <Ordner>`.

SHA-256 (Anfang): `train.csv` 48db896241f03243, `train_original.csv` 20cf29d61ca47bb2, `train_augmented.csv` d23346223c6b722c, `val.csv` 4f4a266f61fc7319, `test.csv` 158e85fc7cf32cbe, `augmentation_manifest.csv` d320058d95c14962.

## Prüfergebnisse

| Prüfung | Ergebnis |
|---|---|
| Aktive Version vor Beginn | `20261106T110038Z`; alle 7.130 Bilder geladen, Labels gegen Pfade ohne Widerspruch |
| Bilder geladen über PetDataset | 12.768 / 1.233 / 999, keine Fehler |
| Labels | 10 Klassen; jedes augmentierte Bild hat die Klasse seines Originals |
| Zahlen je Klasse | Wie geplant |
| Herkunft | Alle Originale aus Train (3.746 verschiedene), keines aus Val oder Test |
| Exakte Duplikate | Keine in 15.000 Prüfsummen |
| Auflösung Train | Kurze Seite mindestens 224, Median 419 Pixel |
| Train zu Test, pHash ≤ 10 in 8 Ausrichtungen | Keine Treffer |
| Train zu Val, gleiche Klasse, pHash ≤ 10 | Keine Treffer |
| Train zu Val, andere Klasse, pHash = 10 | 23 Paare, vermutlich Zufall, nicht am Bild angesehen, nicht blockierend |
| Val, Test, Train original | Listen bytegleich zur aktiven Version; `test.csv` bytegleich zum Original-Split |
| Trainingsskripte mit `--check-only` | CNN und ResNet18 laden die Variante und rechnen einen Batch |

**Nahe Paare innerhalb Train (pHash ≤ 5, Originalausrichtung):** 214 Paare gleicher Herkunft sind bei Augmentation zu erwarten. 1 Paar stammt aus zwei verschiedenen Originalen (Bengal, Abstand 4); die Dateinamen deuten auf aufeinanderfolgende Fotos einer Catza-Serie hin. Beide liegen in Train und wurden nicht entfernt.

**Sichtkontrolle:** 70 zufällige Paare aus Original und augmentiertem Bild wurden angesehen.
- Farben und Fellmerkmale blieben erkennbar, leere Ecken traten nicht auf.
- In einer ersten Fassung mit 80–100 % Zuschnitt schnitt ein Bild den Kopf fast ganz ab. Der Zuschnitt wurde deshalb enger gefasst und alles neu erzeugt.
- In der Endfassung bleiben Einzelfälle: Bei 2 von 70 Bildern ist der Kopf am Rand angeschnitten (Beispiele #30 Ragdoll, #56 Great Pyrenees). Bei 7.870 Bildern ist mit weiteren solchen Fällen zu rechnen. Die Quote ist wegen der kleinen Stichprobe nur ein Anhaltspunkt.

## Einschränkungen

- **Schwache Originale werden vervielfacht:** Die Augmentation übernimmt auch Schwächen der Quellen, etwa mehrere Tiere, dominante Personen, Graustufen oder einen roten Rahmen. Catza-, Oxford- und Stanford-Bilder wurden für diesen Schritt nicht neu gesichtet.
- **Wiederverwendung:** Keeshond und Newfoundland bestehen im Training zu über 80 % aus Varianten. Das gleicht die Klassen aus, bringt aber keine neuen Tiere.
- **Dasselbe Tier in mehreren Splits:** Das lässt sich außerhalb von Catza weiterhin nicht ausschließen.
- **Rechte:** Augmentierte Bilder erben die Bedingungen ihrer Originale. 469 stammen aus ImageNet, 2.713 aus Stanford Dogs. Nicht öffentlich weitergeben.
- **Pfade:** Die Listen enthalten für viele Bilder absolute Windows-Pfade. Auf einem anderen Rechner müssten sie angepasst werden.
- **Trainingsstart:** Das Skript `train_*.py` muss im Projektordner gestartet werden.

## Noch nötig für den Modellvergleich

- Drittes Modell (Architektur noch nicht genannt)
- Gemeinsame Testauswertung für den Oxford-Test
