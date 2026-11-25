# Aktive Datensatzversion 20261106T110038Z

Aktiviert am 06.11.2026 nach dem ADOPT-Eintrag und der Freigabe von Sipan Issa, aus `KANDIDAT_20261106T101453Z`.

- **Gesamt:** 7.130 Originalbilder (4.898 Train, 1.233 Val, 999 Test)
- **Bis 20.000 fehlen:** 12.870
- **Vorherige Version, unverändert erhalten:** `combined_versions/20261104T194414Z` (8.307 Bilder)
- **Datenaugmentation und Training:** nicht gestartet

## Verteilung je Klasse

| Klasse | Train | Val | Test | Gesamt | Fehlt bis 2.000 |
|---|---:|---:|---:|---:|---:|
| British Shorthair | 494 | 119 | 100 | 713 | 1.287 |
| Russian Blue | 627 | 113 | 100 | 840 | 1.160 |
| Persian | 940 | 243 | 100 | 1.283 | 717 |
| Ragdoll | 912 | 258 | 100 | 1.270 | 730 |
| Bengal | 446 | 117 | 100 | 663 | 1.337 |
| Samoyed | 303 | 78 | 100 | 481 | 1.519 |
| Pomeranian | 443 | 123 | 100 | 666 | 1.334 |
| Keeshond | 217 | 53 | 99 | 369 | 1.631 |
| Great Pyrenees | 275 | 68 | 100 | 443 | 1.557 |
| Newfoundland | 241 | 61 | 100 | 402 | 1.598 |
| **Gesamt** | **4.898** | **1.233** | **999** | **7.130** | **12.870** |

## Herkunft

| Quelle | Train | Val | Test | Gesamt | Status |
|---|---:|---:|---:|---:|---|
| Catza | 2.958 | 744 | | 3.702 | CC BY 3.0, Urheber je Bild |
| Oxford-IIIT Pet | 755 | 182 | 999 | 1.936 | CC BY-SA 4.0 für den Datensatz, Fotorechte bei den Inhabern |
| Stanford Dogs | 779 | 186 | | 965 | Aus ImageNet, keine Lizenz je Bild |
| ImageNet | 205 | 55 | | 260 | Nichtkommerzielle Forschung und Lehre, ADOPT vom 06.11.2026 |
| Open Images | 103 | 30 | | 133 | CC BY 2.0 je Datei |
| Openverse | 93 | 36 | | 129 | CC BY, CC BY-SA oder CC0 je Datei |
| FeliBreed | 5 | | | 5 | CC BY 4.0 für den Datensatz |

## Bestandene Kontrollen

Das Protokoll steht in `data/dataset_expansion/activation_20261106.log`.

- ADOPT-Eintrag, `ACCEPTANCE.json` und Prüfsumme der Liste mit den 260 Bildern
- Kandidatenlisten identisch zu den geprüften Listen; die neuen Listen unterscheiden sich davon nur in der Herkunftsangabe der 260 ImageNet-Zeilen
- `test.csv` bytegleich zur vorherigen Version und zum Original-Split; alle 999 Testbilder unter den 4.153 geschützten Dateien, 0 geändert
- Trainingsskripte und `data/final_splits/*.csv` unverändert
- Alle Bilder lesbar, Train und Val mindestens 224 Pixel, keine exakten Duplikate
- pHash in 8 Ausrichtungen: kein Paar ≤ 5, kein Train- oder Val-Bild ≤ 10 zum Test, kein Paar ≤ 10 über Train und Val in derselben Klasse
- 10 Klassen, Grenze von 2.000 eingehalten, PetDataset erfolgreich
- Ordner der vorherigen Version vor und nach der Aktivierung identisch

## Einschränkungen

- **Training:** `train_cnn.py` und `train_resnet18.py` lesen weiter `data/final_splits/*.csv` (`train.csv` mit 2.510 Zeilen). Die aktive Version wirkt erst, wenn das vor dem Training gesondert geklärt ist.
- **Neu bauen:** `build_combined_version.py` startet von der Catza-Basis und würde ausgeschlossene Bilder wieder aufnehmen. Eine künftige Version muss von dieser Version oder ihren Manifesten ausgehen.
- **Statusübersicht:** `STATUS_COUNTS.md` zählt unter „Akademischer Pool“ auch 286 Bilder, die in den Pool-Manifesten noch ausgewählt sind, aber in der Abschlusskontrolle oder durch den Testschutz herausfielen. Sie sind nicht freigegeben. Maßgeblich ist `ausschluss_manifest.csv`.
- **Weitergabe:** ImageNet-Bilder nur an Personen, die den Bedingungen vorher zugestimmt haben. Keine öffentliche Weitergabe.
- **Dasselbe Tier in mehreren Splits:** lässt sich außerhalb von Catza nicht ausschließen, auch nicht gegenüber dem Test.
- **Sichtung und Rechte:** KI-Einschätzungen ohne fachliche Rassebestätigung. Eine rechtliche Prüfung hat nicht stattgefunden.

## Zurücksetzen

In `data/dataset_expansion/LATEST_COMBINED_VERSION.txt` den Pfad der vorherigen Version eintragen:
`C:\Users\Sipan\DeepLearning\data\dataset_expansion\combined_versions\20261104T194414Z`
