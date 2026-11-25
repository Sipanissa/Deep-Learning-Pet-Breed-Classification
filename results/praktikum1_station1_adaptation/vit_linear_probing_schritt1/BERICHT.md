# Station 1, Schritt 1: ViT-B/16 für Linear Probing

**Ergebnis:** Das Modell ist vorbereitet, der Funktionstest ist bestanden. Es lief kein Training, und es wurde kein Modell gespeichert.

Code: `vit_linear_probe.py`. Test: `tools/check_vit_linear_probe.py`. Rohdaten: `funktionstest.json`, `funktionstest.log`, `architektur.txt`.

## Umgebung

Python 3.11.9, PyTorch 2.4.1+cpu, torchvision 0.19.1, torch-directml 0.2.5.dev240914, AMD Radeon RX 6600, Windows 11.

## Modell

| Punkt | Wert |
|---|---|
| Architektur | ViT-B/16: Patchgröße 16, Bildgröße 224, 197 Tokens, 12 Encoder-Schichten, 12 Attention-Köpfe, Einbettung 768, MLP 3.072 |
| Gewichte | `ViT_B_16_Weights.IMAGENET1K_V1`, Top-1 auf ImageNet-1K laut torchvision 81,072 % |
| Datei | `vit_b_16-c867db91.pth`, SHA-256 beginnt mit `c867db91`, im Torch-Cache des Benutzers |
| Klassifikationskopf | `Linear(768, 10)` |
| Parameter gesamt | 85.806.346 |
| Trainierbar | 7.690 (7.680 Gewichte, 10 Bias), nur `heads.head` |
| Eingefroren | 85.798.656 |

Alle 150 Tensoren des Backbones stimmen mit dem Checkpoint überein.

## Daten und Vorverarbeitung

| Punkt | Ergebnis |
|---|---|
| Training | `20261106T110038Z_15000/train.csv`, 12.768 Bildbeispiele, 1.276 bis 1.277 je Klasse |
| Validierung | 1.233 Originalbilder, `val.csv` bytegleich zur aktiven Version |
| Oxford-Test | Nur Prüfsumme der Liste geprüft, keine Testbilder geladen |
| Klassenreihenfolge | 0 British Shorthair, 1 Russian Blue, 2 Persian, 3 Ragdoll, 4 Bengal, 5 Samoyed, 6 Pomeranian, 7 Keeshond, 8 Great Pyrenees, 9 Newfoundland; Stichprobe je Klasse und Split stimmt |
| Vorverarbeitung | Größe auf 256 (bilinear), Mittenzuschnitt 224, Normalisierung mit ImageNet-Werten; identisch zu einer manuellen Berechnung |

## Funktionstest

| Prüfung | Ergebnis |
|---|---|
| Forward auf DirectML | 10 Bilder, je eins pro Klasse; Ausgabe 10 × 10 |
| Loss | 2,358; erwartbar bei zufälligem Kopf (ln 10 ≈ 2,303) |
| Gradienten | Nur `heads.head.weight` und `heads.head.bias`; alle anderen Parameter ohne Gradient |
| Parameter nach dem Backward | Unverändert, es lief kein Optimizer-Schritt am Modell |
| Merkmale DirectML gegen CPU | Größte Abweichung 0,000011 |
| Validierungsbatch | Läuft, Ausgaben endlich |

## DirectML

**Kompatibel, mit zwei Eigenheiten:**

1. **Attention:** Im reinen `eval()`-Modus nutzt PyTorch einen schnellen Pfad (`_native_multi_head_attention`), der auf DirectML zur CPU zurückfällt. `set_probe_mode()` lässt die Attention-Module deshalb im Trainingsmodus. Ihr Dropout ist 0, die Merkmale ändern sich nicht. Die Zeit für 32 Bilder sinkt von 1,68 s auf 0,85 s.
   Für Training und Validierung muss deshalb `set_probe_mode()` statt `model.eval()` verwendet werden.
2. **AdamW:** `lerp` läuft auf der CPU. Getestet auf einer Kopie des Kopfes, dort lief der Schritt fehlerfrei. Das betrifft nur die 7.690 Kopfparameter.

Eine Epoche mit Validierung dauert grob 7 Minuten. Diese Schätzung beruht auf 32 Bildern, die reale Ladezeit kann höher liegen.

## Hinweise für Schritt 2

- Da der Backbone fest ist, ließen sich die Merkmale einmal berechnen und zwischenspeichern; das würde die weiteren Epochen stark verkürzen.
- Die Validierung ist unausgewogen: Ragdoll 258, Keeshond 53 Bilder. Neben der Accuracy wäre ein Macro-F1 sinnvoll.

## Offener Befund außerhalb dieses Schritts

`train_cnn.py` fehlt im Projekt und liegt seit 07.11.2026, 09:46 Uhr im Papierkorb. Der Test für den ViT hängt nicht davon ab. `tools/check_project_integrity.py` schlägt dadurch aber fehl, weil die Datei geschützt ist.
