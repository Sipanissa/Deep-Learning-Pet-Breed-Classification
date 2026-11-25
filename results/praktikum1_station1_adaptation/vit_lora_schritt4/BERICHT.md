# Station 1, Schritt 4: ViT-B/16 mit LoRA vorbereitet

**Ergebnis:** LoRA ist auf Windows mit PyTorch 2.4.1 und DirectML zuverlässig umsetzbar, aber nur mit zwei Umgehungen für Fehler in torch-directml. Damit sind alle Tests bestanden. Ein Training lief nicht.

Code: `vit_lora.py`. Tests: `tools/check_vit_lora.py`, `tools/measure_vit_lora_memory.py`.

## Aufbau in torchvision

Jede der 12 Encoder-Schichten hat eine `nn.MultiheadAttention` mit **einer kombinierten QKV-Projektion** `in_proj_weight` (2.304 × 768: Zeilen 0–767 Query, 768–1.535 Key, 1.536–2.303 Value) und `out_proj`. Getrennte Query- oder Value-Layer gibt es nicht. `peft` ist nicht installiert und unterstützt diese Form ohnehin nur eingeschränkt.

**Umsetzung:** `LoRAAttention` ersetzt die Attention jeder Schicht. Sie nutzt die eingefrorenen `in_proj_weight`, `in_proj_bias` und `out_proj` und addiert `scaling · B(A(dropout(x)))` nur auf Query und Value. Key und alle übrigen Gewichte bleiben unverändert. Die Attention ist mit einfachen Tensoroperationen geschrieben (Softmax, Matmul) und läuft vollständig auf der GPU.

## Konfigurationen

| Rank | Alpha | Skalierung | Dropout | Ziele | LoRA-Parameter | Trainierbar mit Kopf |
|---:|---:|---:|---:|---|---:|---:|
| 4 | 8 | 2,0 | 0,1 | Query, Value, 12 Schichten | 147.456 | 155.146 |
| **8** | **16** | **2,0** | **0,1** | Query, Value, 12 Schichten | **294.912** | **302.602** |
| 16 | 32 | 2,0 | 0,1 | Query, Value, 12 Schichten | 589.824 | 597.514 |

Alpha ist jeweils das Doppelte des Rangs. So bleibt die Skalierung bei allen drei Rängen gleich, und die Ränge lassen sich fair vergleichen. A wird mit Kaiming-uniform initialisiert, B mit 0: Zu Beginn ist das Modell exakt der vortrainierte ViT. Der Kopf ist `Linear(768, 10)` mit 7.690 Parametern. Gespeichert in `lora_r4_config.json`, `lora_r8_config.json`, `lora_r16_config.json`.

## Gefundene DirectML-Fehler und ihre Umgehung

| Problem in torch-directml 0.2.5 | Wirkung | Umgehung |
|---|---|---|
| Backward von LayerNorm mit eingefrorenen Gewichten | **Treiberabsturz** („GPU-Geräteinstanz angehalten“); danach ist die GPU im Prozess unbrauchbar | `FrozenLayerNorm`: dieselbe Rechnung von Hand, mit denselben eingefrorenen Gewichten |
| Backward von GELU bei einem eingehenden Gradienten mit Schrittweite 0, etwa aus `sum().backward()` | **Still falscher Gradient** | Tritt im Training mit CrossEntropyLoss nicht auf; durch den Gradientenvergleich unten abgesichert |
| `lerp` in `torch.optim.AdamW` | CPU-Rückfall | `AdamWNoLerp`: gleiche Rechnung ohne `lerp`, Abweichung zu PyTorch-AdamW 1·10⁻⁷ |
| Schneller Pfad von `nn.MultiheadAttention` | CPU-Rückfall | Entfällt durch die eigene Attention |

Fehlermeldungen von Windows kommen auf Deutsch und lassen sich in Python nicht als UTF-8 lesen. Sie erscheinen deshalb zunächst als `UnicodeDecodeError`.

## Technischer Test (Rank 8)

| Prüfung | Ergebnis |
|---|---|
| Mit B = 0 gleich dem Originalmodell | Abweichung 4·10⁻⁷ |
| LoRA gleich den zusammengeführten Q/V-Gewichten | Abweichung 6·10⁻⁶; Key unverändert |
| **Gradienten des ganzen Modells: DirectML gegen CPU mit eingebauter LayerNorm** | Alle 50 Tensoren, größte relative Abweichung 3·10⁻⁵ |
| Forward, Backward, 3 Optimierungsschritte, Validierung | Ohne CPU-Rückfall; Loss 2,338 → 1,879 → 1,460 |
| Gradienten | Nur LoRA und Kopf; LoRA-A im ersten Schritt exakt 0, weil B bei 0 startet |
| Nach den Schritten | Alle 50 LoRA- und Kopftensoren verändert, Prüfsumme aller eingefrorenen Gewichte gleich |
| Validierung an 64 Bildern | Loss, Accuracy und Macro-F1 berechnet (nach 3 Schritten erwartungsgemäß niedrig) |
| Checkpoint speichern und laden | Nur LoRA, Kopf und Konfiguration, 1,2 MB; Ausgabe nach dem Laden identisch |
| Reproduzierbarkeit auf DirectML | Zwei Läufe mit Seed 42 identisch |
| Daten | 12.768 Training und 1.233 Validierung, Vorverarbeitung von `IMAGENET1K_V1`; keine Oxford-Testbilder geöffnet |
| Unverändert | Datensatzlisten, Linear-Probing-Checkpoint |

## Speicher und Stabilität (frischer Prozess je Batchgröße)

| Batch | Dediziert | Ausgelagert | Bilder/s | Trainingsepoche | Status |
|---:|---:|---:|---:|---:|---|
| 8 | 6,39 GB | 0,4 GB | 14,6 | ≈ 14,5 min | stabil |
| 16 | 6,73 GB | 1,2 GB | 13,9 | ≈ 15,3 min | stabil |
| 32 | 6,83 GB | 1,3 GB | 13,1 | ≈ 16,3 min | stabil |

Andere Programme belegen rund 1,6 GB GPU-Speicher. Damit sind die 8 GB der RX 6600 bei jeder Batchgröße ausgeschöpft, und DirectML lagert in den Arbeitsspeicher aus. Der Durchsatz hängt dadurch kaum von der Batchgröße ab. Gemessen wurden jeweils 6 Schritte; ob es über Stunden stabil bleibt, ist offen. Die Bildladezeit bei 12.768 Bildern je Epoche ist in den Schätzungen nicht enthalten.

## Empfehlung für die ersten vollständigen Experimente

1. **Reihenfolge:** zuerst Rank 8, dann Rank 4 und 16 mit sonst gleicher Einstellung.
2. **Batch 32:** Gleich wie beim Linear Probing, stabil, kaum langsamer als kleinere Batches. Bei Instabilität auf Batch 16 ausweichen.
3. **Lernraten:** LoRA-Parameter 5·10⁻⁴, Kopf 1·10⁻³ wie beim Linear Probing; AdamWNoLerp mit Weight Decay 0,01.
4. **Epochen:** 10 Epochen je Rang, Auswahl nach Validierungs-Macro-F1. Das sind etwa 3 Stunden je Rang und 8 bis 9 Stunden für alle drei.
5. **Checkpoints:** Weil ein Lauf Stunden dauert, nach jeder Epoche einen Checkpoint mit Optimizer-Zustand speichern und das Fortsetzen einbauen. `save_lora_checkpoint` nimmt dafür zusätzliche Inhalte auf.
6. **Ablauf:** Jeden Lauf in einem eigenen Prozess starten und andere GPU-Programme schließen, um Auslagerung zu vermeiden.

Die eigentliche Trainingsroutine für LoRA ist noch nicht geschrieben; das wäre der nächste Schritt.
