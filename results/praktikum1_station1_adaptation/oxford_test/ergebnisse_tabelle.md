# Ergebnisse auf dem Oxford-Test

| Konfiguration | Beste Epoche | Val-Macro-F1 | Richtig | Accuracy (95-%-Wilson) | Macro-F1 (95-%-Bootstrap) | Macro-Precision | Macro-Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViT-B/16 Linear Probing | 11 | 0,9735 | 966/999 | 96,70 % (95,4–97,6) | 0,9670 (0,955–0,978) | 0,9677 | 0,9670 |
| ViT-B/16 LoRA r = 4 | 1 | 0,9773 | 970/999 | 97,10 % (95,9–98,0) | 0,9710 (0,960–0,981) | 0,9722 | 0,9710 |
| ViT-B/16 LoRA r = 8 | 3 | 0,9760 | 973/999 | 97,40 % (96,2–98,2) | 0,9741 (0,964–0,983) | 0,9748 | 0,9740 |
| ViT-B/16 LoRA r = 16 | 3 | 0,9766 | 973/999 | 97,40 % (96,2–98,2) | 0,9741 (0,963–0,984) | 0,9750 | 0,9740 |
| ResNet18 Fine-Tuning | 10 | 0,9359 | 929/999 | 92,99 % (91,2–94,4) | 0,9286 (0,911–0,944) | 0,9327 | 0,9300 |

Ohne Bilder mit pHash-Abstand ≤ 10 zu ImageNet-Bildern (mögliche Vortrainingsüberschneidung):

| Konfiguration | Entfernt | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| ViT-B/16 Linear Probing | 35 | 96,58 % | 0,9660 |
| ViT-B/16 LoRA r = 4 | 35 | 96,99 % | 0,9699 |
| ViT-B/16 LoRA r = 8 | 35 | 97,30 % | 0,9732 |
| ViT-B/16 LoRA r = 16 | 35 | 97,30 % | 0,9732 |
| ResNet18 Fine-Tuning | 35 | 93,05 % | 0,9297 |
