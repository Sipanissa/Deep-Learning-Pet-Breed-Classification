"""Pretrained ViT-B/16 prepared for linear probing: frozen backbone, new trainable classification head."""
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from pet_dataset import CLASS_NAMES

WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1


def build_vit_linear_probe(num_classes=len(CLASS_NAMES)):
    """Returns the model and the preprocessing that belongs to its weights."""
    # Vortrainiertes Modell laden
    model = vit_b_16(weights=WEIGHTS)
    # Gewichte des Backbones einfrieren
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Neuer Klassifikationskopf für unsere zehn Rassen
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model, WEIGHTS.transforms()


# Nur der Kopf ist im Trainingsmodus
def set_probe_mode(model):
    """Use for training and validation: frozen backbone in evaluation mode, only the head trains.

    In evaluation mode PyTorch's attention takes a fast path that DirectML runs on the CPU. The attention modules stay
    in training mode to avoid that; this changes no result, because all their dropout rates are zero.
    """
    model.eval()
    model.heads.train()
    for module in model.modules():
        # Sonst würde DirectML diese Schichten auf der CPU rechnen.
        # Das Ergebnis bleibt gleich, weil hier kein Dropout aktiv ist.
        if isinstance(module, nn.MultiheadAttention):
            assert module.dropout == 0.0
            module.train()


# Merkmale eines Bildes berechnen, ohne den Kopf
def vit_features(model, images):
    """The class-token features that enter the head; same steps as VisionTransformer.forward without self.heads."""
    x = model._process_input(images)
    # Klassen-Token vor die Bildausschnitte setzen
    x = torch.cat([model.class_token.expand(x.shape[0], -1, -1), x], dim=1)
    return model.encoder(x)[:, 0]


# Gespeicherten Kopf laden und Modell wieder zusammensetzen
def load_linear_probe(checkpoint_path):
    """Rebuilds the full model from the pretrained weights and a saved head checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Checkpoint muss zu Gewichten und Klassen passen
    assert checkpoint["weights"] == str(WEIGHTS) and checkpoint["classes"] == CLASS_NAMES
    model, preprocess = build_vit_linear_probe()
    model.heads.head.load_state_dict(checkpoint["head"])
    return model, preprocess, checkpoint
