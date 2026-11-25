"""ViT-B/16 (IMAGENET1K_V1) with LoRA on the query and value projections; backbone frozen, only LoRA and head train.

torchvision's ViT uses nn.MultiheadAttention with one combined QKV projection (in_proj_weight, 2304 x 768). The
wrapper below keeps those frozen weights and the frozen output projection and adds the low-rank terms
scaling * B(A(dropout(x))) to the query and value parts only. With B = 0 it computes exactly the original attention.
Only plain tensor operations are used, so DirectML runs everything on the GPU.
"""
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from pet_dataset import CLASS_NAMES

WEIGHTS = ViT_B_16_Weights.IMAGENET1K_V1
CONFIGS = {4: dict(rank=4, alpha=8, dropout=0.1), 8: dict(rank=8, alpha=16, dropout=0.1), 16: dict(rank=16, alpha=32, dropout=0.1)}
TARGETS = ("query", "value")


# Attention-Schicht mit LoRA für Query und Value
class LoRAAttention(nn.Module):
    def __init__(self, attention, rank, alpha, dropout):
        super().__init__()
        assert attention.batch_first and attention._qkv_same_embed_dim and attention.dropout == 0.0
        self.num_heads = attention.num_heads
        self.embed_dim = attention.embed_dim
        # Vortrainierte Gewichte übernehmen, sie bleiben eingefroren
        self.in_proj_weight = attention.in_proj_weight
        self.in_proj_bias = attention.in_proj_bias
        self.out_proj = attention.out_proj
        # Stärke der LoRA-Korrektur
        self.scaling = alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        # B startet mit null. So rechnet das Modell am Anfang
        # genauso wie das vortrainierte.
        self.lora_A_query = nn.Parameter(torch.empty(rank, self.embed_dim))
        self.lora_B_query = nn.Parameter(torch.zeros(self.embed_dim, rank))
        self.lora_A_value = nn.Parameter(torch.empty(rank, self.embed_dim))
        self.lora_B_value = nn.Parameter(torch.zeros(self.embed_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_query, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_value, a=math.sqrt(5))

    def forward(self, query, key, value, need_weights=False):
        assert query is key is value and not need_weights
        x = query
        batch, tokens, _ = x.shape
        # Query, Key und Value mit den eingefrorenen Gewichten berechnen
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        # LoRA-Korrektur nur zu Query und Value addieren
        d = self.lora_dropout(x)
        q = q + F.linear(F.linear(d, self.lora_A_query), self.lora_B_query) * self.scaling
        v = v + F.linear(F.linear(d, self.lora_A_value), self.lora_B_value) * self.scaling
        # Ab hier normale Attention, aufgeteilt auf mehrere Köpfe
        head_dim = self.embed_dim // self.num_heads
        q, k, v = (t.reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2) for t in (q, k, v))
        attention = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)
        out = torch.matmul(attention, v).transpose(1, 2).reshape(batch, tokens, self.embed_dim)
        return self.out_proj(out), None


# Ersatz für LayerNorm, weil die eingebaute auf DirectML abstürzt
class FrozenLayerNorm(nn.Module):
    """LayerNorm written out with the same frozen weights.

    On DirectML (torch-directml 0.2.5) the backward pass of the built-in layer norm stops the GPU device ("device
    removed") when weight and bias need no gradient. The written-out form gives the same values and correct gradients.
    """

    def __init__(self, norm):
        super().__init__()
        self.weight, self.bias, self.eps = norm.weight, norm.bias, norm.eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.eps) * self.weight + self.bias


# ViT mit LoRA aufbauen
def build_vit_lora(rank=8, alpha=None, dropout=0.1, num_classes=len(CLASS_NAMES), directml_safe_layernorm=True):
    """Returns model, preprocessing and the configuration as a dictionary."""
    alpha = CONFIGS.get(rank, {}).get("alpha", 2 * rank) if alpha is None else alpha
    # Vortrainiertes Modell laden
    model = vit_b_16(weights=WEIGHTS)
    # Gewichte des Backbones einfrieren
    for parameter in model.parameters():
        parameter.requires_grad = False
    # LoRA-Schichten hinzufügen
    for layer in model.encoder.layers:
        layer.self_attention = LoRAAttention(layer.self_attention, rank, alpha, dropout)
        if directml_safe_layernorm:
            layer.ln_1, layer.ln_2 = FrozenLayerNorm(layer.ln_1), FrozenLayerNorm(layer.ln_2)
    if directml_safe_layernorm:
        model.encoder.ln = FrozenLayerNorm(model.encoder.ln)
    # Neuer Klassifikationskopf für unsere zehn Rassen
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    # Einstellungen für das Protokoll sammeln
    config = dict(model="ViT-B/16", weights=str(WEIGHTS), weights_url=WEIGHTS.url, rank=rank, alpha=alpha, scaling=alpha / rank, dropout=dropout,
                  targets=list(TARGETS), layers=len(model.encoder.layers), classes=CLASS_NAMES, directml_safe_layernorm=directml_safe_layernorm,
                  init="A: kaiming_uniform(a=sqrt(5)), B: zeros", trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  lora_parameters=sum(p.numel() for n, p in model.named_parameters() if "lora_" in n))
    return model, WEIGHTS.transforms(), config


# Nur die trainierbaren Gewichte holen
def trainable_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


# Nur LoRA und Kopf speichern, der Backbone ist bekannt
def save_lora_checkpoint(path, model, config, **extra):
    """Stores only LoRA and head tensors plus the configuration; the frozen backbone comes from the pretrained weights."""
    torch.save(dict(config=json.dumps(config, sort_keys=True), trainable=trainable_state(model), **extra), path)


# Gespeicherte LoRA-Gewichte laden
def load_lora_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = json.loads(checkpoint["config"])
    assert config["weights"] == str(WEIGHTS) and config["classes"] == CLASS_NAMES
    model, preprocess, _ = build_vit_lora(config["rank"], config["alpha"], config["dropout"], directml_safe_layernorm=config.get("directml_safe_layernorm", True))
    # Es darf kein Gewicht fehlen oder zu viel sein
    missing = set(trainable_state(model)) ^ set(checkpoint["trainable"])
    assert not missing, missing
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in checkpoint["trainable"]:
                parameter.copy_(checkpoint["trainable"][name])
    return model, preprocess, config, checkpoint


# AdamW ohne die Operation lerp, damit alles auf der GPU läuft
class AdamWNoLerp(torch.optim.Optimizer):
    """AdamW with the same update as torch.optim.AdamW, written without lerp, which DirectML runs on the CPU."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                # Schritte zählen, für die Korrektur am Anfang
                state["step"] += 1
                step = state["step"]
                # Weight Decay anwenden
                p.mul_(1 - group["lr"] * group["weight_decay"])
                # Laufende Mittelwerte der Gradienten aktualisieren
                state["exp_avg"].mul_(beta1).add_(p.grad, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                denom = (state["exp_avg_sq"].sqrt() / math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                # Gewichte aktualisieren
                p.addcdiv_(state["exp_avg"], denom, value=-group["lr"] / (1 - beta1 ** step))
