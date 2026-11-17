"""Fusion head trained on cached, frozen encoder embeddings.

  text  [B, 7, 768]  -> LayerMix -> LayerNorm -> 768 ┐
                                                    ├ concat 1536 -> MLP -> 7 logits
  audio [B, 13, 768] -> LayerMix -> LayerNorm -> 768 ┘

LayerMix is a learned softmax-weighted average over encoder layers (one scalar
per layer), so the head decides which layers carry emotion information.
"""

import torch
from torch import nn

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class LayerMix(nn.Module):
    def __init__(self, n_layers: int, dim: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # x: [B, L, H]
        w = torch.softmax(self.logits, 0)
        return self.norm((w[None, :, None] * x).sum(1))


class FusionHead(nn.Module):
    def __init__(self, mode: str = "text+audio", text_layers: int = 7, audio_layers: int = 13,
                 dim: int = 768, hidden: int = 256, dropout: float = 0.3, n_classes: int = 7):
        super().__init__()
        assert mode in ("text+audio", "text", "audio")
        self.mode = mode
        self.text_mix = LayerMix(text_layers, dim) if "text" in mode else None
        self.audio_mix = LayerMix(audio_layers, dim) if "audio" in mode else None
        in_dim = dim * (("text" in mode) + ("audio" in mode))
        self.mlp = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, text=None, audio=None):
        parts = []
        if self.text_mix is not None:
            parts.append(self.text_mix(text))
        if self.audio_mix is not None:
            parts.append(self.audio_mix(audio))
        return self.mlp(torch.cat(parts, -1))
