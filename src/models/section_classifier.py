"""
SectionClassifier — small CNN for 2-s log-mel patches → 6-class softmax.

Input  : (B, 1, n_mels=128, T~87)
Output : (B, n_classes=6) logits

~50k params. Designed to be cheap so we can run it during inference for
every 1-s hop without slowing the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SectionConfig:
    n_mels: int = 128
    n_classes: int = 6
    dropout: float = 0.3
    base_channels: int = 24


class _ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # stride-2 in both dims
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SectionClassifier(nn.Module):
    """Tiny CNN for section-type prediction."""

    def __init__(self, cfg: SectionConfig | None = None):
        super().__init__()
        cfg = cfg or SectionConfig()
        self.cfg = cfg

        c = cfg.base_channels
        self.b1 = _ConvBlock(1, c)        # /2
        self.b2 = _ConvBlock(c, c * 2)    # /4
        self.b3 = _ConvBlock(c * 2, c * 4)  # /8

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(c * 4, cfg.n_classes)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.b1(mel)
        x = self.b2(x)
        x = self.b3(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
