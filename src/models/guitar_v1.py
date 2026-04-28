"""Guitar V1 — neural models.

Two-stage architecture mirroring the drums V14 pipeline:

  Stage 1 — :class:`GuitarOnsetCRNN`
    Input  : (B, 1, n_mels=128, T)        log-mel of a 5-s audio segment
    Output : (B, T)                       per-frame onset logits

  Stage 2 — :class:`GuitarFretClassifier`
    Input  : (B, 1, n_mels=128, W=22)     per-onset mel patch
             (W = 100 ms before + 400 ms after, hop 512 @ 22050 Hz)
    Output : (B, 5)                       per-fret BCE logits — multi-label
             so single notes and chords use the same head (chord = ≥2 bits)

Both modules are pure PyTorch ``nn.Module``s; training and inference glue
lives in ``scripts/train_guitar_v1.py`` and
``src/inference/guitar_neural.py`` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── Shared building blocks ─────────────────────────
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(-2, -1))
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ResBlock(nn.Module):
    """Residual conv block with optional spatial downsampling and SE attention."""

    def __init__(self, in_ch: int, out_ch: int,
                 stride: tuple[int, int] = (1, 1),
                 use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        if stride != (1, 1) or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


# ─────────────────────────── Stage 1 — Onset CRNN ───────────────────────────
@dataclass
class OnsetCRNNConfig:
    n_mels: int = 128
    in_channels: int = 1
    cnn_channels: tuple[int, ...] = (32, 64, 128, 256)
    rnn_hidden: int = 256
    rnn_layers: int = 2
    rnn_dropout: float = 0.2
    head_hidden: int = 128
    head_dropout: float = 0.2


class GuitarOnsetCRNN(nn.Module):
    """Frame-level binary onset detector for guitar.

    Strides only along the frequency axis so output time resolution equals
    input time resolution.
    """

    def __init__(self, cfg: OnsetCRNNConfig | None = None):
        super().__init__()
        cfg = cfg or OnsetCRNNConfig()
        self.cfg = cfg

        chans = (cfg.in_channels,) + tuple(cfg.cnn_channels)
        blocks: list[nn.Module] = []
        for i in range(len(chans) - 1):
            # Downsample frequency only (stride=(2,1)). Keep time intact.
            blocks.append(ResBlock(chans[i], chans[i + 1],
                                   stride=(2, 1), use_se=True))
        self.cnn = nn.Sequential(*blocks)

        n_freq_after = cfg.n_mels
        for _ in range(len(cfg.cnn_channels)):
            n_freq_after //= 2
        self._n_freq_after = n_freq_after          # 8 for n_mels=128, 4 blocks
        rnn_in = cfg.cnn_channels[-1] * n_freq_after

        self.rnn = nn.GRU(
            input_size=rnn_in,
            hidden_size=cfg.rnn_hidden,
            num_layers=cfg.rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.rnn_dropout if cfg.rnn_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.rnn_hidden * 2, cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Args:
            mel: (B, 1, n_mels, T) log-mel spectrogram

        Returns:
            (B, T) per-frame onset logits.
        """
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        x = self.cnn(mel)                          # (B, C, F', T)
        B, C, Fp, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()     # (B, T, C, F')
        x = x.view(B, T, C * Fp)                   # (B, T, C*F')
        x, _ = self.rnn(x)                         # (B, T, 2*H)
        logits = self.head(x).squeeze(-1)          # (B, T)
        return logits


# ─────────────────────────── Stage 2 — Fret Classifier ──────────────────────
@dataclass
class FretClassifierConfig:
    n_mels: int = 128
    n_frames: int = 22
    in_channels: int = 1
    cnn_channels: tuple[int, ...] = (32, 64, 128, 256)
    head_hidden: int = 256
    dropout: float = 0.3
    n_frets: int = 5
    aux_chord_head: bool = True


class GuitarFretClassifier(nn.Module):
    """Per-onset multi-label fret classifier.

    Output is 5 independent logits — bit *i* = "fret *i* is in this onset".
    A single note → one bit on; a chord → two or more bits on. An auxiliary
    chord/single binary head provides extra training signal.
    """

    def __init__(self, cfg: FretClassifierConfig | None = None):
        super().__init__()
        cfg = cfg or FretClassifierConfig()
        self.cfg = cfg

        chans = (cfg.in_channels,) + tuple(cfg.cnn_channels)
        blocks: list[nn.Module] = []
        for i in range(len(chans) - 1):
            stride = (2, 1) if i < len(cfg.cnn_channels) - 1 else (1, 1)
            blocks.append(ResBlock(chans[i], chans[i + 1],
                                   stride=stride, use_se=True))
        self.cnn = nn.Sequential(*blocks)

        feat_dim = cfg.cnn_channels[-1]
        self.fret_head = nn.Sequential(
            nn.Linear(feat_dim, cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, cfg.n_frets),
        )
        self.chord_head = (
            nn.Sequential(
                nn.Linear(feat_dim, cfg.head_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.head_hidden // 2, 1),
            )
            if cfg.aux_chord_head else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
            mel: (B, 1, n_mels, n_frames) log-mel patch around onset

        Returns:
            dict with keys ``fret_logits`` (B, 5) and optional
            ``chord_logits`` (B, 1).
        """
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        x = self.cnn(mel)                          # (B, C, F', T')
        feat = x.mean(dim=(-2, -1))                # (B, C) global pool
        out = {"fret_logits": self.fret_head(feat)}
        if self.chord_head is not None:
            out["chord_logits"] = self.chord_head(feat)
        return out


# ─────────────────────────── Convenience constructors ───────────────────────
def build_onset_crnn_from_dict(cfg: dict) -> GuitarOnsetCRNN:
    return GuitarOnsetCRNN(OnsetCRNNConfig(**cfg))


def build_fret_classifier_from_dict(cfg: dict) -> GuitarFretClassifier:
    return GuitarFretClassifier(FretClassifierConfig(**cfg))


__all__ = [
    "GuitarOnsetCRNN",
    "GuitarFretClassifier",
    "OnsetCRNNConfig",
    "FretClassifierConfig",
    "build_onset_crnn_from_dict",
    "build_fret_classifier_from_dict",
]
