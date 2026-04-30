"""
SectionRouter — predicts per-1s section labels and routes onset detection.

Wraps the trained SectionClassifier. For each 1-s hop in audio, predicts a
label in {silence, constant_strum, chord_stab, lead_line, single_notes, mixed}
plus per-class confidences.

Used by guitar_bass.py to switch onset-detection strategies per section.

Set STRUM_GB_USE_ROUTER=0 to disable; the classifier checkpoint is optional
and the router degrades gracefully to "mixed" everywhere if not present.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

LABELS = ["silence", "constant_strum", "chord_stab", "lead_line", "single_notes", "mixed"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

# Same audio constants as scripts/preprocess_section_windows.py
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 30.0
FMAX = 8000.0
WINDOW_S = 2.0
HOP_S = 1.0
WINDOW_SAMPLES = int(WINDOW_S * SAMPLE_RATE)
WINDOW_FRAMES = WINDOW_SAMPLES // HOP_LENGTH + 1   # ~87
HOP_SAMPLES = int(HOP_S * SAMPLE_RATE)

DEFAULT_CKPT = "checkpoints/section_classifier/best.pt"


@dataclass
class Section:
    t_start_s: float
    t_end_s: float
    label: str
    probs: np.ndarray  # shape (6,)


_ROUTER_CACHE: dict[str, "SectionRouter"] = {}


def get_router(checkpoint: str = DEFAULT_CKPT) -> Optional["SectionRouter"]:
    """Singleton accessor; returns None if disabled or checkpoint missing."""
    if os.environ.get("STRUM_GB_USE_ROUTER", "1") == "0":
        return None
    key = checkpoint
    if key in _ROUTER_CACHE:
        return _ROUTER_CACHE[key]
    if not Path(checkpoint).exists():
        logger.warning("section classifier checkpoint not found: %s (router disabled)", checkpoint)
        _ROUTER_CACHE[key] = None  # type: ignore[assignment]
        return None
    try:
        router = SectionRouter(checkpoint)
        _ROUTER_CACHE[key] = router
        return router
    except Exception as exc:
        logger.warning("failed to load section router: %s", exc)
        _ROUTER_CACHE[key] = None  # type: ignore[assignment]
        return None


class SectionRouter:
    """Loads SectionClassifier and predicts section labels over an audio array."""

    def __init__(self, checkpoint: str = DEFAULT_CKPT, device: Optional[str] = None):
        from src.models.section_classifier import SectionClassifier

        # Force CPU by default — model is tiny (162k params), and torchaudio's
        # GPU STFT JIT-compile fails on GB10 (NVRTC sm-arch error). CPU is
        # fast enough (~50ms/song). Override with STRUM_SECTION_DEVICE=cuda.
        env_dev = os.environ.get("STRUM_SECTION_DEVICE")
        if device is None:
            device = env_dev or "cpu"
        self.device = torch.device(device)
        self.model = SectionClassifier().to(self.device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

        logger.info(
            "SectionRouter loaded: %s (val_acc=%.3f) on %s",
            checkpoint, ckpt.get("val_acc", float("nan")), self.device,
        )

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sr: int) -> list[Section]:
        """Predict per-1s-hop section labels for an audio array."""
        # Resample if needed (use librosa to avoid torchaudio NVRTC issue)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)

        if len(audio) < WINDOW_SAMPLES:
            return [Section(0.0, len(audio) / SAMPLE_RATE, "mixed", np.zeros(len(LABELS)))]

        # Slice into overlapping windows (HOP_S stride)
        starts: list[int] = list(range(0, len(audio) - WINDOW_SAMPLES + 1, HOP_SAMPLES))
        if starts[-1] + WINDOW_SAMPLES < len(audio):
            starts.append(len(audio) - WINDOW_SAMPLES)
        n = len(starts)

        # Compute one big mel spectrogram via librosa, then slice patches.
        # Faster than running mel per patch (~5x for typical song lengths).
        import librosa
        mel_full = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX, power=2.0,
        )
        mel_full = np.log(mel_full + 1e-8).astype(np.float32)  # (n_mels, T_total)

        # Number of mel frames covered by one window
        frames_per_window = WINDOW_FRAMES
        frames_per_hop = HOP_SAMPLES // HOP_LENGTH

        patches = np.zeros((n, N_MELS, frames_per_window), dtype=np.float32)
        for i, s in enumerate(starts):
            f_start = s // HOP_LENGTH
            f_end = f_start + frames_per_window
            if f_end > mel_full.shape[1]:
                # Right-pad with edge replication
                slab = mel_full[:, f_start:]
                pad = frames_per_window - slab.shape[1]
                slab = np.pad(slab, ((0, 0), (0, pad)), mode="edge")
                patches[i] = slab
            else:
                patches[i] = mel_full[:, f_start:f_end]

        # Z-norm per patch (matches training)
        mu = patches.mean(axis=(1, 2), keepdims=True)
        sd = patches.std(axis=(1, 2), keepdims=True) + 1e-5
        patches = (patches - mu) / sd

        mel_t = torch.from_numpy(patches).unsqueeze(1).to(self.device)  # (N,1,M,T)
        logits = self.model(mel_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        sections = []
        for i, s in enumerate(starts):
            sections.append(Section(
                t_start_s=s / SAMPLE_RATE,
                t_end_s=(s + WINDOW_SAMPLES) / SAMPLE_RATE,
                label=LABELS[int(preds[i])],
                probs=probs[i],
            ))
        return sections


def label_at_time(sections: list[Section], t_s: float) -> str:
    """Return the label of the section containing time t_s (or nearest)."""
    if not sections:
        return "mixed"
    # Linear scan; sections are short enough this is fine.
    best_label = sections[0].label
    for sec in sections:
        if sec.t_start_s <= t_s < sec.t_end_s:
            return sec.label
        if sec.t_start_s <= t_s:
            best_label = sec.label
    return best_label
