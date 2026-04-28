"""Shared helper: BG-subtracted log-mel computation.

Used by:
- src/models/drums_v14_dataset.py (training)
- scripts/batch_infer_hybrid.py (inference)

Both must use identical computation so the model sees the same input
distribution at training and inference time.

Implementation note: rolling percentile is computed via non-overlapping
blocks of width window_frames//2 followed by linear interpolation between
block centers. This is ~100x faster than per-frame numpy.percentile while
producing essentially the same background floor (smooth, slowly varying).
"""

from __future__ import annotations

import os

import numpy as np
import torch


def bg_subtract_log_mel(
    power_mel: torch.Tensor,
    sample_rate: int,
    hop_length: int,
    *,
    model_tag: str = "all",
) -> torch.Tensor:
    """Compute log-mel from linear power-mel, optionally with BG subtraction.

    Reads env vars:
      STRUM_MEL_BG_SUBTRACT: which models to apply to:
        "0" = off, "1"/"all" = all, "v14" = V14 only, "p3"/"phase3" = P3 only
      STRUM_MEL_BG_ALPHA: subtraction strength 0..1 (default 0.3)
      STRUM_MEL_BG_PCTL: percentile for background floor (default 20)
      STRUM_MEL_BG_WINDOW_SEC: rolling window seconds (default 1.0)

    Args:
        power_mel: (B, F, T) or (F, T) linear power mel-spectrogram.
        sample_rate: audio sample rate (Hz).
        hop_length: mel hop length (samples).
        model_tag: "v14", "phase3", or "all" — selects which env-var modes apply.

    Returns:
        log_mel: same shape as input, with log applied (and optionally BG
        subtracted before log).
    """
    flag = os.environ.get("STRUM_MEL_BG_SUBTRACT", "0").lower()
    enabled = (
        flag in ("1", "all")
        or (flag == "v14" and model_tag == "v14")
        or (flag in ("p3", "phase3") and model_tag == "phase3")
    )
    if not enabled:
        return torch.log(power_mel + 1e-8)

    alpha = float(os.environ.get("STRUM_MEL_BG_ALPHA", "0.3"))
    pctl = float(os.environ.get("STRUM_MEL_BG_PCTL", "20"))
    window_sec = float(os.environ.get("STRUM_MEL_BG_WINDOW_SEC", "1.0"))
    window_frames = max(8, int(window_sec * sample_rate / hop_length))

    # Squeeze batch dim if present (we operate on (F, T) numpy)
    squeeze_batch = power_mel.dim() == 3
    P_t = power_mel.squeeze(0) if squeeze_batch else power_mel
    P = P_t.detach().cpu().numpy()  # (F, T)
    F, T = P.shape

    # Block-percentile + linear interp:
    # Compute percentile over non-overlapping blocks of width = window_frames//2,
    # then interpolate between block centers. ~100x faster than per-frame.
    block = max(4, window_frames // 2)
    n_blocks = max(1, (T + block - 1) // block)

    # Per-block percentile: shape (F, n_blocks)
    block_pctl = np.empty((F, n_blocks), dtype=P.dtype)
    block_centers = np.empty(n_blocks, dtype=np.float32)
    for bi in range(n_blocks):
        s = bi * block
        e = min(T, s + block)
        block_pctl[:, bi] = np.percentile(P[:, s:e], pctl, axis=1)
        block_centers[bi] = (s + e - 1) / 2.0

    # Interpolate per frequency bin to all T frames
    if n_blocks == 1:
        bg = np.broadcast_to(block_pctl, (F, T)).copy()
    else:
        all_t = np.arange(T, dtype=np.float32)
        # vectorized linear interp per row
        bg = np.empty((F, T), dtype=P.dtype)
        for f in range(F):
            bg[f] = np.interp(all_t, block_centers, block_pctl[f])

    P_sub = np.clip(P - alpha * bg, 1e-8, None)
    out_t = torch.from_numpy(P_sub).to(power_mel.device).to(power_mel.dtype)
    if squeeze_batch:
        out_t = out_t.unsqueeze(0)
    return torch.log(out_t + 1e-8)
