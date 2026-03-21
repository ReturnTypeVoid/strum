#!/usr/bin/env python3
"""
Hybrid inference pipeline: V14 onset detection + onset classifier ensemble.

Stage 1: V14's TwoStageDrumsCRNN onset detector (93.9% F1 for WHERE)
Stage 2: Onset classifier ensemble (85.2% F1 for WHAT)

Pipeline:
  1. Load audio → Demucs separation → drums stem
  2. Run V14 onset detector → frame-level onset probabilities
  3. Peak detection → onset times
  4. Extract dual-resolution mel windows at each onset
  5. Run onset classifier ensemble → per-class probabilities
  6. Apply per-class thresholds → MIDI export

Usage:
    python scripts/batch_infer_hybrid.py
    python scripts/batch_infer_hybrid.py --skip-separation
    python scripts/batch_infer_hybrid.py --onset-threshold 0.35
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from omegaconf import OmegaConf
from scipy.signal import find_peaks
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.drums_v13 import TwoStageDrumsCRNN
from src.models.onset_classifier import OnsetClassifier
from src.preprocessing.parsers.midi_parser import DrumHit, DrumChart, TempoEvent, TimeSignature
from scripts.chart_postprocess import postprocess_chart
from src.export.midi import export_all_difficulties

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output/hybrid")

# ── V14 onset detector ──
V14_CHECKPOINT = "checkpoints/drums_v14/best.pt"
V14_SR = 44100
V14_N_FFT = 2048
V14_HOP = 512
V14_N_MELS = 128

# ── Onset classifier ensemble ──
ENSEMBLE_MODELS = [
    {"name": "V2", "config": "configs/onset_classifier.yaml",
     "checkpoint": "checkpoints/onset_classifier/best_f1.pt"},
    {"name": "V4", "config": "configs/onset_classifier_v4.yaml",
     "checkpoint": "checkpoints/onset_classifier_v4/best_f1.pt"},
    {"name": "V6", "config": "configs/onset_classifier_v6.yaml",
     "checkpoint": "checkpoints/onset_classifier_v6/best_f1.pt"},
    {"name": "V12c", "config": "configs/onset_classifier_v12_clean.yaml",
     "checkpoint": "checkpoints/onset_classifier_v12_clean/best_f1.pt"},
    {"name": "V15", "config": "configs/onset_classifier_v15.yaml",
     "checkpoint": "checkpoints/onset_classifier_v15/best_f1.pt"},
    {"name": "V16", "config": "configs/onset_classifier_v16.yaml",
     "checkpoint": "checkpoints/onset_classifier_v16/best_f1.pt"},
]

# ── Onset window extraction params (must match training preprocessing) ──
OC_SR = 44100
FINE_N_FFT = 1024
FINE_HOP = 256
COARSE_N_FFT = 4096
COARSE_HOP = 512
N_MELS = 128
WINDOW_BEFORE_MS = 100.0
WINDOW_AFTER_MS = 400.0

WINDOW_BEFORE_SAMP = int(WINDOW_BEFORE_MS / 1000 * OC_SR)
WINDOW_AFTER_SAMP = int(WINDOW_AFTER_MS / 1000 * OC_SR)
WINDOW_SAMPLES = WINDOW_BEFORE_SAMP + WINDOW_AFTER_SAMP
FINE_FRAMES = WINDOW_SAMPLES // FINE_HOP + 1    # 87
COARSE_FRAMES = WINDOW_SAMPLES // COARSE_HOP + 1  # 44

# CQT params for V12c
CQT_FMIN = 30.0
CQT_BINS_PER_OCT = 24
CQT_N_BINS = 144
CQT_TRIM_BINS = 128  # Trimmed to match n_mels
CQT_HOP = 512
CQT_FRAMES = WINDOW_SAMPLES // CQT_HOP + 1  # 44

# ── Class mapping ──
CLASS_NAMES = ['Kick', 'Snare', 'HiHat', 'HighTom', 'Ride', 'LowTom', 'Crash', 'FloorTom']
CLASS_TO_LANE = {
    0: (0, False),  # Kick → lane 0
    1: (1, False),  # Snare → lane 1
    2: (2, True),   # HiHat → lane 2, cymbal
    3: (2, False),  # HighTom → lane 2, tom
    4: (3, True),   # Ride → lane 3, cymbal
    5: (3, False),  # LowTom → lane 3, tom
    6: (4, True),   # Crash → lane 4, cymbal
    7: (4, False),  # FloorTom → lane 4, tom
}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac"}


# ══════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════

def load_v14_onset_detector(device: torch.device) -> TwoStageDrumsCRNN:
    """Load V14 TwoStageDrumsCRNN for onset detection only."""
    logger.info(f"Loading V14 onset detector from {V14_CHECKPOINT}")
    ckpt = torch.load(V14_CHECKPOINT, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}).get("model", {})

    model = TwoStageDrumsCRNN(
        n_mels=cfg.get("n_mels", 128),
        conv_channels=cfg.get("conv_channels", [64, 128, 256, 512]),
        freq_subbands=cfg.get("freq_subbands", [32, 64, 96, 128]),
        subband_proj_dim=cfg.get("subband_proj_dim", 256),
        lstm_hidden=cfg.get("lstm_hidden", 640),
        lstm_layers=cfg.get("lstm_layers", 3),
        attention_heads=cfg.get("attention_heads", 10),
        attention_type=cfg.get("attention_type", "flash"),
        attention_window=cfg.get("attention_window", 512),
        dropout=0.0,
        onset_detector_hidden=cfg.get("onset_detector_hidden", 320),
        classifier_hidden=cfg.get("classifier_hidden", 640),
        num_classes=cfg.get("num_classes", 8),
        predict_velocity=cfg.get("predict_velocity", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    best_f1 = ckpt.get("best_f1", "?")
    logger.info(f"  V14 loaded (epoch {epoch}, best_f1={best_f1})")
    return model


def build_onset_classifier(config) -> OnsetClassifier:
    """Build onset classifier from config."""
    m = config.model
    return OnsetClassifier(
        num_classes=m.num_classes,
        branch_channels=list(m.branch_channels),
        context_size=m.context_size,
        context_hidden=m.context_hidden,
        classifier_hidden=m.classifier_hidden,
        spectral_dim=m.get("spectral_dim", 32),
        dropout=0.0,  # no dropout at inference
        use_freq_attn=m.get("use_freq_attn", False),
        use_hpss=m.get("use_hpss", False),
        enhanced_spectral=m.get("enhanced_spectral", False),
        use_contrastive=False,
        use_aux_head=False,
        use_dual_head=m.get("use_dual_head", False),
        tom_head_hidden=m.get("tom_head_hidden", 256),
        use_lowfreq_branch=m.get("use_lowfreq_branch", False),
        use_lowfreq_spectral=m.get("use_lowfreq_spectral", False),
    )


def load_ensemble(device: torch.device) -> list[dict]:
    """Load all onset classifier ensemble models."""
    models = []
    for eidx, entry in enumerate(ENSEMBLE_MODELS):
        name = entry["name"]
        cfg_path = entry["config"]
        ckpt_path = entry["checkpoint"]

        if not Path(ckpt_path).exists() or not Path(cfg_path).exists():
            logger.warning(f"  {name}: missing checkpoint or config, skipping")
            continue

        config = OmegaConf.load(cfg_path)
        model = build_onset_classifier(config).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        needs_cqt = config.model.get("use_lowfreq_branch", False) or config.model.get("use_lowfreq_spectral", False)
        epoch = ckpt.get("epoch", -1) + 1
        logger.info(f"  {name}: epoch {epoch}, needs_cqt={needs_cqt}")

        models.append({
            "name": name,
            "model": model,
            "config": config,
            "needs_cqt": needs_cqt,
            "ensemble_idx": eidx,  # original index in ENSEMBLE_MODELS for weight lookup
        })

    logger.info(f"  Loaded {len(models)} ensemble models")
    return models


# ══════════════════════════════════════════════════════════════
# Audio processing
# ══════════════════════════════════════════════════════════════

def separate_drums(audio_path: Path, output_dir: Path) -> Path:
    """Separate drums stem using Demucs Python API."""
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    demucs_out = output_dir / "demucs_temp"
    demucs_out.mkdir(parents=True, exist_ok=True)

    logger.info("  Separating drums with Demucs (htdemucs)...")

    y, sr = librosa.load(str(audio_path), sr=44100, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])  # mono → stereo

    model_demucs = get_model("htdemucs")
    model_demucs.eval()

    wav_tensor = torch.from_numpy(y).float().unsqueeze(0)
    with torch.no_grad():
        sources = apply_model(model_demucs, wav_tensor, progress=True)

    source_names = model_demucs.sources
    drums_idx = source_names.index("drums")
    drums_audio = sources[0, drums_idx].cpu().numpy()

    drums_path = demucs_out / "drums.wav"
    sf.write(str(drums_path), drums_audio.T, 44100)
    logger.info(f"  Drums stem: {drums_path}")
    return drums_path


def analyze_audio(audio_path: Path) -> dict:
    """Analyze audio for tempo, tempo changes, and duration.

    Returns a dict with:
      - tempo_bpm: global BPM (refined via grid-alignment search)
      - tempo_events: list of TempoEvent for tempo changes
      - duration_ms / duration_sec: audio length
    """
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration_sec = len(y) / sr

    # --- Beat tracking ---
    # Note: do NOT pass onset_envelope to beat_track – its internal onset
    # strength computation uses different defaults and produces more stable
    # beat positions than a separately-computed envelope.
    tempo_est, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Initial BPM from median IBI (robust to outlier beats).
    # Median IBI is more reliable than librosa's global estimate because
    # it is not fooled by octave ambiguity or short-term noise.
    if len(beat_times) >= 4:
        ibis = np.diff(beat_times)
        base_bpm = 60.0 / np.median(ibis)
    else:
        base_bpm = float(tempo_est[0]) if hasattr(tempo_est, "__len__") else float(tempo_est)

    # Normalize to 60-200 range
    while base_bpm < 60:
        base_bpm *= 2
    while base_bpm > 200:
        base_bpm /= 2

    # --- Grid-alignment BPM refinement ---
    # Search ±5 BPM around the initial estimate at 0.1 resolution.
    # For each candidate BPM, compute the optimal beat offset via
    # circular statistics and measure how well detected beats align
    # to the grid.  This is much more accurate than median IBI alone
    # because it jointly optimizes BPM and phase.
    bpm = round(base_bpm * 2) / 2  # fallback: nearest 0.5
    if len(beat_times) >= 8:
        best_error = float("inf")
        lo = max(400, int(base_bpm * 10) - 50)
        hi = int(base_bpm * 10) + 51
        for bpm_10x in range(lo, hi):
            cand = bpm_10x / 10.0
            per = 60.0 / cand
            phases = (beat_times % per) / per * 2 * np.pi
            mean_phase = np.arctan2(np.sin(phases).mean(), np.cos(phases).mean())
            if mean_phase < 0:
                mean_phase += 2 * np.pi
            offset = mean_phase / (2 * np.pi) * per
            residuals = (beat_times - offset) % per
            errors = np.minimum(residuals, per - residuals)
            me = float(np.mean(errors))
            if me < best_error:
                best_error = me
                bpm = cand

    # --- Tempo change detection ---
    # Use windowed median of IBIs.  Only flag a change if the local BPM
    # deviates from the current segment BPM by >3 BPM for >=8 consecutive
    # beats.  Beat tracker noise has std ≈ 1.7 BPM, so 3 BPM avoids
    # false positives while catching real tempo shifts.
    tempo_events: list[TempoEvent] = []
    if len(beat_times) >= 12:
        ibis = np.diff(beat_times)
        window = 4
        local_bpms = []
        local_times = []
        for i in range(len(ibis) - window + 1):
            local_bpms.append(60.0 / np.median(ibis[i : i + window]))
            local_times.append(beat_times[i + window // 2])

        # Walk through local BPMs and detect change points
        current_bpm = bpm  # start with global refined BPM
        tempo_events.append(TempoEvent(tick=0, tempo_bpm=round(current_bpm, 1), time_ms=0.0))
        CHANGE_THRESH = 3.0  # BPM
        MIN_PERSIST = 8  # consecutive windows

        i = 0
        while i < len(local_bpms):
            if abs(local_bpms[i] - current_bpm) > CHANGE_THRESH:
                # Check persistence
                new_bpm_values = []
                j = i
                while j < len(local_bpms) and abs(local_bpms[j] - current_bpm) > CHANGE_THRESH:
                    new_bpm_values.append(local_bpms[j])
                    j += 1
                if len(new_bpm_values) >= MIN_PERSIST:
                    new_bpm = round(float(np.median(new_bpm_values)), 1)
                    # Normalize to 60-200
                    while new_bpm < 60:
                        new_bpm *= 2
                    while new_bpm > 200:
                        new_bpm /= 2
                    change_time_ms = round(local_times[i] * 1000, 1)
                    tempo_events.append(TempoEvent(tick=0, tempo_bpm=new_bpm, time_ms=change_time_ms))
                    current_bpm = new_bpm
                    i = j
                    continue
            i += 1
    else:
        tempo_events.append(TempoEvent(tick=0, tempo_bpm=round(bpm, 1), time_ms=0.0))

    logger.info(f"  Tempo: {bpm:.1f} BPM, {len(tempo_events)} tempo event(s)")

    return {
        "tempo_bpm": round(bpm, 1),
        "tempo_events": tempo_events,
        "duration_ms": duration_sec * 1000,
        "duration_sec": duration_sec,
    }


# ══════════════════════════════════════════════════════════════
# Stage 1: V14 onset detection
# ══════════════════════════════════════════════════════════════

def detect_onsets_v14(
    model: TwoStageDrumsCRNN,
    audio_path: Path,
    device: torch.device,
    onset_threshold: float = 0.4,
    segment_duration: float = 10.0,
    overlap: float = 0.5,
    min_distance_ms: float = 20.0,
) -> list[float]:
    """Run V14 onset detector and return onset times in milliseconds.

    Returns:
        Sorted list of onset times (ms).
    """
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=V14_SR, mono=True)
    audio = torch.from_numpy(y).unsqueeze(0)

    # Compute mel spectrogram (same as V14 training)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=V14_SR, n_fft=V14_N_FFT, hop_length=V14_HOP, n_mels=V14_N_MELS,
    )
    mel_spec = mel_transform(audio)
    mel_spec = torch.log(mel_spec + 1e-8)

    total_frames = mel_spec.shape[-1]
    segment_frames = int(segment_duration * V14_SR / V14_HOP)
    hop_frames = int(segment_frames * (1 - overlap))

    # Overlap-averaged onset probabilities
    onset_probs = np.zeros(total_frames)
    count = np.zeros(total_frames)

    num_segments = max(1, (total_frames - segment_frames) // hop_frames + 1)

    with torch.no_grad():
        for i in tqdm(range(num_segments), desc="  V14 onset detection", leave=False):
            start = i * hop_frames
            end = min(start + segment_frames, total_frames)
            segment = mel_spec[:, :, start:end]
            if segment.shape[-1] < segment_frames:
                segment = torch.nn.functional.pad(segment, (0, segment_frames - segment.shape[-1]))
            segment = segment.unsqueeze(0).to(device)
            outputs = model(segment)
            probs = outputs["onset_probs"].squeeze(0).squeeze(-1).cpu().numpy()
            actual_len = min(end - start, len(probs))
            onset_probs[start:start + actual_len] += probs[:actual_len]
            count[start:start + actual_len] += 1

    count = np.maximum(count, 1)
    onset_probs /= count

    # Peak detection
    min_distance_frames = max(1, int(min_distance_ms / 1000 * V14_SR / V14_HOP))
    peaks, properties = find_peaks(
        onset_probs,
        height=onset_threshold,
        distance=min_distance_frames,
    )

    # Convert frame indices to milliseconds
    onset_times_ms = [(frame * V14_HOP / V14_SR) * 1000 for frame in peaks]

    logger.info(f"  V14 detected {len(onset_times_ms)} onsets "
                f"(threshold={onset_threshold}, "
                f"prob range={onset_probs.min():.3f}–{onset_probs.max():.3f})")

    return onset_times_ms


# ══════════════════════════════════════════════════════════════
# Stage 2: Onset classification with ensemble
# ══════════════════════════════════════════════════════════════

def extract_onset_windows(
    audio_path: Path,
    onset_times_ms: list[float],
    needs_cqt: bool = True,
) -> dict:
    """Extract dual-resolution mel windows (+ optional CQT) at onset positions.

    Uses torchaudio mel transforms (matching training preprocessing exactly)
    and per-window extraction (not full-song slicing) for fidelity.

    Returns dict with:
        mel_fine: (N, 128, 87) float32
        mel_coarse: (N, 128, 44) float32
        cqt: (N, 128, 44) float32 or None
        valid_mask: (N,) bool — True if window fits within audio
    """
    y, sr = librosa.load(str(audio_path), sr=OC_SR, mono=True)
    audio_len = len(y)

    # torchaudio mel transforms — must match training preprocessing exactly
    fine_mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=OC_SR, n_fft=FINE_N_FFT, hop_length=FINE_HOP,
        n_mels=N_MELS, power=2.0,
    )
    coarse_mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=OC_SR, n_fft=COARSE_N_FFT, hop_length=COARSE_HOP,
        n_mels=N_MELS, power=2.0,
    )

    # Pre-compute full CQT if needed (librosa — same as training)
    cqt_full = None
    if needs_cqt:
        C = np.abs(librosa.cqt(
            y=y, sr=OC_SR, hop_length=CQT_HOP,
            fmin=CQT_FMIN, n_bins=CQT_N_BINS, bins_per_octave=CQT_BINS_PER_OCT,
        ))
        cqt_full = np.log(C + 1e-8).astype(np.float32)
        cqt_full = cqt_full[:CQT_TRIM_BINS, :]  # Trim 144 → 128

    N = len(onset_times_ms)
    mel_fine_windows = np.zeros((N, N_MELS, FINE_FRAMES), dtype=np.float32)
    mel_coarse_windows = np.zeros((N, N_MELS, COARSE_FRAMES), dtype=np.float32)
    cqt_windows = np.zeros((N, CQT_TRIM_BINS, CQT_FRAMES), dtype=np.float32) if needs_cqt else None
    valid_mask = np.ones(N, dtype=bool)

    for i, t_ms in enumerate(onset_times_ms):
        center_sample = int(t_ms / 1000 * OC_SR)
        start_sample = center_sample - WINDOW_BEFORE_SAMP
        end_sample = center_sample + WINDOW_AFTER_SAMP

        if start_sample < 0 or end_sample > audio_len:
            valid_mask[i] = False
            continue

        # Extract audio window and compute mel per-window (matches training)
        window = y[start_sample:end_sample]
        if len(window) < WINDOW_SAMPLES:
            window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
        audio_t = torch.from_numpy(window).float().unsqueeze(0)

        with torch.no_grad():
            mf = torch.log(fine_mel_transform(audio_t) + 1e-8).squeeze(0).numpy()
            mc = torch.log(coarse_mel_transform(audio_t) + 1e-8).squeeze(0).numpy()

        # Pad/trim to exact frame count (matches training preprocess)
        if mf.shape[-1] >= FINE_FRAMES:
            mel_fine_windows[i] = mf[:, :FINE_FRAMES]
        else:
            mel_fine_windows[i, :, :mf.shape[-1]] = mf

        if mc.shape[-1] >= COARSE_FRAMES:
            mel_coarse_windows[i] = mc[:, :COARSE_FRAMES]
        else:
            mel_coarse_windows[i, :, :mc.shape[-1]] = mc

        # CQT window (sliced from full-song CQT — same as training)
        if needs_cqt and cqt_full is not None:
            cqt_start = start_sample // CQT_HOP
            cqt_end = cqt_start + CQT_FRAMES
            if cqt_end <= cqt_full.shape[1]:
                cqt_windows[i] = cqt_full[:, cqt_start:cqt_end]
            else:
                avail = max(0, cqt_full.shape[1] - cqt_start)
                if avail > 0:
                    cqt_windows[i, :, :avail] = cqt_full[:, cqt_start:cqt_start + avail]

    return {
        "mel_fine": mel_fine_windows,
        "mel_coarse": mel_coarse_windows,
        "cqt": cqt_windows,
        "valid_mask": valid_mask,
    }


def build_context_vectors(onset_times_ms: list[float], class_probs: np.ndarray | None = None) -> np.ndarray:
    """Build onset context vectors: 4 prev + 4 next onsets' class probabilities.

    For the first pass (before classification), use zeros.
    For a second pass, use the class probabilities from the first.

    Returns: (N, 64) float32 — 8 classes × 8 neighbors
    """
    N = len(onset_times_ms)
    context = np.zeros((N, 64), dtype=np.float32)

    if class_probs is None:
        return context

    for i in range(N):
        vec = []
        # 4 previous onsets
        for j in range(i - 4, i):
            if 0 <= j < N:
                vec.extend(class_probs[j].tolist())
            else:
                vec.extend([0.0] * 8)
        # 4 next onsets
        for j in range(i + 1, i + 5):
            if 0 <= j < N:
                vec.extend(class_probs[j].tolist())
            else:
                vec.extend([0.0] * 8)
        context[i] = np.array(vec[:64], dtype=np.float32)

    return context


def classify_onsets_ensemble(
    ensemble: list[dict],
    windows: dict,
    context: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run onset classifier ensemble with per-class model weighting.

    Uses PER_CLASS_WEIGHTS to weight each model differently for each class,
    then returns (N, 8) sigmoid probabilities.
    """
    N = windows["mel_fine"].shape[0]
    all_logits = []

    for entry in ensemble:
        model = entry["model"]
        name = entry["name"]
        needs_cqt = entry["needs_cqt"]

        model_logits = np.zeros((N, 8), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)

                t_fine = torch.from_numpy(
                    windows["mel_fine"][start:end]
                ).unsqueeze(1).to(device)
                t_coarse = torch.from_numpy(
                    windows["mel_coarse"][start:end]
                ).unsqueeze(1).to(device)
                t_ctx = torch.from_numpy(context[start:end]).to(device)

                kwargs = {}
                if needs_cqt and windows["cqt"] is not None:
                    t_cqt = torch.from_numpy(
                        windows["cqt"][start:end]
                    ).unsqueeze(1).to(device)
                    kwargs["mel_lowfreq"] = t_cqt

                logits = model(t_fine, t_coarse, t_ctx, **kwargs)
                model_logits[start:end] = logits.cpu().numpy()

        all_logits.append(model_logits)
        logger.info(f"    {name}: inference done")

    # Per-class weighted average of logits
    # Each model gets a different weight for each class
    # Use ensemble_idx to map loaded models to their weight positions
    weighted_logits = np.zeros((N, 8), dtype=np.float32)
    for c in range(8):
        weights = PER_CLASS_WEIGHTS[c]
        total_w = 0.0
        for model_idx, entry in enumerate(ensemble):
            eidx = entry.get("ensemble_idx", model_idx)
            if eidx < len(weights):
                w = weights[eidx]
            else:
                w = 0.0
            if w > 0:
                weighted_logits[:, c] += w * all_logits[model_idx][:, c]
                total_w += w
        # Renormalize if some models were missing
        if total_w > 0 and total_w < 0.99:
            weighted_logits[:, c] /= total_w

    return weighted_logits  # return logits; sigmoid applied after spectral correction


# ══════════════════════════════════════════════════════════════
# Per-class thresholds & weights (optimized via eval_hybrid_ensemble.py
# on the test set using V2/V4/V6/V12c ensemble)
# ══════════════════════════════════════════════════════════════

# Per-class model weights (order matches ENSEMBLE_MODELS: V2, V4, V6, V12c, V15, V16)
# V12c dominates most classes; V6 good on toms; V15 bleed robustness
# V16: FloorTom specialist (76.4% F1, only 1.3% FloorTom→Crash confusion)
PER_CLASS_WEIGHTS = {
    0: [0.25, 0.0, 0.25, 0.30, 0.15, 0.05],   # Kick      - V16 minimal
    1: [0.25, 0.15, 0.10, 0.25, 0.20, 0.05],   # Snare     - V16 minimal
    2: [0.25, 0.05, 0.15, 0.30, 0.20, 0.05],   # HiHat     - V16 minimal
    3: [0.20, 0.0, 0.30, 0.20, 0.30, 0.00],    # HighTom   - V16 OFF (37% F1)
    4: [0.15, 0.05, 0.25, 0.30, 0.25, 0.00],   # Ride      - V16 OFF
    5: [0.15, 0.10, 0.10, 0.30, 0.30, 0.05],   # LowTom    - V16 small
    6: [0.20, 0.0, 0.22, 0.28, 0.25, 0.05],    # Crash     - V16 small
    7: [0.08, 0.02, 0.25, 0.15, 0.25, 0.25],   # FloorTom  - V16 specialist (low Crash confusion)
}

DEFAULT_CLASS_THRESHOLDS = [
    0.30,  # Kick
    0.40,  # Snare
    0.44,  # HiHat
    0.05,  # HighTom  (lowered — ensemble median 0.47 for true toms, 0.004 for cymbals)
    0.42,  # Ride
    0.05,  # LowTom   (lowered — ensemble median 0.73 for true toms, 0.003 for cymbals)
    0.44,  # Crash
    0.12,  # FloorTom (raised from 0.08 — V16 adds FloorTom-biased signal)
]

# Lane pairs for tom/cymbal conflict resolution:
# (cymbal_class_idx, tom_class_idx)
LANE_CONFLICT_PAIRS = [
    (2, 3),  # Yellow: HiHat vs HighTom
    (4, 5),  # Blue: Ride vs LowTom
    (6, 7),  # Green: Crash vs FloorTom
]


# ══════════════════════════════════════════════════════════════
# Spectral tom/cymbal disambiguation
# ══════════════════════════════════════════════════════════════

def compute_spectral_features(audio_path: Path, onset_times_ms: list[float]) -> dict:
    """Compute spectral features at each onset for tom/cymbal reclassification.

    Computes:
      - low_ratio: energy ratio of 200-1000 Hz vs 3000-10000 Hz
        High ratio (>3) = tonal/tom-like, Low ratio (<0.5) = broadband/cymbal-like

    Returns:
        dict with 'low_ratio': (N,) array
    """
    y, sr = librosa.load(str(audio_path), sr=OC_SR, mono=True)
    n_onsets = len(onset_times_ms)
    low_ratios = np.ones(n_onsets, dtype=np.float32)  # neutral default

    window_samples = int(0.040 * sr)  # 40ms analysis window
    offset_samples = int(0.005 * sr)  # 5ms after onset

    for i, t_ms in enumerate(onset_times_ms):
        center = int(t_ms / 1000 * sr)
        start = center + offset_samples
        end = start + window_samples

        if start < 0 or end > len(y):
            continue

        window = y[start:end]
        if np.max(np.abs(window)) < 1e-6:
            continue

        S = np.abs(np.fft.rfft(window * np.hanning(len(window))))
        freqs = np.fft.rfftfreq(len(window), 1.0 / sr)

        low_mask = (freqs >= 200) & (freqs <= 1000)
        high_mask = (freqs >= 3000) & (freqs <= 10000)
        low_energy = np.sum(S[low_mask] ** 2)
        high_energy = np.sum(S[high_mask] ** 2)
        low_ratios[i] = low_energy / (high_energy + 1e-10)

    return {"low_ratio": low_ratios}


def _compute_spectral_centroid_features(
    audio_path: Path,
    onset_times_ms: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spectral centroid and high-frequency energy percentage per onset.

    Used by the spectral safety net in build_chart to identify cymbal hits
    that are spectrally tom-like (low centroid, no high-freq shimmer).

    Returns:
        (centroids, high_pcts): arrays of shape (N,)
    """
    y, sr = librosa.load(str(audio_path), sr=OC_SR, mono=True)
    n = len(onset_times_ms)
    centroids = np.full(n, 5000.0, dtype=np.float32)  # default = cymbal-like
    high_pcts = np.full(n, 50.0, dtype=np.float32)

    window_samples = int(0.040 * sr)
    offset_samples = int(0.005 * sr)

    for i, t_ms in enumerate(onset_times_ms):
        center = int(t_ms / 1000 * sr)
        start = center + offset_samples
        end = start + window_samples
        if start < 0 or end > len(y):
            continue
        chunk = y[start:end]
        if np.max(np.abs(chunk)) < 1e-6:
            continue

        S = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

        total_s = np.sum(S)
        if total_s > 0:
            centroids[i] = np.sum(freqs * S) / total_s

        total_e = np.sum(S ** 2)
        if total_e > 0:
            high_mask = freqs > 3000
            high_pcts[i] = np.sum(S[high_mask] ** 2) / total_e * 100

    return centroids, high_pcts


def apply_spectral_reclassification(
    probs: np.ndarray,
    spectral: dict,
    onset_times_ms: list[float],
    tom_ratio_threshold: float = 5.0,
    transfer_pct: float = 0.4,
    kick_gate: float = 0.25,
    cymbal_cap: float = 0.80,
) -> np.ndarray:
    """Reclassify cymbal hits as toms using fill detection + spectral evidence.

    Two-stage approach:
    1. Detect fill candidate regions: rapid bursts of 3+ hits where hihat/ride
       drops out and spectral ratio is elevated. Aggressively reclassify within
       confirmed fill regions.
    2. For remaining onsets, apply conservative spectral reclassification only
       when spectral evidence is very strong (ratio > 2*threshold) and kick
       probability is low.
    """
    corrected = probs.copy()
    pairs = [(2, 3), (4, 5), (6, 7)]  # (cymbal_idx, tom_idx) = (HiHat/HighTom, Ride/LowTom, Crash/FloorTom)
    low_ratio = spectral["low_ratio"]
    times = np.array(onset_times_ms)
    n = len(times)

    # ── Stage 1: Fill detection ──
    # Find rapid-fire clusters (gaps < 150ms, 3+ onsets)
    fill_mask = np.zeros(n, dtype=bool)
    n_fill_reclassified = 0
    n_fills_found = 0

    if n >= 3:
        gaps = np.diff(times)
        # Build clusters of rapid hits
        cluster_start = 0
        clusters = []
        for i in range(1, n):
            if gaps[i - 1] > 150:  # break
                if i - cluster_start >= 3:
                    clusters.append((cluster_start, i))
                cluster_start = i
        if n - cluster_start >= 3:
            clusters.append((cluster_start, n))

        for start, end in clusters:
            indices = list(range(start, end))

            # Check 1: hihat/ride must be LOW (absolute) — during fills,
            # drummer lifts hand from hihat. Use absolute threshold, not relative.
            cluster_hihat = np.mean(corrected[indices, 2])  # HiHat
            cluster_ride = np.mean(corrected[indices, 4])   # Ride
            hihat_absent = cluster_hihat < 0.30 and cluster_ride < 0.30

            # Check 2: spectral evidence — at least 30% of cluster is tonal
            cluster_ratios = low_ratio[indices]
            pct_tonal = np.mean(cluster_ratios > tom_ratio_threshold * 0.6)

            # Check 3: is a crash within 400ms after cluster end?
            crash_follows = False
            end_time = times[end - 1]
            crash_idx = np.where((times > end_time) & (times < end_time + 400))[0]
            if len(crash_idx) > 0:
                crash_follows = np.any(corrected[crash_idx, 6] > 0.3)  # Crash col

            # Score the fill candidate — hihat absence is mandatory
            if not hihat_absent:
                continue  # Not a fill if hihat/ride is still playing

            fill_score = 0.4  # hihat absence confirmed
            if pct_tonal > 0.3:
                fill_score += 0.3
            if crash_follows:
                fill_score += 0.3

            if fill_score >= 0.7:
                n_fills_found += 1
                # Reclassify: for non-kick/snare onsets in the fill, transfer cymbal→tom
                # Only reclassify onsets that individually have spectral tom evidence
                fill_transfer = min(0.6, transfer_pct + fill_score * 0.3)
                for i in indices:
                    fill_mask[i] = True
                    # Skip kick-dominated onsets even in fills
                    if corrected[i, 0] > 0.5:
                        continue
                    # Require per-onset spectral evidence
                    if low_ratio[i] < tom_ratio_threshold * 0.5:
                        continue
                    for cym_idx, tom_idx in pairs:
                        if corrected[i, cym_idx] < 0.10:
                            continue
                        boost = corrected[i, cym_idx] * fill_transfer
                        corrected[i, tom_idx] = min(1.0, corrected[i, tom_idx] + boost)
                        corrected[i, cym_idx] *= (1.0 - fill_transfer)
                        n_fill_reclassified += 1

    # ── Stage 2: Disabled ──
    # Isolated spectral reclassification creates too many scattered false-positive
    # toms. Only fills-based reclassification (Stage 1) is used.
    n_spectral_reclassified = 0

    tom_like = int(np.sum(low_ratio > tom_ratio_threshold))
    logger.info(f"  Fill detection: {n_fills_found} fills found, {n_fill_reclassified} transfers")
    logger.info(f"  (Stage 2 disabled, {tom_like} tom-like onsets not reclassified)")

    return corrected


# ══════════════════════════════════════════════════════════════
# Chart building
# ══════════════════════════════════════════════════════════════

def build_chart(
    onset_times_ms: list[float],
    class_probs: np.ndarray,
    valid_mask: np.ndarray,
    tempo_events: list[TempoEvent],
    thresholds: list[float] | None = None,
    spectral_centroids: np.ndarray | None = None,
    spectral_high_pcts: np.ndarray | None = None,
) -> DrumChart:
    """Convert classification results into a DrumChart.

    For each onset, emit hits for classes above threshold.
    Resolves tom/cymbal lane conflicts by picking higher probability.
    Caps hand notes at 2 per onset (drummer has 2 hands).

    Spectral safety net: when spectral features are provided, cymbal hits
    on shared lanes are overridden to toms when the audio is spectrally
    tom-like (centroid < 2000Hz, <1% energy above 3kHz) and kick is low.
    """
    if thresholds is None:
        thresholds = DEFAULT_CLASS_THRESHOLDS

    hits = []
    class_counts = {c: 0 for c in range(8)}
    lane_conflicts_resolved = 0
    hand_caps_applied = 0
    spectral_overrides = 0
    kick_suppressed_ftom = 0
    spectral_conflict_bias = 0

    # Hand classes = everything except Kick (idx 0). Kick is played with foot.
    HAND_CLASSES = {1, 2, 3, 4, 5, 6, 7}
    has_spectral = (spectral_centroids is not None and spectral_high_pcts is not None)

    for i, t_ms in enumerate(onset_times_ms):
        if not valid_mask[i]:
            continue

        # Collect which classes fire (above threshold)
        fired = set()
        for cls in range(8):
            if class_probs[i, cls] > thresholds[cls]:
                fired.add(cls)

        # Resolve lane conflicts on shared tom/cymbal lanes.
        # When both fire, use spectral evidence to bias the decision.
        # Background percussion (shaker, tambourine) bleeds cymbal
        # energy into onsets that are primarily toms.  When the spectral
        # centroid is low-to-moderate, the dominant sound is tom-like
        # and the cymbal probability is inflated by background bleed.
        for cym_idx, tom_idx in LANE_CONFLICT_PAIRS:
            if cym_idx in fired and tom_idx in fired:
                lane_conflicts_resolved += 1
                cym_p = class_probs[i, cym_idx]
                tom_p = class_probs[i, tom_idx]
                # Spectral bias: if centroid is low-moderate, boost tom's
                # effective probability.  Background percussion raises
                # cymbal probs but doesn't change the spectral centroid
                # as much as a real cymbal hit would.
                if has_spectral:
                    centroid = spectral_centroids[i]
                    if centroid < 2500:
                        # Strong tom bias — centroid clearly tom-like
                        tom_p *= 2.0
                        spectral_conflict_bias += 1
                    elif centroid < 4000:
                        # Moderate bias — ambiguous zone
                        tom_p *= 1.3
                        spectral_conflict_bias += 1
                if cym_p >= tom_p:
                    fired.discard(tom_idx)
                else:
                    fired.discard(cym_idx)
            elif tom_idx in fired and cym_idx not in fired:
                # Tom fires alone — but does the cymbal have higher prob?
                # Only swap to cymbal if centroid supports it (high freq)
                cym_p = class_probs[i, cym_idx]
                tom_p = class_probs[i, tom_idx]
                swap = False
                if cym_p > tom_p:
                    swap = True
                    # Block the swap when spectrum is tom-like
                    if has_spectral and spectral_centroids[i] < 3000:
                        swap = False
                if swap:
                    fired.discard(tom_idx)
                    fired.add(cym_idx)
                    lane_conflicts_resolved += 1

        # Kick-suppresses-FloorTom: when Kick fires confidently and
        # FloorTom barely cleared its low threshold, suppress FloorTom.
        # Kick and FloorTom share low-frequency energy; real simultaneous
        # kick+floor tom hits are rare.
        KICK_IDX, FTOM_IDX = 0, 7
        if KICK_IDX in fired and FTOM_IDX in fired:
            kick_prob = class_probs[i, KICK_IDX]
            ftom_prob = class_probs[i, FTOM_IDX]
            if kick_prob > 0.5 and ftom_prob < 0.35:
                fired.discard(FTOM_IDX)
                kick_suppressed_ftom += 1

        # Spectral safety net: override cymbal → tom when audio is
        # clearly tom-like (low centroid, no high freq shimmer) and
        # kick isn't strongly dominating the spectrum.
        # Threshold at 2500 Hz — accounts for background percussion
        # (shaker/tambourine) pushing centroid slightly higher.
        if has_spectral and class_probs[i, 0] < 0.7:
            centroid = spectral_centroids[i]
            high_pct = spectral_high_pcts[i]
            if centroid < 2500 and high_pct < 2.0:
                for cym_idx, tom_idx in LANE_CONFLICT_PAIRS:
                    if cym_idx in fired and tom_idx not in fired:
                        fired.discard(cym_idx)
                        fired.add(tom_idx)
                        spectral_overrides += 1

        # Cap hand notes at 2 per onset (drummer has 2 hands)
        hand_fired = fired & HAND_CLASSES
        if len(hand_fired) > 2:
            hand_caps_applied += 1
            # Keep the 2 hand classes with highest probability
            ranked = sorted(hand_fired, key=lambda c: class_probs[i, c], reverse=True)
            for cls in ranked[2:]:
                fired.discard(cls)

        for cls in fired:
            lane, is_cymbal = CLASS_TO_LANE[cls]
            hits.append(DrumHit(
                time_ms=t_ms,
                tick=0,
                lane=lane,
                is_cymbal=is_cymbal,
                velocity=100,
            ))
            class_counts[cls] += 1

    # Sort by time
    hits.sort(key=lambda h: h.time_ms)

    # Streak smoothing on shared tom/cymbal lanes: prevent flip-flop.
    # Find short streaks (≤2 consecutive same-type hits) flanked by
    # longer runs of the opposite type, and convert the short streak to
    # match its surroundings.  Bidirectional: fixes both stray toms in
    # cymbal sections AND stray cymbals in tom sections.
    # Runs iteratively until no more changes (cascading short streaks).
    # Class count adjustments: lane → (cymbal_class, tom_class)
    LANE_CLASS_MAP = {2: (2, 3), 3: (4, 5), 4: (6, 7)}  # yellow, blue, green
    green_smoothed = 0  # total across all lanes
    for smooth_lane, (cym_cls, tom_cls) in LANE_CLASS_MAP.items():
        lane_indices = [idx for idx, h in enumerate(hits) if h.lane == smooth_lane]
        if len(lane_indices) < 3:
            continue

        for _pass in range(5):  # max 5 iterations
            changed_this_pass = 0
            # Build streaks: [(is_cymbal, start_j, end_j), ...]
            streaks = []
            s_start = 0
            for j in range(1, len(lane_indices)):
                if hits[lane_indices[j]].is_cymbal != hits[lane_indices[s_start]].is_cymbal:
                    streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, j - 1))
                    s_start = j
            streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, len(lane_indices) - 1))

            # Identify short streaks (≤2) flanked by opposite runs
            for si, (is_cym, j_start, j_end) in enumerate(streaks):
                streak_len = j_end - j_start + 1
                if streak_len > 2:
                    continue
                if si == 0 or si == len(streaks) - 1:
                    continue
                prev_is_cym, _, prev_end = streaks[si - 1]
                next_is_cym, next_start, _ = streaks[si + 1]
                prev_len = prev_end - streaks[si - 1][1] + 1
                next_len = streaks[si + 1][2] - next_start + 1
                if prev_is_cym == is_cym or next_is_cym == is_cym:
                    continue
                # At least one neighbor must be longer than our streak
                if prev_len <= streak_len and next_len <= streak_len:
                    continue
                target_cymbal = prev_is_cym
                for j in range(j_start, j_end + 1):
                    idx = lane_indices[j]
                    if hits[idx].is_cymbal != target_cymbal:
                        hits[idx].is_cymbal = target_cymbal
                        if target_cymbal:
                            class_counts[cym_cls] += 1; class_counts[tom_cls] -= 1
                        else:
                            class_counts[cym_cls] -= 1; class_counts[tom_cls] += 1
                        changed_this_pass += 1
                        green_smoothed += 1

            if changed_this_pass == 0:
                break

    # Dedup: remove near-simultaneous hits on same lane (keep first).
    # Sort so cymbals come before toms at same time — when two nearby
    # onsets produce a cymbal and a tom on the same lane, the cymbal
    # (dominant class) survives.
    hits.sort(key=lambda h: (h.time_ms, h.lane, not h.is_cymbal))
    deduped = []
    for hit in hits:
        if deduped and abs(hit.time_ms - deduped[-1].time_ms) < 15 \
                and hit.lane == deduped[-1].lane:
            continue
        deduped.append(hit)
    hits = deduped

    logger.info(f"  {len(hits)} hits after thresholding:")
    for cls in range(8):
        logger.info(f"    {CLASS_NAMES[cls]}: {class_counts[cls]}")
    if lane_conflicts_resolved > 0:
        logger.info(f"  Lane conflicts resolved (tom won): {lane_conflicts_resolved}")
    if hand_caps_applied > 0:
        logger.info(f"  Hand cap applied (3+ → 2 hands): {hand_caps_applied}")
    if spectral_overrides > 0:
        logger.info(f"  Spectral cymbal→tom overrides: {spectral_overrides}")
    if kick_suppressed_ftom > 0:
        logger.info(f"  Kick-suppressed FloorTom: {kick_suppressed_ftom}")
    if spectral_conflict_bias > 0:
        logger.info(f"  Spectral-biased lane conflicts: {spectral_conflict_bias}")
    if green_smoothed > 0:
        logger.info(f"  Streak smoothing: {green_smoothed} flips fixed (bidirectional)")

    # Compute tick positions for tempo change events.
    ticks_per_beat = 480
    for idx in range(1, len(tempo_events)):
        prev = tempo_events[idx - 1]
        curr = tempo_events[idx]
        delta_ms = curr.time_ms - prev.time_ms
        ms_per_beat = 60_000.0 / prev.tempo_bpm
        ms_per_tick = ms_per_beat / ticks_per_beat
        delta_ticks = int(delta_ms / ms_per_tick)
        tempo_events[idx] = TempoEvent(
            tick=prev.tick + delta_ticks,
            tempo_bpm=curr.tempo_bpm,
            time_ms=curr.time_ms,
        )

    return DrumChart(
        hits=hits,
        tempo_events=tempo_events,
        time_signatures=[TimeSignature(tick=0, numerator=4, denominator=4, time_ms=0.0)],
        ticks_per_beat=480,
    )


# ══════════════════════════════════════════════════════════════
# IO helpers
# ══════════════════════════════════════════════════════════════

def convert_to_ogg(input_path: Path, output_path: Path) -> None:
    """Copy/convert audio to output as song.ogg."""
    if input_path.suffix.lower() == ".ogg":
        shutil.copy(str(input_path), str(output_path))
    else:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(input_path),
                 "-vn", "-c:a", "libvorbis", "-q:a", "6", str(output_path)],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            shutil.copy(str(input_path), output_path.parent / f"song{input_path.suffix}")


def create_song_ini(output_dir: Path, artist: str, title: str, tempo_bpm: float, duration_ms: float) -> None:
    """Create song.ini metadata file."""
    ini_content = f"""[song]
name = {title}
artist = {artist}
charter = STRUM
diff_drums = 4
diff_drums_real = 4
diff_drums_real_ps = 4
pro_drums = True
preview_start_time = {int(min(30000, duration_ms * 0.25))}
song_length = {int(duration_ms)}
"""
    (output_dir / "song.ini").write_text(ini_content, encoding="utf-8")


def parse_filename(filepath: Path) -> tuple[str, str]:
    """Extract artist and title from 'Title - Artist' filename."""
    name = filepath.stem
    if " - " in name:
        parts = name.split(" - ", 1)
        # Files are named "Title - Artist"
        return parts[1].strip(), parts[0].strip()
    return "Unknown Artist", name


# ══════════════════════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════════════════════

def process_song(
    v14_model: TwoStageDrumsCRNN,
    ensemble: list[dict],
    audio_path: Path,
    output_dir: Path,
    device: torch.device,
    skip_separation: bool = False,
    onset_threshold: float = 0.4,
    class_thresholds: list[float] | None = None,
    two_pass_context: bool = True,
    postprocess: bool = True,
    full_audio_dir: Path | None = None,
) -> Path:
    """Process a single song through the hybrid pipeline."""
    artist, title = parse_filename(audio_path)
    safe_name = re.sub(r'[<>:"/\\|?*]', "", f"{artist} - {title}")
    song_folder = output_dir / safe_name
    song_folder.mkdir(parents=True, exist_ok=True)

    logger.info(f"Song folder: {song_folder}")

    # 1. Analyze audio (tempo, beat offset, tempo changes)
    audio_info = analyze_audio(audio_path)

    # 2. Separate drums
    if skip_separation:
        drums_path = audio_path
        logger.info("  Skipping separation (using input as drums stem)")
    else:
        drums_path = separate_drums(audio_path, song_folder)

    # 3. Stage 1: V14 onset detection
    t0 = time.time()
    onset_times_ms = detect_onsets_v14(
        v14_model, drums_path, device,
        onset_threshold=onset_threshold,
    )
    logger.info(f"  Stage 1 done: {len(onset_times_ms)} onsets ({time.time() - t0:.1f}s)")

    if not onset_times_ms:
        logger.warning("  No onsets detected! Creating empty chart.")
        chart = DrumChart(
            hits=[],
            tempo_events=audio_info["tempo_events"],
            time_signatures=[TimeSignature(tick=0, numerator=4, denominator=4, time_ms=0.0)],
            ticks_per_beat=480,
        )
    else:
        # 4. Extract mel windows at each onset
        t0 = time.time()
        any_needs_cqt = any(e["needs_cqt"] for e in ensemble)
        windows = extract_onset_windows(drums_path, onset_times_ms, needs_cqt=any_needs_cqt)
        logger.info(f"  Window extraction done ({time.time() - t0:.1f}s)")

        # 5. Stage 2: Ensemble classification
        # Pass 1: classify with zero context
        t0 = time.time()
        context = build_context_vectors(onset_times_ms)
        logits_pass1 = classify_onsets_ensemble(ensemble, windows, context, device)
        probs_pass1 = 1.0 / (1.0 + np.exp(-logits_pass1))  # sigmoid for context
        logger.info(f"  Stage 2 pass 1 done ({time.time() - t0:.1f}s)")

        if two_pass_context:
            # Pass 2: re-classify with context from pass 1
            t0 = time.time()
            context = build_context_vectors(onset_times_ms, probs_pass1)
            logits = classify_onsets_ensemble(ensemble, windows, context, device)
            logger.info(f"  Stage 2 pass 2 done ({time.time() - t0:.1f}s)")
        else:
            logits = logits_pass1

        # Convert logits to probabilities
        probs = 1.0 / (1.0 + np.exp(-logits))

        # 5b. Spectral safety net: compute spectral features for cymbal→tom
        # override. When audio at an onset is clearly tom-like (low centroid,
        # no high-frequency shimmer) and kick isn't dominant, override cymbal
        # classification to tom on shared lanes.
        t0 = time.time()
        spectral_centroids, spectral_high_pcts = _compute_spectral_centroid_features(
            drums_path, onset_times_ms
        )
        logger.info(f"  Spectral features computed ({time.time() - t0:.1f}s)")

        # 6. Build chart
        chart = build_chart(
            onset_times_ms, probs, windows["valid_mask"],
            tempo_events=audio_info["tempo_events"],
            thresholds=class_thresholds,
            spectral_centroids=spectral_centroids,
            spectral_high_pcts=spectral_high_pcts,
        )

    # 6b. Post-processing
    if chart.hits and postprocess:
        t0 = time.time()
        chart = postprocess_chart(chart)
        logger.info(f"  Post-processing done ({time.time() - t0:.1f}s)")

    # 7. Export
    notes_path = song_folder / "notes.mid"
    export_all_difficulties(chart, notes_path)
    logger.info(f"  Created: notes.mid ({len(chart.hits)} Expert hits)")

    # Use full mix audio for song.ogg, not drums stem
    song_ogg = song_folder / "song.ogg"
    ogg_source = audio_path
    if full_audio_dir is not None:
        for ext in (".mp3", ".ogg", ".wav", ".flac"):
            candidate = full_audio_dir / (audio_path.stem + ext)
            if candidate.exists():
                ogg_source = candidate
                break
    convert_to_ogg(ogg_source, song_ogg)
    logger.info(f"  Created: song.ogg (from {ogg_source.name})")

    create_song_ini(
        song_folder, artist, title,
        audio_info["tempo_bpm"], audio_info["duration_ms"],
    )
    logger.info(f"  Created: song.ini")

    return song_folder


def main():
    parser = argparse.ArgumentParser(description="Hybrid drum transcription pipeline")
    parser.add_argument("--skip-separation", action="store_true",
                        help="Skip Demucs separation (use input as drums stem)")
    parser.add_argument("--onset-threshold", type=float, default=0.35,
                        help="V14 onset detection threshold (default: 0.35)")
    parser.add_argument("--input-dir", type=str, default="input",
                        help="Input directory with audio files")
    parser.add_argument("--output-dir", type=str, default="output/hybrid",
                        help="Output directory for song folders")
    parser.add_argument("--no-two-pass", action="store_true",
                        help="Disable two-pass context (faster but less accurate)")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable chart post-processing")
    parser.add_argument("--full-audio-dir", type=str, default=None,
                        help="Directory with original full-mix audio (for song.ogg when using --skip-separation)")
    args = parser.parse_args()

    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Find input songs
    songs = sorted(
        f for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not songs:
        songs = sorted(
            f for f in INPUT_DIR.iterdir()
            if f.is_symlink() or (f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS)
        )
    logger.info(f"Found {len(songs)} songs in {INPUT_DIR}")

    # Load models
    logger.info("\n═══ Loading models ═══")
    v14_model = load_v14_onset_detector(device)
    ensemble = load_ensemble(device)

    if not ensemble:
        logger.error("No ensemble models loaded! Exiting.")
        sys.exit(1)

    # Process each song
    logger.info(f"\n═══ Processing {len(songs)} songs ═══")
    results = []
    for i, song_path in enumerate(songs):
        logger.info(f"\n{'─' * 60}")
        logger.info(f"[{i+1}/{len(songs)}] {song_path.name}")
        logger.info(f"{'─' * 60}")

        t0 = time.time()
        try:
            folder = process_song(
                v14_model, ensemble, song_path, OUTPUT_DIR, device,
                skip_separation=args.skip_separation,
                onset_threshold=args.onset_threshold,
                two_pass_context=not args.no_two_pass,
                postprocess=not args.no_postprocess,
                full_audio_dir=Path(args.full_audio_dir) if args.full_audio_dir else None,
            )
            elapsed = time.time() - t0
            results.append({"song": song_path.name, "folder": str(folder), "time": elapsed})
            logger.info(f"  Done in {elapsed:.1f}s")
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({"song": song_path.name, "error": str(e)})

    # Summary
    logger.info(f"\n{'═' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'═' * 60}")
    for r in results:
        if "error" in r:
            logger.info(f"  FAIL: {r['song']} — {r['error']}")
        else:
            logger.info(f"  OK:   {r['song']} ({r['time']:.1f}s)")

    logger.info(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
