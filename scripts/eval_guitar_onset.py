#!/usr/bin/env python3
"""Evaluate GuitarOnsetCRNN at the onset level (50 ms tolerance).

Loads best.pt, runs full-song mel inference, peak-picks, compares against
GT onset times from the manifest. This is the metric that actually matters
(frame-level BCE F1 over-counts because targets are smeared).

Usage:
    python scripts/eval_guitar_onset.py --split val --max-songs 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.guitar_v1 import GuitarOnsetCRNN, OnsetCRNNConfig  # noqa: E402

# Reuse exact preprocessing constants
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import preprocess_guitar_windows as pgw  # noqa: E402


def load_audio(path: Path, sr: int = 22050) -> np.ndarray:
    y, file_sr = sf.read(str(path), always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def compute_full_logmel(y: np.ndarray, mel_extractor) -> torch.Tensor:
    """Return (1, n_mels, T) log-mel tensor on CPU."""
    x = torch.from_numpy(y).float().unsqueeze(0)
    mel = mel_extractor(x)                          # (1, n_mels, T)
    return torch.log1p(mel)


def peak_pick(probs: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    """Local-max peak picking on a 1-D prob array."""
    above = probs >= threshold
    out = []
    last = -10**9
    for i in range(1, len(probs) - 1):
        if not above[i]:
            continue
        if probs[i] < probs[i - 1] or probs[i] < probs[i + 1]:
            continue
        if i - last < min_distance:
            # keep the larger one
            if out and probs[i] > probs[out[-1]]:
                out[-1] = i
                last = i
            continue
        out.append(i)
        last = i
    return np.array(out, dtype=np.int64)


def onset_f1(pred_times: np.ndarray, gt_times: np.ndarray,
             tolerance_s: float = 0.05) -> tuple[int, int, int]:
    """Greedy bipartite match within tolerance. Returns (tp, fp, fn)."""
    if len(pred_times) == 0:
        return 0, 0, len(gt_times)
    if len(gt_times) == 0:
        return 0, len(pred_times), 0
    pred_sorted = np.sort(pred_times)
    gt_sorted = np.sort(gt_times)
    used_gt = np.zeros(len(gt_sorted), dtype=bool)
    tp = 0
    j_start = 0
    for pt in pred_sorted:
        # advance j_start past any GT outside tolerance window
        while j_start < len(gt_sorted) and gt_sorted[j_start] < pt - tolerance_s:
            j_start += 1
        # find nearest unused GT within tolerance
        best_j = -1
        best_diff = tolerance_s + 1
        j = j_start
        while j < len(gt_sorted) and gt_sorted[j] <= pt + tolerance_s:
            if not used_gt[j]:
                d = abs(gt_sorted[j] - pt)
                if d < best_diff:
                    best_diff = d
                    best_j = j
            j += 1
        if best_j >= 0:
            used_gt[best_j] = True
            tp += 1
    fp = len(pred_sorted) - tp
    fn = len(gt_sorted) - tp
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/guitar_v1.yaml")
    ap.add_argument("--manifest", default="configs/guitar_v1_manifest.json")
    ap.add_argument("--checkpoint", default="checkpoints/guitar_v1/guitar_v1_onset/best.pt")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--max-songs", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override peak threshold (default from config)")
    ap.add_argument("--min-distance-ms", type=float, default=None)
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sweep", action="store_true",
                    help="Sweep thresholds 0.1-0.7 in 0.05 steps and report best")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    o = cfg["onset"]
    inf = o["inference"]
    sr = cfg["audio"]["sample_rate"]
    hop = cfg["audio"]["hop_length"]
    frame_ms = hop / sr * 1000.0

    threshold = args.threshold if args.threshold is not None else inf["peak_threshold"]
    min_dist_ms = (args.min_distance_ms
                   if args.min_distance_ms is not None
                   else inf["peak_min_distance_frames"] * frame_ms)
    min_dist_frames = max(1, int(round(min_dist_ms / frame_ms)))

    print(f"config: threshold={threshold} min_dist={min_dist_ms:.1f}ms "
          f"({min_dist_frames}f) tolerance={args.tolerance_ms}ms")

    # Load model
    model = GuitarOnsetCRNN(OnsetCRNNConfig(**o["model"])).to(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint}  epoch={ck.get('epoch','?')} "
          f"frame_f1={ck.get('val_f1','?'):.3f}")

    manifest = json.load(open(args.manifest))
    songs = [s for s in manifest["songs"] if s["split"] == args.split]
    songs = songs[:args.max_songs]
    print(f"evaluating {len(songs)} {args.split} songs")

    mel_ext = pgw.get_mel()

    if args.sweep:
        thresholds = [round(0.10 + 0.05 * i, 2) for i in range(13)]  # 0.10..0.70
    else:
        thresholds = [threshold]

    # Cache per-song probs to avoid re-running inference for sweep
    all_probs: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    skipped = 0

    t0 = time.time()
    with torch.no_grad():
        for i, s in enumerate(songs):
            try:
                y = load_audio(Path(s["audio_path"]), sr=sr)
                mel = compute_full_logmel(y, mel_ext).to(args.device)   # (1, n_mels, T)
                mel = mel.unsqueeze(0)                                   # (1, 1, n_mels, T)
                logits = model(mel)                                      # (1, T)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                all_probs.append(probs)

                # Re-parse GT onset times from notes.mid
                events = pgw.parse_onsets_from_manifest(Path(s["midi_path"]))
                gt_times = np.array([t / 1000.0 for (t, _frets) in events])
                all_gt.append(gt_times)
            except Exception as exc:
                print(f"  [skip] {s['id']}: {exc}")
                skipped += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(songs)}  ({time.time() - t0:.1f}s)")

    print(f"inference done in {time.time() - t0:.1f}s "
          f"({len(all_probs)} ok, {skipped} skipped)")

    # Sweep / eval
    print()
    print(f"{'thr':>5}  {'TP':>6} {'FP':>6} {'FN':>6}  "
          f"{'P':>6} {'R':>6} {'F1':>6}")
    print("-" * 50)
    best = (-1.0, None)
    for th in thresholds:
        tp = fp = fn = 0
        for probs, gt in zip(all_probs, all_gt):
            peaks = peak_pick(probs, th, min_dist_frames)
            pred_times = peaks * (frame_ms / 1000.0)
            t, f_p, f_n = onset_f1(pred_times, gt, args.tolerance_ms / 1000.0)
            tp += t; fp += f_p; fn += f_n
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"{th:>5.2f}  {tp:>6d} {fp:>6d} {fn:>6d}  "
              f"{p:>6.3f} {r:>6.3f} {f1:>6.3f}")
        if f1 > best[0]:
            best = (f1, th)

    if args.sweep:
        print(f"\nBEST: threshold={best[1]} F1={best[0]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
