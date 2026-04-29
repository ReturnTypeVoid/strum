#!/usr/bin/env python3
"""Compute per-song MIDI/audio alignment offsets using the V1 onset model.

For each song in the manifest:
  1. Run GuitarOnsetCRNN over the full audio to get a frame-level onset probability.
  2. Peak-pick predictions at a low threshold to get candidate onset times.
  3. Sweep alignment offsets in ±SWEEP_MS (5 ms grid) and pick the offset
     that maximises recall@50 ms against the MIDI ground truth.
  4. Score confidence as (best_recall - mean_recall) / (1 - mean_recall + eps).
     Songs with low confidence keep offset=0 (no alignment) and are flagged.

Writes a NEW manifest with `audio_offset_ms` and `align_confidence` per song.

Usage:
    python scripts/align_manifest_offsets.py \
        --manifest configs/guitar_v1_manifest.json \
        --checkpoint checkpoints/guitar_v1/guitar_v1_onset/best.pt \
        --output configs/guitar_v1_manifest_aligned.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.signal import find_peaks
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.guitar_v1 import GuitarOnsetCRNN, OnsetCRNNConfig  # noqa: E402
import preprocess_guitar_windows as pgw  # noqa: E402
import eval_guitar_onset as ego  # noqa: E402


def compute_offset_for_song(
    song: dict,
    model: torch.nn.Module,
    mel_extractor,
    device: str,
    sr: int,
    frame_s: float,
    sweep_ms: int,
    grid_ms: int,
    peak_threshold: float,
    tolerance_s: float,
) -> tuple[int, float, float, int, int] | None:
    """Returns (best_offset_ms, confidence, baseline_recall, n_pred, n_gt) or None."""
    audio_path = Path(song["audio_path"])
    midi_path = Path(song["midi_path"])
    if not audio_path.exists() or not midi_path.exists():
        return None
    try:
        y = ego.load_audio(audio_path, sr=sr)
        mel = ego.compute_full_logmel(y, mel_extractor).to(device).unsqueeze(0)
        with torch.no_grad():
            probs = torch.sigmoid(model(mel)).squeeze(0).cpu().numpy()
        events = pgw.parse_onsets_from_manifest(midi_path)
        if not events:
            return None
        gt = np.array([t / 1000.0 for (t, _) in events])
    except Exception:
        return None

    peaks, _ = find_peaks(probs, height=peak_threshold, distance=1)
    pred = peaks * frame_s
    if len(pred) == 0 or len(gt) == 0:
        return None

    offsets = np.arange(-sweep_ms, sweep_ms + 1, grid_ms)
    recalls = np.empty(len(offsets), dtype=np.float32)
    for i, off_ms in enumerate(offsets):
        tp, _, fn = ego.onset_f1(pred + off_ms / 1000.0, gt, tolerance_s)
        recalls[i] = tp / max(tp + fn, 1)
    best_idx = int(np.argmax(recalls))
    best_off = int(offsets[best_idx])
    best_r = float(recalls[best_idx])
    mean_r = float(np.mean(recalls))
    baseline_r = float(recalls[len(offsets) // 2])  # offset=0
    # Confidence = how much better than random alignment
    conf = (best_r - mean_r) / max(1.0 - mean_r, 1e-3)
    return best_off, conf, baseline_r, len(pred), len(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/guitar_v1_manifest.json")
    ap.add_argument("--checkpoint", default="checkpoints/guitar_v1/guitar_v1_onset/best.pt")
    ap.add_argument("--config", default="configs/guitar_v1.yaml")
    ap.add_argument("--output", default="configs/guitar_v1_manifest_aligned.json")
    ap.add_argument("--sweep-ms", type=int, default=400)
    ap.add_argument("--grid-ms", type=int, default=5)
    ap.add_argument("--peak-threshold", type=float, default=0.10)
    ap.add_argument("--tolerance-ms", type=int, default=50)
    ap.add_argument("--min-confidence", type=float, default=0.10,
                    help="Below this, keep offset=0 and flag low_align_confidence")
    ap.add_argument("--max-shift-ms", type=int, default=350,
                    help="Reject |offset| > this (likely false alignment)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    sr = cfg["audio"]["sample_rate"]
    hop = cfg["audio"]["hop_length"]
    frame_s = hop / sr

    print(f"loading model from {args.checkpoint}")
    model = GuitarOnsetCRNN(OnsetCRNNConfig(**cfg["onset"]["model"])).to(args.device)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    mel_extractor = pgw.get_mel()

    manifest = json.load(open(args.manifest))
    songs = manifest["songs"]
    if args.limit > 0:
        songs = songs[: args.limit]
    print(f"processing {len(songs)} songs (sweep ±{args.sweep_ms}ms / grid {args.grid_ms}ms)")

    results = []
    skipped = 0
    low_conf = 0
    out_of_range = 0
    aligned_count = 0
    t0 = time.time()

    for s in tqdm(songs, desc="align"):
        out = compute_offset_for_song(
            s, model, mel_extractor, args.device, sr, frame_s,
            args.sweep_ms, args.grid_ms, args.peak_threshold, args.tolerance_ms / 1000.0,
        )
        s2 = dict(s)
        if out is None:
            skipped += 1
            s2["audio_offset_ms"] = 0
            s2["align_confidence"] = 0.0
            s2["align_status"] = "skipped"
        else:
            best_off, conf, base_r, n_pred, n_gt = out
            s2["align_baseline_recall"] = round(base_r, 4)
            s2["align_confidence"] = round(float(conf), 4)
            s2["align_n_pred"] = n_pred
            s2["align_n_gt"] = n_gt
            if conf < args.min_confidence:
                low_conf += 1
                s2["audio_offset_ms"] = 0
                s2["align_status"] = "low_confidence"
            elif abs(best_off) > args.max_shift_ms:
                out_of_range += 1
                s2["audio_offset_ms"] = 0
                s2["align_status"] = "out_of_range"
                s2["align_proposed_offset_ms"] = best_off
            else:
                aligned_count += 1
                s2["audio_offset_ms"] = best_off
                s2["align_status"] = "aligned"
        results.append(s2)

    elapsed = time.time() - t0
    offs = np.array([r["audio_offset_ms"] for r in results
                     if r.get("align_status") == "aligned"])
    print(f"\n=== alignment complete in {elapsed/60:.1f} min ===")
    print(f"  aligned:        {aligned_count} ({aligned_count/len(songs)*100:.1f}%)")
    print(f"  low_confidence: {low_conf}")
    print(f"  out_of_range:   {out_of_range}")
    print(f"  skipped:        {skipped}")
    if len(offs):
        print(f"  |offset| stats:  median={np.median(np.abs(offs)):.0f}ms"
              f"  mean={np.mean(np.abs(offs)):.0f}ms"
              f"  p90={np.percentile(np.abs(offs),90):.0f}ms"
              f"  max={np.max(np.abs(offs))}ms")
        # Sign distribution
        n_pos = int((offs > 0).sum()); n_neg = int((offs < 0).sum()); n_zero = int((offs == 0).sum())
        print(f"  sign:  +{n_pos}  -{n_neg}  =0:{n_zero}")

    out_manifest = dict(manifest)
    out_manifest["songs"] = results
    out_manifest["alignment"] = {
        "checkpoint": args.checkpoint,
        "sweep_ms": args.sweep_ms,
        "grid_ms": args.grid_ms,
        "peak_threshold": args.peak_threshold,
        "tolerance_ms": args.tolerance_ms,
        "min_confidence": args.min_confidence,
        "max_shift_ms": args.max_shift_ms,
        "aligned": aligned_count,
        "low_confidence": low_conf,
        "out_of_range": out_of_range,
        "skipped": skipped,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out_manifest, open(args.output, "w"), indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
