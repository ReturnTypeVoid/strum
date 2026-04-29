#!/usr/bin/env python3
"""Sweep fret thresholds (per-bit and global) to maximize end-to-end F1.

Caches predictions per song to /mnt/ml-data/guitar_v2_cache/eval_cache_{split}.npz
so we only run inference once. Then sweeps:
  1. Global single threshold 0.10..0.90
  2. Per-bit threshold (greedy coordinate descent on bit-F1)

Reports onset / fret-bit / event-exact F1 at each setting.

Usage:
    python scripts/sweep_fret_thresholds.py --split val --max-songs 286
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.inference.guitar_neural import GuitarNeuralCharter  # noqa: E402
import preprocess_guitar_windows as pgw  # noqa: E402
from eval_guitar_neural import load_audio, parse_gt  # noqa: E402


def build_cache(args, charter, songs, cache_path: Path):
    """Run onset + fret head once per song; cache (probs, peaks, fret_probs, gt)."""
    t0 = time.time()
    songs_data = []
    for i, s in enumerate(songs):
        try:
            y = load_audio(Path(s["audio_path"]))
            offset_ms = float(s.get("audio_offset_ms", 0) or 0)
            gt = parse_gt(Path(s["midi_path"]), offset_ms)

            log_mel = charter.compute_logmel(y)
            probs = charter.predict_onset_probs(log_mel)
            peaks = charter.peak_pick(probs, args.onset_threshold,
                                      charter.default_min_dist_frames)
            fret_probs, valid = charter.predict_frets(log_mel, peaks)
            # keep only valid rows
            keep = valid
            peak_times = peaks[keep] * (pgw.HOP_LENGTH / pgw.SAMPLE_RATE)
            fret_probs = fret_probs[keep]

            songs_data.append({
                "id": s["id"],
                "peak_times": peak_times.astype(np.float32),
                "fret_probs": fret_probs.astype(np.float16),
                "gt_times": np.array([t for t, _ in gt], dtype=np.float32),
                "gt_frets": np.array(
                    [[1 if k in fs else 0 for k in range(5)] for _, fs in gt],
                    dtype=np.uint8,
                ),
            })
        except Exception as exc:
            print(f"  [skip] {s['id']}: {exc}")
        if (i + 1) % 20 == 0:
            print(f"  inference {i+1}/{len(songs)}  ({time.time()-t0:.1f}s)")

    print(f"inference done in {time.time()-t0:.1f}s")

    # Save as npz with object arrays
    np.savez(cache_path, data=np.array(songs_data, dtype=object), allow_pickle=True)
    print(f"cached to {cache_path}")
    return songs_data


def load_cache(cache_path: Path):
    z = np.load(cache_path, allow_pickle=True)
    return list(z["data"])


def evaluate(songs_data, fret_thresholds, tolerance_s: float = 0.05):
    """Compute aggregated stats given per-bit thresholds (length-5 array)."""
    fret_thresholds = np.asarray(fret_thresholds, dtype=np.float32)

    onset_tp = onset_fp = onset_fn = 0
    fret_bit_tp = fret_bit_fp = fret_bit_fn = 0
    event_tp = 0
    n_pred_total = n_gt_total = 0

    for sd in songs_data:
        pred_times = sd["peak_times"]
        fp = sd["fret_probs"].astype(np.float32)
        gt_times = sd["gt_times"]
        gt_frets = sd["gt_frets"]

        n_pred = len(pred_times)
        n_gt = len(gt_times)
        n_pred_total += n_pred
        n_gt_total += n_gt

        if n_pred == 0:
            onset_fn += n_gt
            fret_bit_fn += int(gt_frets.sum())
            continue
        if n_gt == 0:
            onset_fp += n_pred
            pred_bits_only = (fp >= fret_thresholds).astype(np.uint8)
            # Argmax fallback for empty rows
            empty = pred_bits_only.sum(1) == 0
            if empty.any():
                am = fp[empty].argmax(1)
                pred_bits_only[empty, am] = 1
            fret_bit_fp += int(pred_bits_only.sum())
            continue

        # Greedy time-match
        used = np.zeros(n_gt, dtype=bool)
        order = np.argsort(pred_times)
        j_start = 0
        matched = 0
        for idx in order:
            pt = pred_times[idx]
            while j_start < n_gt and gt_times[j_start] < pt - tolerance_s:
                j_start += 1
            best_j = -1
            best_diff = tolerance_s + 1
            j = j_start
            while j < n_gt and gt_times[j] <= pt + tolerance_s:
                if not used[j]:
                    d = abs(gt_times[j] - pt)
                    if d < best_diff:
                        best_diff = d
                        best_j = j
                j += 1
            pred_bits = (fp[idx] >= fret_thresholds).astype(np.uint8)
            if pred_bits.sum() == 0:
                pred_bits[fp[idx].argmax()] = 1
            if best_j < 0:
                # Unmatched pred — counts as FP onset, all pred bits are FP
                fret_bit_fp += int(pred_bits.sum())
                continue
            used[best_j] = True
            matched += 1
            gt_bits = gt_frets[best_j]
            tp = int(((pred_bits == 1) & (gt_bits == 1)).sum())
            fp_b = int(((pred_bits == 1) & (gt_bits == 0)).sum())
            fn_b = int(((pred_bits == 0) & (gt_bits == 1)).sum())
            fret_bit_tp += tp
            fret_bit_fp += fp_b
            fret_bit_fn += fn_b
            if np.array_equal(pred_bits, gt_bits):
                event_tp += 1
        # Unmatched GT: all GT bits become FN bits
        for j in range(n_gt):
            if not used[j]:
                fret_bit_fn += int(gt_frets[j].sum())
        onset_tp += matched
        onset_fp += n_pred - matched
        onset_fn += n_gt - matched

    def f1(tp, fp_, fn_):
        p = tp / max(tp + fp_, 1)
        r = tp / max(tp + fn_, 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)

    on_p, on_r, on_f = f1(onset_tp, onset_fp, onset_fn)
    fb_p, fb_r, fb_f = f1(fret_bit_tp, fret_bit_fp, fret_bit_fn)
    event_f = 2 * event_tp / max(n_pred_total + n_gt_total, 1)
    return {
        "onset_p": on_p, "onset_r": on_r, "onset_f1": on_f,
        "fret_bit_p": fb_p, "fret_bit_r": fb_r, "fret_bit_f1": fb_f,
        "event_f1": event_f,
        "event_tp": event_tp,
        "fb_tp": fret_bit_tp, "fb_fp": fret_bit_fp, "fb_fn": fret_bit_fn,
        "n_pred": n_pred_total, "n_gt": n_gt_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/guitar_v2.yaml")
    ap.add_argument("--manifest",
                    default="configs/guitar_v1_manifest_aligned_filtered.json")
    ap.add_argument("--onset-ckpt",
                    default="checkpoints/guitar_v2/guitar_v2_onset/best.pt")
    ap.add_argument("--fret-ckpt",
                    default="checkpoints/guitar_v2/guitar_v2_fret/best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-songs", type=int, default=9999)
    ap.add_argument("--onset-threshold", type=float, default=0.35)
    ap.add_argument("--cache",
                    default="/mnt/ml-data/guitar_v2_cache/eval_cache_{split}.npz")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--objective", choices=["event_f1", "fret_bit_f1"],
                    default="event_f1")
    args = ap.parse_args()

    cache_path = Path(args.cache.format(split=args.split))

    if args.rebuild_cache or not cache_path.exists():
        manifest = json.load(open(args.manifest))
        songs = [s for s in manifest["songs"] if s["split"] == args.split][:args.max_songs]
        print(f"building cache: {len(songs)} {args.split} songs")
        charter = GuitarNeuralCharter(
            onset_ckpt=Path(args.onset_ckpt),
            fret_ckpt=Path(args.fret_ckpt),
            config_path=Path(args.config),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        songs_data = build_cache(args, charter, songs, cache_path)
    else:
        print(f"loading cache: {cache_path}")
        songs_data = load_cache(cache_path)
    print(f"loaded {len(songs_data)} songs")

    obj_key = args.objective

    # ─── Phase 1: Global threshold sweep ───────────────────────────────────
    print("\n══ Global single-threshold sweep ══")
    print(f"{'thr':>5}  {'on_F1':>6} {'fb_F1':>6} {'ev_F1':>6}  {'fb_tp':>7} {'fb_fp':>7} {'fb_fn':>7}")
    print("-" * 60)
    best_global = (-1.0, None, None)
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        m = evaluate(songs_data, [thr] * 5)
        score = m[obj_key]
        marker = " *" if score > best_global[0] else ""
        print(f"{thr:>5.2f}  {m['onset_f1']:>6.3f} {m['fret_bit_f1']:>6.3f} {m['event_f1']:>6.3f}  "
              f"{m['fb_tp']:>7d} {m['fb_fp']:>7d} {m['fb_fn']:>7d}{marker}")
        if score > best_global[0]:
            best_global = (score, thr, m)
    print(f"\nbest global: thr={best_global[1]} {obj_key}={best_global[0]:.3f}")

    # ─── Phase 2: Per-bit greedy coordinate descent ─────────────────────────
    print("\n══ Per-bit greedy optimization ══")
    thresholds = [best_global[1]] * 5
    best_score = best_global[0]
    grid = np.arange(0.10, 0.81, 0.025)

    for sweep in range(3):
        improved = False
        for bit in range(5):
            best_t = thresholds[bit]
            best_b = best_score
            for t in grid:
                trial = list(thresholds)
                trial[bit] = float(t)
                m = evaluate(songs_data, trial)
                if m[obj_key] > best_b + 1e-5:
                    best_b = m[obj_key]
                    best_t = float(t)
            if best_t != thresholds[bit]:
                thresholds[bit] = best_t
                improved = True
                print(f"  sweep {sweep} bit {bit}: thr->{best_t:.3f}  {obj_key}={best_b:.4f}")
                best_score = best_b
        if not improved:
            print(f"  sweep {sweep}: no improvement")
            break

    print(f"\noptimal per-bit thresholds: {[round(t,3) for t in thresholds]}")
    final = evaluate(songs_data, thresholds)
    print(f"\n══ Final with optimal per-bit thresholds ══")
    print(f"  ONSET F1   : {final['onset_f1']:.3f}  (P={final['onset_p']:.3f} R={final['onset_r']:.3f})")
    print(f"  FRET-BIT F1: {final['fret_bit_f1']:.3f}  (P={final['fret_bit_p']:.3f} R={final['fret_bit_r']:.3f})")
    print(f"  EVENT F1   : {final['event_f1']:.3f}  (event_tp={final['event_tp']})")

    # Save thresholds for later use
    out_path = Path("outputs/guitar_v2/optimal_fret_thresholds.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "split": args.split,
        "objective": args.objective,
        "onset_threshold": args.onset_threshold,
        "fret_thresholds_per_bit": thresholds,
        "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                    for k, v in final.items()},
    }, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
