#!/usr/bin/env python3
"""Diagnose what onsets the V1 onset CRNN is missing.

For a few val songs:
  1. Run inference
  2. Peak-pick predictions
  3. Greedy-match against GT
  4. For misses (FN) and false positives (FP), compute:
       - inter-onset interval (close to other notes? burst/sustain?)
       - prob value at GT frame (just below threshold?)
       - n_frets at GT (single note vs chord)
       - velocity / energy in mel near onset
       - position in song (intro/outro?)

Outputs a markdown report and a per-song JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from src.models.guitar_v1 import GuitarOnsetCRNN, OnsetCRNNConfig  # noqa: E402
import preprocess_guitar_windows as pgw  # noqa: E402
import eval_guitar_onset as ego  # noqa: E402


def analyze_song(model, song, cfg, device, threshold, min_dist_frames,
                 tolerance_s, frame_ms):
    sr = cfg["audio"]["sample_rate"]
    y = ego.load_audio(Path(song["audio_path"]), sr=sr)
    mel_ext = pgw.get_mel()
    mel = ego.compute_full_logmel(y, mel_ext).to(device).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(mel)).squeeze(0).cpu().numpy()

    # GT
    events = pgw.parse_onsets_from_manifest(Path(song["midi_path"]))
    gt_times = np.array([t / 1000.0 for (t, _f) in events])
    gt_frets = [f for (_t, f) in events]

    # Peak pick
    peaks = ego.peak_pick(probs, threshold, min_dist_frames)
    pred_times = peaks * (frame_ms / 1000.0)

    # Greedy match
    used = np.zeros(len(gt_times), dtype=bool)
    matched_gt: list[int] = []
    matched_pred: list[int] = []
    fp_idx: list[int] = []
    for pi, pt in enumerate(pred_times):
        best_j, best_d = -1, tolerance_s + 1
        for j, gt in enumerate(gt_times):
            if used[j]:
                continue
            if abs(gt - pt) <= tolerance_s and abs(gt - pt) < best_d:
                best_j, best_d = j, abs(gt - pt)
        if best_j >= 0:
            used[best_j] = True
            matched_gt.append(best_j); matched_pred.append(pi)
        else:
            fp_idx.append(pi)
    fn_idx = [j for j in range(len(gt_times)) if not used[j]]

    # ── Miss diagnostics ───────────────────────────────────────────────
    miss_records = []
    for j in fn_idx:
        t = gt_times[j]
        frame = int(t * 1000.0 / frame_ms)
        # peak prob in ±2 frame window
        lo, hi = max(0, frame - 2), min(len(probs), frame + 3)
        local_peak = float(probs[lo:hi].max()) if hi > lo else 0.0
        # nearest neighbor IOI
        ioi = 999.0
        if j > 0:
            ioi = min(ioi, t - gt_times[j - 1])
        if j < len(gt_times) - 1:
            ioi = min(ioi, gt_times[j + 1] - t)
        # chord size
        n_frets = len(gt_frets[j])
        miss_records.append({
            "time": float(t),
            "ioi_ms": float(ioi * 1000.0),
            "n_frets": n_frets,
            "local_peak_prob": local_peak,
            "below_thr": local_peak < threshold,
        })

    fp_records = []
    for pi in fp_idx:
        t = pred_times[pi]
        # nearest GT
        nearest = float(min((abs(g - t) for g in gt_times), default=999))
        fp_records.append({
            "time": float(t),
            "prob": float(probs[peaks[pi]]),
            "nearest_gt_ms": float(nearest * 1000.0),
        })

    return {
        "song_id": song["id"],
        "n_gt": len(gt_times),
        "n_pred": len(pred_times),
        "tp": len(matched_gt),
        "fp": len(fp_idx),
        "fn": len(fn_idx),
        "precision": len(matched_gt) / max(len(pred_times), 1),
        "recall": len(matched_gt) / max(len(gt_times), 1),
        "misses": miss_records,
        "false_positives": fp_records,
    }


def summarize(reports, threshold):
    print()
    print("=" * 70)
    print(f"SUMMARY across {len(reports)} songs (threshold={threshold})")
    print("=" * 70)

    # Aggregate
    all_miss = [m for r in reports for m in r["misses"]]
    all_fp = [f for r in reports for f in r["false_positives"]]
    print(f"Total misses: {len(all_miss)}   Total FPs: {len(all_fp)}")

    if all_miss:
        # 1) How often is the prob actually high but peak picker missed it?
        below = sum(1 for m in all_miss if m["below_thr"])
        above = len(all_miss) - below
        print(f"\nMiss reason:")
        print(f"  below threshold ({threshold}): {below:>5} ({below/len(all_miss)*100:.1f}%)  "
              f"← model under-confident here")
        print(f"  above threshold (peak/distance lost): {above:>5} ({above/len(all_miss)*100:.1f}%)  "
              f"← peak-picker / min-distance suppressed")

        # 2) Local peak prob distribution for missed
        peaks_below = [m["local_peak_prob"] for m in all_miss if m["below_thr"]]
        if peaks_below:
            arr = np.array(peaks_below)
            print(f"\nLocal peak prob @ missed onsets (when below thr):")
            for q in [10, 25, 50, 75, 90]:
                print(f"  p{q}: {np.percentile(arr, q):.3f}")

        # 3) IOI distribution for misses
        iois = np.array([m["ioi_ms"] for m in all_miss if m["ioi_ms"] < 999])
        print(f"\nIOI distribution (ms) at missed onsets:")
        for q in [10, 25, 50, 75, 90]:
            print(f"  p{q}: {np.percentile(iois, q):.0f}")
        # very-close-neighbor count
        burst = (iois < 100).sum()
        print(f"  IOI<100ms (burst): {burst} / {len(iois)} ({burst/len(iois)*100:.1f}%)")

        # 4) Chord vs single
        nfret_dist = Counter(m["n_frets"] for m in all_miss)
        total = sum(nfret_dist.values())
        print(f"\nMiss by chord size:")
        for k in sorted(nfret_dist):
            print(f"  {k}-fret: {nfret_dist[k]:>5} ({nfret_dist[k]/total*100:.1f}%)")

    if all_fp:
        # Distance to nearest GT
        nearest = np.array([f["nearest_gt_ms"] for f in all_fp])
        within_100 = (nearest < 100).sum()
        within_50 = (nearest < 50).sum()
        between_50_100 = within_100 - within_50
        far = len(all_fp) - within_100
        print(f"\nFP nearest-GT distance:")
        print(f"  <50ms (just outside tol): {within_50:>5} ({within_50/len(all_fp)*100:.1f}%)")
        print(f"  50-100ms (echo/decay):    {between_50_100:>5} ({between_50_100/len(all_fp)*100:.1f}%)")
        print(f"  >100ms (true FP):         {far:>5} ({far/len(all_fp)*100:.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/guitar_v1.yaml")
    ap.add_argument("--manifest", default="configs/guitar_v1_manifest.json")
    ap.add_argument("--checkpoint", default="checkpoints/guitar_v1/guitar_v1_onset/best.pt")
    ap.add_argument("--n-songs", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--output", default="outputs/guitar_v1_failure_analysis.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    o = cfg["onset"]
    sr = cfg["audio"]["sample_rate"]
    hop = cfg["audio"]["hop_length"]
    frame_ms = hop / sr * 1000.0
    min_dist_frames = o["inference"]["peak_min_distance_frames"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GuitarOnsetCRNN(OnsetCRNNConfig(**o["model"])).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    manifest = json.load(open(args.manifest))
    songs = [s for s in manifest["songs"] if s["split"] == "val"][:args.n_songs]
    print(f"analyzing {len(songs)} val songs at threshold={args.threshold}")

    reports = []
    for s in songs:
        try:
            r = analyze_song(model, s, cfg, device, args.threshold,
                             min_dist_frames, args.tolerance_ms / 1000.0, frame_ms)
            reports.append(r)
            print(f"  {s['id'][:60]:<60}  P={r['precision']:.2f} R={r['recall']:.2f}  "
                  f"({r['tp']}/{r['n_gt']})")
        except Exception as exc:
            print(f"  [skip] {s['id']}: {exc}")

    summarize(reports, args.threshold)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nfull report → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
