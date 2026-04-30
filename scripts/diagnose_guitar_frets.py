#!/usr/bin/env python3
"""Diagnose WHY guitar fret-set accuracy is so bad (event F1 = 0.04).

For matched onsets (timing within 50ms), break down failure modes:
  1. GT is a chord, pred is single note  -> 'pred_missing_chord'
  2. GT is single, pred is chord         -> 'pred_extra_notes'
  3. Both single, wrong fret             -> 'wrong_fret_X' (delta)
  4. Exact match                         -> 'correct'
  5. Both chord, partial match           -> 'chord_partial'

Also show: GT chord rate, pred chord rate, fret distribution histograms.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import preprocess_guitar_windows as pgw  # noqa: E402
from src.inference.guitar_bass import transcribe_guitar  # noqa: E402


def parse_gt(midi_path, offset_ms):
    raw = pgw.parse_onsets_from_manifest(midi_path)
    out = []
    for tms, frets in raw:
        t_sec = (tms - offset_ms) / 1000.0
        if t_sec < 0:
            continue
        playable = frozenset(f for f in frets if 0 <= f <= 4)
        if not playable:
            continue
        out.append((t_sec, playable))
    out.sort()
    return out


def chart_events(chart):
    out = []
    for n in chart.notes:
        out.append((n.time_ms / 1000.0, frozenset([n.fret])))
    for c in chart.chords:
        out.append((c.time_ms / 1000.0, frozenset(c.frets)))
    out.sort()
    return out


def diagnose(pred, gt, tol=0.05):
    pt = np.array([t for t, _ in pred])
    pf = [f for _, f in pred]
    gt_t = np.array([t for t, _ in gt])
    gt_f = [f for _, f in gt]

    n_pred, n_gt = len(pt), len(gt_t)
    used = np.zeros(n_gt, dtype=bool)

    cats = Counter()
    fret_deltas = []
    gt_fret_dist = Counter()
    pred_fret_dist = Counter()

    # Counts of chord vs single in GT and pred
    gt_chord_count = sum(1 for f in gt_f if len(f) > 1)
    pred_chord_count = sum(1 for f in pf if len(f) > 1)

    for f in gt_f:
        for x in f:
            gt_fret_dist[x] += 1
    for f in pf:
        for x in f:
            pred_fret_dist[x] += 1

    j_start = 0
    for i in np.argsort(pt):
        ptime = pt[i]
        pfrets = pf[i]
        while j_start < n_gt and gt_t[j_start] < ptime - tol:
            j_start += 1
        best_j = -1
        best_d = tol + 1
        j = j_start
        while j < n_gt and gt_t[j] <= ptime + tol:
            if not used[j]:
                d = abs(gt_t[j] - ptime)
                if d < best_d:
                    best_d = d
                    best_j = j
            j += 1
        if best_j < 0:
            cats["pred_no_match"] += 1
            continue
        used[best_j] = True
        gtfrets = gt_f[best_j]

        if pfrets == gtfrets:
            cats["correct"] += 1
        elif len(gtfrets) > 1 and len(pfrets) == 1:
            cats["gt_chord_pred_single"] += 1
        elif len(pfrets) > 1 and len(gtfrets) == 1:
            cats["gt_single_pred_chord"] += 1
        elif len(pfrets) == 1 and len(gtfrets) == 1:
            d = abs(next(iter(pfrets)) - next(iter(gtfrets)))
            cats[f"single_off_by_{d}"] += 1
            fret_deltas.append(next(iter(pfrets)) - next(iter(gtfrets)))
        else:  # both chords, not equal
            shared = pfrets & gtfrets
            if shared:
                cats["chord_partial_overlap"] += 1
            else:
                cats["chord_no_overlap"] += 1

    cats["gt_unmatched"] = int((~used).sum())
    return cats, fret_deltas, gt_fret_dist, pred_fret_dist, gt_chord_count, pred_chord_count, n_gt, n_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/guitar_v1_manifest.json")
    ap.add_argument("--max-songs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-root", default="/mnt/ml-data/eval_stems_cache")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    songs = [s for s in manifest["songs"] if s["split"] == "val"]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(songs)
    songs = songs[: args.max_songs]

    total_cats = Counter()
    total_deltas = []
    total_gt_fret = Counter()
    total_pred_fret = Counter()
    total_gt_chord = total_pred_chord = total_n_gt = total_n_pred = 0

    for s in songs:
        sid = s["id"]
        cache = Path(args.cache_root) / sid.replace("/", "_")
        stem = cache / "guitar.wav"
        if not stem.exists():
            stem = cache / "other.wav"
        if not stem.exists():
            print(f"[skip] {sid}: no cached stem")
            continue
        try:
            chart = transcribe_guitar(stem)
            offset_ms = float(s.get("audio_offset_ms", 0) or 0)
            gt = parse_gt(Path(s["midi_path"]), offset_ms)
            pred = chart_events(chart)
            cats, deltas, gd, pdist, gc, pc, ng, np_ = diagnose(pred, gt)
            print(f"{sid[:55]:55} gt_chord={gc}/{ng} ({100*gc/max(ng,1):.0f}%) "
                  f"pred_chord={pc}/{np_} ({100*pc/max(np_,1):.0f}%)")
            total_cats.update(cats)
            total_deltas.extend(deltas)
            total_gt_fret.update(gd)
            total_pred_fret.update(pdist)
            total_gt_chord += gc
            total_pred_chord += pc
            total_n_gt += ng
            total_n_pred += np_
        except Exception as e:
            print(f"[skip] {sid}: {e}")

    print("\n" + "=" * 70)
    print("FAILURE BREAKDOWN (across matched + unmatched onsets)")
    total = sum(total_cats.values())
    for cat, ct in sorted(total_cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:30} {ct:6d}  ({100*ct/total:.1f}%)")

    print(f"\nGT chord rate:   {100*total_gt_chord/max(total_n_gt,1):.1f}%  ({total_gt_chord}/{total_n_gt})")
    print(f"Pred chord rate: {100*total_pred_chord/max(total_n_pred,1):.1f}%  ({total_pred_chord}/{total_n_pred})")

    print("\nFret distribution (counts of each fret across all events):")
    for f in range(5):
        gv = total_gt_fret.get(f, 0)
        pv = total_pred_fret.get(f, 0)
        gp = 100 * gv / max(sum(total_gt_fret.values()), 1)
        pp = 100 * pv / max(sum(total_pred_fret.values()), 1)
        print(f"  fret {f}: GT {gv:5d} ({gp:4.1f}%)   PRED {pv:5d} ({pp:4.1f}%)")

    if total_deltas:
        d = np.array(total_deltas)
        print(f"\nSingle-note fret delta (pred - gt): "
              f"median={np.median(d):+.1f}  mean={d.mean():+.2f}  "
              f"|abs|mean={np.abs(d).mean():.2f}")
        print(f"  delta distribution: {Counter(d.tolist())}")
    print("=" * 70)


if __name__ == "__main__":
    main()
