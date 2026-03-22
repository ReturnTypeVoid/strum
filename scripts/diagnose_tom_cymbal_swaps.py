#!/usr/bin/env python3
"""Diagnose tom/cymbal swaps in drum rolls.

Traces the is_cymbal decision for every hit on shared lanes (2/3/4)
through the full pipeline:
  1. Initial classification (above threshold)
  2. Spectral lane-conflict resolution
  3. Streak smoothing

Reports which stage flips toms to cymbals in rapid cross-lane patterns (rolls/fills).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
import logging
from collections import defaultdict

from scripts.batch_infer_hybrid import (
    load_v14_onset_detector,
    load_ensemble,
    detect_onsets_v14,
    extract_onset_windows,
    classify_onsets_ensemble,
    build_context_vectors,
    _compute_spectral_centroid_features,
    CLASS_NAMES,
    CLASS_TO_LANE,
    DEFAULT_CLASS_THRESHOLDS,
    OC_SR,
)
from src.preprocessing.parsers.midi_parser import DrumHit, TempoEvent

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LANE_CONFLICT_PAIRS = [(2, 3), (4, 5), (6, 7)]
LANE_CLASS_MAP = {2: (2, 3), 3: (4, 5), 4: (6, 7)}
HAND_CLASSES = {1, 2, 3, 4, 5, 6, 7}


def trace_build_chart(
    onset_times_ms: list[float],
    class_probs: np.ndarray,
    valid_mask: np.ndarray,
    spectral_centroids: np.ndarray,
    spectral_high_pcts: np.ndarray,
    thresholds: list[float],
):
    """Replicate build_chart with tracing at each decision point."""

    # Stage 1: Initial classification
    initial_hits = []
    for i, t_ms in enumerate(onset_times_ms):
        if not valid_mask[i]:
            continue
        fired = set()
        for cls in range(8):
            if class_probs[i, cls] > thresholds[cls]:
                fired.add(cls)
        for cls in fired:
            lane, is_cymbal = CLASS_TO_LANE[cls]
            initial_hits.append({
                'time_ms': t_ms,
                'onset_idx': i,
                'cls': cls,
                'cls_name': CLASS_NAMES[cls],
                'lane': lane,
                'is_cymbal': is_cymbal,
                'prob': class_probs[i, cls],
                'centroid': spectral_centroids[i],
                'high_pct': spectral_high_pcts[i],
            })

    # Stage 2: After spectral lane-conflict resolution + all disambiguation
    resolved_hits = []
    for i, t_ms in enumerate(onset_times_ms):
        if not valid_mask[i]:
            continue
        fired = set()
        for cls in range(8):
            if class_probs[i, cls] > thresholds[cls]:
                fired.add(cls)

        for cym_idx, tom_idx in LANE_CONFLICT_PAIRS:
            if cym_idx in fired and tom_idx in fired:
                cym_p = class_probs[i, cym_idx]
                tom_p = class_probs[i, tom_idx]
                centroid = spectral_centroids[i]
                if centroid < 2500:
                    tom_p *= 2.0
                elif centroid < 4000:
                    tom_p *= 1.3
                if cym_p >= tom_p:
                    fired.discard(tom_idx)
                else:
                    fired.discard(cym_idx)
            elif tom_idx in fired and cym_idx not in fired:
                cym_p = class_probs[i, cym_idx]
                tom_p = class_probs[i, tom_idx]
                swap = False
                if cym_p > tom_p:
                    swap = True
                    if spectral_centroids[i] < 3000:
                        swap = False
                if swap:
                    fired.discard(tom_idx)
                    fired.add(cym_idx)

        if 0 in fired and 7 in fired:
            if class_probs[i, 0] > 0.5 and class_probs[i, 7] < 0.35:
                fired.discard(7)

        if class_probs[i, 0] < 0.7:
            centroid = spectral_centroids[i]
            high_pct = spectral_high_pcts[i]
            if centroid < 2500 and high_pct < 2.0:
                for cym_idx, tom_idx in LANE_CONFLICT_PAIRS:
                    if cym_idx in fired and tom_idx not in fired:
                        fired.discard(cym_idx)
                        fired.add(tom_idx)

        hand_fired = fired & HAND_CLASSES
        if len(hand_fired) > 2:
            ranked = sorted(hand_fired, key=lambda c: class_probs[i, c], reverse=True)
            for cls in ranked[2:]:
                fired.discard(cls)

        for cls in fired:
            lane, is_cymbal = CLASS_TO_LANE[cls]
            resolved_hits.append({
                'time_ms': t_ms,
                'onset_idx': i,
                'cls': cls,
                'cls_name': CLASS_NAMES[cls],
                'lane': lane,
                'is_cymbal': is_cymbal,
                'prob': class_probs[i, cls],
                'centroid': spectral_centroids[i],
                'high_pct': spectral_high_pcts[i],
            })

    # Stage 3: Streak smoothing
    hits = []
    for h in resolved_hits:
        hits.append(DrumHit(
            time_ms=h['time_ms'],
            tick=0,
            lane=h['lane'],
            is_cymbal=h['is_cymbal'],
            velocity=100,
        ))
    hits.sort(key=lambda h: h.time_ms)

    # Build centroid lookup
    centroid_by_time = {}
    for i, t_ms in enumerate(onset_times_ms):
        centroid_by_time[t_ms] = spectral_centroids[i]

    flips = []
    for smooth_lane, (cym_cls, tom_cls) in LANE_CLASS_MAP.items():
        lane_indices = [idx for idx, h in enumerate(hits) if h.lane == smooth_lane]
        if len(lane_indices) < 3:
            continue

        for _pass in range(5):
            changed = 0
            streaks = []
            s_start = 0
            for j in range(1, len(lane_indices)):
                if hits[lane_indices[j]].is_cymbal != hits[lane_indices[s_start]].is_cymbal:
                    streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, j - 1))
                    s_start = j
            streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, len(lane_indices) - 1))

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
                if prev_len <= streak_len and next_len <= streak_len:
                    continue
                target_cymbal = prev_is_cym
                for j in range(j_start, j_end + 1):
                    idx = lane_indices[j]
                    if hits[idx].is_cymbal != target_cymbal:
                        centroid = centroid_by_time.get(hits[idx].time_ms, 5000.0)
                        flips.append({
                            'time_ms': hits[idx].time_ms,
                            'lane': smooth_lane,
                            'from': 'tom' if not hits[idx].is_cymbal else 'cymbal',
                            'to': 'cymbal' if target_cymbal else 'tom',
                            'centroid': centroid,
                            'streak_len': streak_len,
                            'prev_len': prev_len,
                            'next_len': next_len,
                        })
                        hits[idx].is_cymbal = target_cymbal
                        changed += 1
            if changed == 0:
                break

    return initial_hits, resolved_hits, flips


def find_rapid_patterns(hits, max_gap_ms=120):
    """Find rapid cross-lane patterns (fills/rolls) on shared lanes."""
    shared = [h for h in hits if h['lane'] in (2, 3, 4)]
    shared.sort(key=lambda h: h['time_ms'])

    patterns = []
    current = [shared[0]] if shared else []
    for h in shared[1:]:
        if h['time_ms'] - current[-1]['time_ms'] <= max_gap_ms:
            current.append(h)
        else:
            if len(current) >= 3:
                patterns.append(current)
            current = [h]
    if len(current) >= 3:
        patterns.append(current)

    return patterns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("song_path", type=Path, help="Path to audio file or song directory")
    parser.add_argument("--onset-threshold", type=float, default=0.30)
    args = parser.parse_args()

    from scripts.batch_infer_hybrid import AUDIO_EXTENSIONS
    if args.song_path.is_file():
        audio_path = args.song_path
    else:
        audio_path = None
        for ext in AUDIO_EXTENSIONS:
            candidates = list(args.song_path.glob(f"*{ext}"))
            if candidates:
                audio_path = candidates[0]
                break
    if audio_path is None:
        logger.error(f"No audio file found at {args.song_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading models on {device}...")
    onset_model = load_v14_onset_detector(device)
    ensemble = load_ensemble(device)

    logger.info(f"Detecting onsets from {audio_path.name}...")
    onset_times_ms = detect_onsets_v14(
        onset_model, audio_path, device,
        onset_threshold=args.onset_threshold,
    )

    if not onset_times_ms:
        logger.error("No onsets detected!")
        return

    logger.info(f"  {len(onset_times_ms)} onsets detected")

    # Extract windows and classify
    any_needs_cqt = any(e["needs_cqt"] for e in ensemble)
    windows = extract_onset_windows(audio_path, onset_times_ms, needs_cqt=any_needs_cqt)

    context = build_context_vectors(onset_times_ms)
    logits_pass1 = classify_onsets_ensemble(ensemble, windows, context, device)
    probs_pass1 = 1.0 / (1.0 + np.exp(-logits_pass1))

    context = build_context_vectors(onset_times_ms, probs_pass1)
    logits = classify_onsets_ensemble(ensemble, windows, context, device)
    class_probs = 1.0 / (1.0 + np.exp(-logits))
    valid_mask = windows["valid_mask"]

    logger.info(f"Computing spectral features for {len(onset_times_ms)} onsets...")
    centroids, high_pcts = _compute_spectral_centroid_features(audio_path, onset_times_ms)

    thresholds = list(DEFAULT_CLASS_THRESHOLDS)

    logger.info("Tracing tom/cymbal decisions...")
    initial, resolved, flips = trace_build_chart(
        onset_times_ms, class_probs, valid_mask,
        centroids, high_pcts, thresholds,
    )

    # Report: Focus on shared lanes
    shared_initial = [h for h in initial if h['lane'] in (2, 3, 4)]
    shared_resolved = [h for h in resolved if h['lane'] in (2, 3, 4)]

    toms_initial = [h for h in shared_initial if not h['is_cymbal']]
    toms_resolved = [h for h in shared_resolved if not h['is_cymbal']]
    cyms_initial = [h for h in shared_initial if h['is_cymbal']]
    cyms_resolved = [h for h in shared_resolved if h['is_cymbal']]

    logger.info("\n" + "=" * 70)
    logger.info("TOM/CYMBAL DECISION TRACE")
    logger.info("=" * 70)
    logger.info(f"\nShared lanes (2/3/4): {len(shared_initial)} initial hits")
    logger.info(f"  Stage 1 (initial):  {len(toms_initial)} toms, {len(cyms_initial)} cymbals")
    logger.info(f"  Stage 2 (resolved): {len(toms_resolved)} toms, {len(cyms_resolved)} cymbals")
    logger.info(f"  Stage 3 (streak):   {len(flips)} flips")

    # Streak flip details
    tom_to_cym = [f for f in flips if f['from'] == 'tom']
    cym_to_tom = [f for f in flips if f['from'] == 'cymbal']
    logger.info(f"\n  Tom → Cymbal flips: {len(tom_to_cym)}")
    logger.info(f"  Cymbal → Tom flips: {len(cym_to_tom)}")

    if tom_to_cym:
        logger.info("\n--- TOM → CYMBAL FLIPS (potential bad swaps) ---")
        lane_names = {2: 'Yellow(HiHat/HiTom)', 3: 'Blue(Ride/LowTom)', 4: 'Green(Crash/FloorTom)'}
        for f in tom_to_cym:
            protected = "WOULD BE PROTECTED" if f['centroid'] < 3000 else "spectral allows flip"
            logger.info(
                f"  {f['time_ms']:8.1f}ms  lane={f['lane']} ({lane_names[f['lane']]})  "
                f"centroid={f['centroid']:.0f}Hz  streak={f['streak_len']}  "
                f"neighbors={f['prev_len']}/{f['next_len']}  [{protected}]"
            )

    # Find rapid cross-lane patterns (fills/rolls) in resolved hits
    logger.info("\n--- RAPID CROSS-LANE PATTERNS (fills/rolls) ---")
    patterns = find_rapid_patterns(shared_resolved, max_gap_ms=120)
    for pi, pattern in enumerate(patterns):
        lanes_used = set(h['lane'] for h in pattern)
        if len(lanes_used) < 2:
            continue  # Single lane patterns less interesting for cross-lane swaps
        has_tom = any(not h['is_cymbal'] for h in pattern)
        has_cym = any(h['is_cymbal'] for h in pattern)
        if not (has_tom and has_cym):
            continue  # Pure tom or pure cymbal patterns — no swaps
        t_start = pattern[0]['time_ms']
        t_end = pattern[-1]['time_ms']
        logger.info(f"\n  Fill #{pi+1}: {t_start:.0f}-{t_end:.0f}ms ({len(pattern)} hits, {len(lanes_used)} lanes)")
        for h in pattern:
            label = "CYM" if h['is_cymbal'] else "TOM"
            logger.info(
                f"    {h['time_ms']:8.1f}ms  lane={h['lane']}  {label:3s}  "
                f"{h['cls_name']:10s}  prob={h['prob']:.3f}  centroid={h['centroid']:.0f}Hz"
            )
        # Check for flips in this time range
        range_flips = [f for f in flips if t_start <= f['time_ms'] <= t_end]
        if range_flips:
            logger.info(f"    >> {len(range_flips)} streak flips in this fill!")
            for f in range_flips:
                logger.info(f"       {f['time_ms']:.1f}ms: {f['from']} → {f['to']} (centroid={f['centroid']:.0f}Hz)")

    # Summary: centroid distribution of tom→cymbal flips
    if tom_to_cym:
        centroids_of_flips = [f['centroid'] for f in tom_to_cym]
        logger.info(f"\n--- TOM→CYMBAL FLIP CENTROID STATS ---")
        logger.info(f"  Mean:   {np.mean(centroids_of_flips):.0f} Hz")
        logger.info(f"  Median: {np.median(centroids_of_flips):.0f} Hz")
        logger.info(f"  Min:    {np.min(centroids_of_flips):.0f} Hz")
        logger.info(f"  Max:    {np.max(centroids_of_flips):.0f} Hz")
        protectable = sum(1 for c in centroids_of_flips if c < 3000)
        logger.info(f"  Would protect (centroid < 3000): {protectable}/{len(centroids_of_flips)} ({100*protectable/len(centroids_of_flips):.0f}%)")


if __name__ == "__main__":
    main()
