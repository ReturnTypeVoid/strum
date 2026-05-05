#!/usr/bin/env python3
"""
Post-processing for generated drum charts.

Applies musical intelligence to clean up ML predictions:
  1. Rhythmic quantization — snap hits to the nearest beat subdivision
  2. Hi-hat/ride pattern completion — fill gaps in steady patterns
  3. Kick/snare backbeat reinforcement — fill obvious gaps in backbone
  4. Minimum gap enforcement — remove impossibly fast repeated hits
  5. Ghost note cleanup — remove isolated low-confidence hits
"""
import logging
import os
from dataclasses import dataclass, replace
from collections import defaultdict
import bisect

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.parsers.midi_parser import DrumChart, DrumHit, TempoEvent

logger = logging.getLogger(__name__)

# ── Env toggles for ablation (1=on, 0=off) ──
# Defaults updated 2026-04 after RB ablation: backbeat, ghost-cleanup, and
# snare-roll fill were all net-harmful (+31 FPs, -6 correct combined).
# Stage 2 of clean_isolated_lane_flips ("noisy zone" resolution) flips toms
# in fills toward surrounding cymbal context — turn off by default.
PP_LANE_FLIPS   = os.environ.get("STRUM_PP_LANE_FLIPS", "0") == "1"
# complete_cymbal_patterns lays a uniform grid over rhythmic patterns and
# loses the original feel — turn off by default.
PP_PATTERN_FILL = os.environ.get("STRUM_PP_PATTERN", "0") == "1"
PP_BACKBEAT     = os.environ.get("STRUM_PP_BACKBEAT", "0") == "1"
PP_SNARE_ROLL   = os.environ.get("STRUM_PP_SNARE_ROLL", "0") == "1"
PP_GHOST_KILL   = os.environ.get("STRUM_PP_GHOST", "0") == "1"


# ══════════════════════════════════════════════════════════════
# 1. Rhythmic quantization
# ══════════════════════════════════════════════════════════════

def _compute_phase_offset(times_ms: list[float], grid_ms: float) -> float:
    """Find the grid phase offset that best aligns with the given hit times.

    Uses circular statistics: treat each hit's fractional position in the
    grid as an angle on a circle, then find the mean angle. This is robust
    to hits on different beats/bars — only the within-grid-cell position
    matters.

    Args:
        times_ms: Onset times in milliseconds.
        grid_ms: Grid spacing in milliseconds.

    Returns:
        Phase offset in ms (0 ≤ offset < grid_ms).
    """
    if not times_ms:
        return 0.0
    phases = np.array([(t % grid_ms) / grid_ms * 2 * np.pi for t in times_ms])
    mean_angle = np.arctan2(np.mean(np.sin(phases)), np.mean(np.cos(phases)))
    if mean_angle < 0:
        mean_angle += 2 * np.pi
    offset = mean_angle / (2 * np.pi) * grid_ms
    return offset


def quantize_hits(chart: DrumChart, max_subdivision: int = 16,
                  phase_offset_ms: float | None = None) -> DrumChart:
    """Snap hit times to the nearest beat subdivision.

    Args:
        chart: Input drum chart
        max_subdivision: Maximum subdivision to quantize to (8=8th notes,
            16=16th notes, 32=32nd notes). Higher = less quantization.
        phase_offset_ms: Grid phase offset. If None, auto-computed from
            hit times using circular statistics.

    Returns:
        New DrumChart with quantized hit times.
    """
    if not chart.hits or not chart.tempo_events:
        return chart

    tempo_bpm = chart.tempo_events[0].tempo_bpm
    ms_per_beat = 60_000.0 / tempo_bpm
    # Grid spacing: one subdivision
    grid_ms = ms_per_beat / (max_subdivision / 4)

    # Compute phase offset if not provided
    if phase_offset_ms is None:
        all_times = [h.time_ms for h in chart.hits]
        phase_offset_ms = _compute_phase_offset(all_times, grid_ms)
    logger.info(f"  Grid phase offset: {phase_offset_ms:.1f}ms "
                f"({phase_offset_ms/grid_ms:.3f} of grid)")

    # Pre-compute hand-hit timeline (lane >= 1) for cross-lane fast-roll
    # detection. When a fast roll like LT→LT→FT crosses lanes within a
    # single grid cell, naive quantization stacks two of the hits on the
    # same tick (different lanes), creating a chord. We leave such hits
    # un-quantized to preserve the roll's micro-timing.
    hand_times_sorted = sorted(h.time_ms for h in chart.hits if h.lane >= 1)
    import bisect as _bisect
    roll_neighbor_ms = grid_ms * 1.1  # slightly more than one grid cell

    def _has_close_neighbor(t: float) -> bool:
        if not hand_times_sorted:
            return False
        pos = _bisect.bisect_left(hand_times_sorted, t)
        if pos > 0 and 0.5 < (t - hand_times_sorted[pos - 1]) <= roll_neighbor_ms:
            return True
        if pos < len(hand_times_sorted) - 1 \
                and 0.5 < (hand_times_sorted[pos + 1] - t) <= roll_neighbor_ms:
            return True
        return False

    new_hits = []
    skipped_roll = 0
    for hit in chart.hits:
        # Find nearest grid point, accounting for phase offset
        grid_pos = round((hit.time_ms - phase_offset_ms) / grid_ms)
        snapped_ms = phase_offset_ms + grid_pos * grid_ms

        # Only snap if within a reasonable window (half a grid cell)
        if abs(hit.time_ms - snapped_ms) <= grid_ms / 2:
            # Cross-lane fast-roll guard: if this is a hand hit with a
            # very close neighbor (within ~1 grid cell) AND snapping it
            # would change its time by more than 1/4 of the grid, leave
            # the original timing so the roll doesn't collapse into a
            # chord with the neighbor.
            if hit.lane >= 1 and _has_close_neighbor(hit.time_ms) \
                    and abs(hit.time_ms - snapped_ms) > grid_ms / 4:
                new_hits.append(hit)
                skipped_roll += 1
            else:
                new_hits.append(replace(hit, time_ms=snapped_ms))
        else:
            new_hits.append(hit)

    if skipped_roll > 0:
        logger.info(f"  Roll-protect: {skipped_roll} hits left un-quantized "
                    f"(close cross-lane neighbor)")

    # Dedup after quantization: if two hits landed on the same grid point
    # for the same lane, keep only one. When a cymbal and tom land on the
    # same tick (from two nearby onsets), keep the cymbal — it's the
    # dominant class and the tom is likely a false positive from a
    # neighboring onset bleeding into the grid cell.
    new_hits.sort(key=lambda h: (h.time_ms, h.lane, not h.is_cymbal))
    deduped = []
    for hit in new_hits:
        if deduped and abs(hit.time_ms - deduped[-1].time_ms) < 1.0 \
                and hit.lane == deduped[-1].lane:
            continue
        deduped.append(hit)

    logger.info(f"  Quantization: {len(chart.hits)} → {len(deduped)} hits "
                f"(grid={grid_ms:.1f}ms, {max_subdivision}th notes)")

    return replace(chart, hits=deduped)


# ══════════════════════════════════════════════════════════════
# 1b. Run regularization (fix hop-size jitter after quantization)
# ══════════════════════════════════════════════════════════════

def regularize_runs(chart: DrumChart, max_subdivision: int = 16,
                    min_run_length: int = 4,
                    phase_offset_ms: float = 0.0) -> DrumChart:
    """Fix uneven spacing in fast same-lane runs caused by detector jitter.

    The V14 onset detector has HOP=512 (11.6ms frame) resolution. When the
    true inter-onset interval falls between two frame durations (e.g. 89.8ms
    is 7.74 frames), detections alternate between adjacent frame counts,
    creating ±1 grid jitter after quantization: e.g. gaps of 2,3,3,4,3,2
    grid units instead of all 3's.

    This function detects runs of ≥min_run_length consecutive same-lane hits
    where gaps deviate by at most ±1 grid unit from the dominant gap, and
    re-spaces them evenly.

    Args:
        chart: Quantized drum chart.
        max_subdivision: The subdivision used during quantization.
        min_run_length: Minimum number of consecutive hits to regularize.

    Returns:
        New DrumChart with regularized run spacing.
    """
    if not chart.hits or not chart.tempo_events:
        return chart

    tempo_bpm = chart.tempo_events[0].tempo_bpm
    ms_per_beat = 60_000.0 / tempo_bpm
    grid_ms = ms_per_beat / (max_subdivision / 4)

    # Group hits by lane key (lane, is_cymbal)
    from collections import Counter
    hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)

    # Build per-lane sorted lists with original indices
    lane_hits: dict[tuple[int, bool], list[tuple[int, DrumHit]]] = defaultdict(list)
    for idx, hit in enumerate(hits_sorted):
        lane_hits[(hit.lane, hit.is_cymbal)].append((idx, hit))

    # Collect all time adjustments: idx -> new_time_ms
    adjustments: dict[int, float] = {}
    total_regularized = 0

    for lane_key, lane_list in lane_hits.items():
        if len(lane_list) < min_run_length:
            continue

        # Find runs: consecutive hits on this lane where no other hit on the
        # same lane interrupts. We define "consecutive" as gaps small enough
        # to be part of a fast run (≤ 8 grid units, i.e. ≤ 2 beats).
        max_run_gap_grids = 8

        runs: list[list[tuple[int, DrumHit]]] = []
        current_run: list[tuple[int, DrumHit]] = [lane_list[0]]

        for i in range(1, len(lane_list)):
            prev_t = lane_list[i - 1][1].time_ms
            curr_t = lane_list[i][1].time_ms
            gap_grids = round((curr_t - prev_t) / grid_ms)
            if gap_grids <= max_run_gap_grids:
                current_run.append(lane_list[i])
            else:
                if len(current_run) >= min_run_length:
                    runs.append(current_run)
                current_run = [lane_list[i]]
        if len(current_run) >= min_run_length:
            runs.append(current_run)

        for run in runs:
            # Compute gaps in grid units
            times = [h.time_ms for _, h in run]
            gaps = [round((times[i] - times[i - 1]) / grid_ms)
                    for i in range(1, len(times))]
            if not gaps:
                continue

            # Find dominant gap
            gap_counts = Counter(gaps)
            dominant_gap = gap_counts.most_common(1)[0][0]
            if dominant_gap < 1:
                continue

            # Check all gaps are within ±1 of dominant
            if not all(abs(g - dominant_gap) <= 1 for g in gaps):
                continue

            # Check that the jitter is actually present (not already regular)
            if all(g == dominant_gap for g in gaps):
                continue

            # Re-space: keep first hit, space remaining at dominant_gap
            first_time = times[0]
            for j, (idx, hit) in enumerate(run):
                new_time = first_time + j * dominant_gap * grid_ms
                adjustments[idx] = new_time
            total_regularized += len(run)

    if not adjustments:
        return chart

    # Apply adjustments
    new_hits = []
    for idx, hit in enumerate(hits_sorted):
        if idx in adjustments:
            new_hits.append(replace(hit, time_ms=adjustments[idx]))
        else:
            new_hits.append(hit)

    # Re-dedup after adjustment (same as quantize_hits)
    new_hits.sort(key=lambda h: (h.time_ms, h.lane, not h.is_cymbal))
    deduped = []
    for hit in new_hits:
        if deduped and abs(hit.time_ms - deduped[-1].time_ms) < 1.0 \
                and hit.lane == deduped[-1].lane:
            continue
        deduped.append(hit)

    removed = len(new_hits) - len(deduped)
    logger.info(f"  Run regularization: {total_regularized} hits in runs re-spaced"
                f" (dominant gap preserved, {removed} dupes removed)")

    return replace(chart, hits=deduped)


def coarsen_non_runs(chart: DrumChart, fine_subdivision: int = 96,
                     coarse_subdivision: int = 32,
                     run_gap_threshold: int = 4,
                     phase_offset_ms: float = 0.0) -> DrumChart:
    """Re-snap isolated notes to a coarser grid to eliminate jitter.

    The fine adaptive grid (e.g. 96th) preserves fast-roll timing, but for
    normal playing (8ths, 16ths, quarters) it's too fine: a ~30ms detector
    jitter snaps notes to positions one 96th-note off from the correct 32nd
    position, producing beat positions like 0.417 instead of 0.5.

    This function identifies notes that are NOT part of fast runs (gap to
    nearest same-lane neighbor > run_gap_threshold fine-grid steps) and
    re-snaps them to the coarser 32nd-note grid.

    Args:
        chart: Drum chart (already fine-quantized and regularized).
        fine_subdivision: The fine subdivision used for initial quantization.
        coarse_subdivision: Target coarser grid for non-run notes.
        run_gap_threshold: Max gap in fine-grid units to consider part of a run.

    Returns:
        New DrumChart with non-run notes snapped to the coarse grid.
    """
    if not chart.hits or not chart.tempo_events:
        return chart

    tempo_bpm = chart.tempo_events[0].tempo_bpm
    ms_per_beat = 60_000.0 / tempo_bpm
    fine_grid_ms = ms_per_beat / (fine_subdivision / 4)
    coarse_grid_ms = ms_per_beat / (coarse_subdivision / 4)

    # If grids are the same, nothing to do
    if fine_subdivision <= coarse_subdivision:
        return chart

    hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)

    # Build per-lane sorted time lists
    lane_times: dict[tuple[int, bool], list[float]] = defaultdict(list)
    for hit in hits_sorted:
        lane_times[(hit.lane, hit.is_cymbal)].append(hit.time_ms)

    # Determine which hits are in fast runs
    in_run: set[int] = set()
    for idx, hit in enumerate(hits_sorted):
        key = (hit.lane, hit.is_cymbal)
        times = lane_times[key]
        # Binary search for neighbors within run_gap_threshold
        import bisect
        pos = bisect.bisect_left(times, hit.time_ms)
        threshold_ms = run_gap_threshold * fine_grid_ms
        has_close_neighbor = False
        # Check left neighbor
        if pos > 0 and (hit.time_ms - times[pos - 1]) <= threshold_ms:
            has_close_neighbor = True
        # Check right neighbor
        if pos < len(times) - 1 and (times[pos + 1] - hit.time_ms) <= threshold_ms:
            has_close_neighbor = True
        if has_close_neighbor:
            in_run.add(idx)

    # Re-snap non-run notes to coarse grid
    coarsened = 0
    new_hits = []
    for idx, hit in enumerate(hits_sorted):
        if idx in in_run:
            new_hits.append(hit)
        else:
            coarse_pos = round((hit.time_ms - phase_offset_ms) / coarse_grid_ms)
            snapped = phase_offset_ms + coarse_pos * coarse_grid_ms
            if abs(snapped - hit.time_ms) < coarse_grid_ms / 2:
                if abs(snapped - hit.time_ms) > 0.5:  # actually moved
                    coarsened += 1
                new_hits.append(replace(hit, time_ms=snapped))
            else:
                new_hits.append(hit)

    # Dedup
    new_hits.sort(key=lambda h: (h.time_ms, h.lane, not h.is_cymbal))
    deduped = []
    for hit in new_hits:
        if deduped and abs(hit.time_ms - deduped[-1].time_ms) < 1.0 \
                and hit.lane == deduped[-1].lane:
            continue
        deduped.append(hit)

    removed = len(new_hits) - len(deduped)
    logger.info(f"  Coarsen non-runs: {coarsened} notes re-snapped to {coarse_subdivision}th grid"
                f" ({len(in_run)} in-run preserved, {removed} dupes removed)")

    return replace(chart, hits=deduped)


# ══════════════════════════════════════════════════════════════
# 1b. Cross-lane simultaneous alignment
# ══════════════════════════════════════════════════════════════

def align_simultaneous_hits(chart: DrumChart, max_subdivision: int = 96,
                            coarse_subdivision: int = 32,
                            phase_offset_ms: float = 0.0) -> DrumChart:
    """Snap notes on different lanes to the same tick when they're nearly simultaneous.

    When a drummer plays kick+snare together, the onset detector may fire
    two separate onsets a few ms apart.  After quantization these land on
    adjacent grid points (e.g. 20 ticks apart in MIDI), looking wrong.

    This groups all hits within half a coarse-grid cell and snaps the whole
    group to the single grid point that the majority already occupy.
    Only merges across lanes — same-lane near-duplicates are handled by dedup.

    Args:
        chart: Quantized drum chart.
        max_subdivision: Fine subdivision used during quantization.
        coarse_subdivision: Coarse subdivision (32nd notes) for non-run notes.
        phase_offset_ms: Grid phase offset for re-snapping.

    Returns:
        New DrumChart with cross-lane simultaneous hits aligned.
    """
    if not chart.hits or not chart.tempo_events:
        return chart

    tempo_bpm = chart.tempo_events[0].tempo_bpm
    ms_per_beat = 60_000.0 / tempo_bpm
    coarse_grid_ms = ms_per_beat / (coarse_subdivision / 4)
    # Merge window: half a coarse grid cell.  After coarsen_non_runs,
    # non-run notes sit on 32nd-note grid points.  Two notes on adjacent
    # grid points (different lanes) that should be simultaneous need this
    # wider window to get grouped together.
    merge_window_ms = coarse_grid_ms * 0.55

    # Fine grid for run detection
    fine_grid_ms = ms_per_beat / (max_subdivision / 4)

    hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)

    # Build per-lane sorted time lists to detect runs
    import bisect
    lane_times: dict[int, list[float]] = defaultdict(list)
    for hit in hits_sorted:
        lane_times[hit.lane].append(hit.time_ms)

    # Cross-lane hand timeline (lane >= 1) — for fast-roll detection that
    # spans multiple lanes (e.g. LowTom→LowTom→FloorTom). Without this,
    # the FloorTom (lone on its lane) gets merged with the nearest LowTom
    # into a chord, collapsing the roll.
    hand_times = sorted(h.time_ms for h in hits_sorted if h.lane >= 1)
    cross_lane_run_ms = 110.0  # ~16th note at 130 BPM

    def _has_close_hand_neighbor(t: float) -> bool:
        if not hand_times:
            return False
        pos = bisect.bisect_left(hand_times, t)
        if pos > 0 and 0.5 < (t - hand_times[pos - 1]) <= cross_lane_run_ms:
            return True
        if pos < len(hand_times) - 1 \
                and 0.5 < (hand_times[pos + 1] - t) <= cross_lane_run_ms:
            return True
        return False

    def _is_in_run(hit_time: float, lane: int, threshold_ms: float = None) -> bool:
        """True if this hit has close same-lane neighbors (part of a fast run)."""
        if threshold_ms is None:
            threshold_ms = fine_grid_ms * 4  # ~4 fine-grid steps
        times = lane_times[lane]
        pos = bisect.bisect_left(times, hit_time)
        # Check left neighbor
        if pos > 0 and (hit_time - times[pos - 1]) <= threshold_ms:
            return True
        # Check right neighbor
        if pos < len(times) - 1 and (times[pos + 1] - hit_time) <= threshold_ms:
            return True
        return False

    # Build groups of hits within merge_window_ms of each other
    groups: list[list[int]] = []  # list of index lists
    current_group: list[int] = [0]

    for i in range(1, len(hits_sorted)):
        if hits_sorted[i].time_ms - hits_sorted[current_group[0]].time_ms <= merge_window_ms:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    groups.append(current_group)

    aligned = 0
    new_hits = list(hits_sorted)

    for group in groups:
        if len(group) < 2:
            continue

        # Check if this group has hits on multiple lanes
        lanes = set(hits_sorted[i].lane for i in group)
        if len(lanes) < 2:
            continue

        # Safety: don't merge notes in fast runs on the same lane.
        # If multiple hits in the group share a lane, skip — they're
        # separate intentional notes, not a simultaneous pair.
        from collections import Counter
        lane_counts = Counter(hits_sorted[i].lane for i in group)
        if any(c > 1 for c in lane_counts.values()):
            continue

        # Run protection: if ALL notes in the group are in fast runs on
        # their own lanes, this is likely a dense section where different
        # lanes have interleaved notes (e.g. hihat 16ths + snare 8ths).
        # Only merge when at least one note is isolated on its lane.
        run_flags = [
            _is_in_run(hits_sorted[i].time_ms, hits_sorted[i].lane)
            for i in group
        ]
        if all(run_flags):
            continue

        # Cross-lane fast-roll protection: if any hit in this group has a
        # close hand-lane neighbor OUTSIDE the group (e.g. LowTom→LowTom→
        # FloorTom roll), do NOT merge. Otherwise the lone-lane FloorTom
        # would be collapsed into a chord with one of the LowToms,
        # turning a 3-note tom roll into "tom + (tom+floortom)".
        # Restrict this check to hand lanes — kick (lane 0) often pairs
        # with snare/cymbal on strong beats and SHOULD be merged.
        group_times = [hits_sorted[i].time_ms for i in group]
        group_lo, group_hi = min(group_times), max(group_times)
        in_roll = False
        for i in group:
            h = hits_sorted[i]
            if h.lane < 1:
                continue
            t = h.time_ms
            pos = bisect.bisect_left(hand_times, t)
            # Look outside the group window for a close neighbor
            for nb_idx in (pos - 1, pos + 1):
                if 0 <= nb_idx < len(hand_times):
                    nb_t = hand_times[nb_idx]
                    if group_lo - 0.5 <= nb_t <= group_hi + 0.5:
                        continue  # neighbor is inside the group
                    if 0.5 < abs(t - nb_t) <= cross_lane_run_ms:
                        in_roll = True
                        break
            if in_roll:
                break
        if in_roll:
            continue

        # Find the target time to snap to.  Prioritize notes that are
        # in runs — they sit on a well-defined grid and shouldn't be
        # moved.  Isolated notes snap TO run notes, not vice versa.
        run_times = [round(hits_sorted[group[j]].time_ms, 2)
                     for j in range(len(group)) if run_flags[j]]
        if run_times:
            time_counts = Counter(run_times)
            target_time = time_counts.most_common(1)[0][0]
        else:
            # No run notes — use mode of all times
            time_counts = Counter(round(hits_sorted[i].time_ms, 2) for i in group)
            target_time = time_counts.most_common(1)[0][0]

        # Snap all hits in the group to the target time
        for i in group:
            if abs(hits_sorted[i].time_ms - target_time) > 0.5:
                new_hits[i] = replace(hits_sorted[i], time_ms=target_time)
                aligned += 1

    # Dedup after alignment (same lane, same time)
    new_hits.sort(key=lambda h: (h.time_ms, h.lane, not h.is_cymbal))
    deduped = []
    for hit in new_hits:
        if deduped and abs(hit.time_ms - deduped[-1].time_ms) < 1.0 \
                and hit.lane == deduped[-1].lane:
            continue
        deduped.append(hit)

    removed = len(new_hits) - len(deduped)
    if aligned > 0:
        logger.info(f"  Cross-lane alignment: {aligned} hits snapped to simultaneous"
                    f" ({removed} dupes removed)")

    return replace(chart, hits=deduped)


# ══════════════════════════════════════════════════════════════
# 2. Minimum gap enforcement
# ══════════════════════════════════════════════════════════════

def enforce_min_gap(chart: DrumChart, min_gap_ms: float = 30.0) -> DrumChart:
    """Remove impossibly fast repeated hits on the same lane.

    A drummer physically can't hit the same drum faster than ~30ms apart.
    When two hits on the same lane are closer than min_gap_ms, keep the first.
    """
    if not chart.hits:
        return chart

    hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)

    # Track last hit time per (lane, is_cymbal)
    last_time: dict[tuple[int, bool], float] = {}
    kept = []

    for hit in hits_sorted:
        key = (hit.lane, hit.is_cymbal)
        prev = last_time.get(key)
        if prev is not None and (hit.time_ms - prev) < min_gap_ms:
            continue
        kept.append(hit)
        last_time[key] = hit.time_ms

    removed = len(chart.hits) - len(kept)
    if removed > 0:
        logger.info(f"  Min gap: removed {removed} hits (<{min_gap_ms}ms apart)")

    return replace(chart, hits=kept)


# ══════════════════════════════════════════════════════════════
# 2c. Clean isolated tom/cymbal flips on shared lanes
# ══════════════════════════════════════════════════════════════

def clean_isolated_lane_flips(
    chart: DrumChart,
    max_isolated_len: int = 2,
    min_neighbor_len: int = 2,
) -> DrumChart:
    """Fix remaining isolated tom↔cymbal flips on shared lanes.

    After onset classification and streak smoothing, some isolated mis-classified
    notes survive because spectral / fill / density protections blocked the
    correction. A single tom note in a long cymbal run (or vice versa) is almost
    never a real instrument change — it's a classification error.

    Two-stage approach:
    1. Fix isolated short streaks (≤max_isolated_len) flanked by ≥min_neighbor_len
       runs of the opposite type on each side.
    2. Fix alternating clusters: rapid CTCTC or TCTCT patterns where the classifier
       oscillates. These are resolved by looking at the dominant type in a wider
       window surrounding the cluster.

    Args:
        chart: Drum chart (after streak smoothing and quantization).
        max_isolated_len: Maximum streak length to consider "isolated".
        min_neighbor_len: Minimum length of flanking runs to trigger correction.
    """
    if not chart.hits:
        return chart

    SHARED_LANES = [2, 3, 4]
    hits = list(chart.hits)

    # Pre-compute tom-density times for fill detection: skip flipping a tom
    # toward cymbal when the tom is part of a fill (≥3 toms within ±300ms).
    all_tom_times = sorted(
        h.time_ms for h in hits
        if h.lane in (2, 3, 4) and not h.is_cymbal
    )

    def _in_fill(t_ms: float) -> bool:
        if len(all_tom_times) < 3:
            return False
        lo = bisect.bisect_left(all_tom_times, t_ms - 300.0)
        hi = bisect.bisect_right(all_tom_times, t_ms + 300.0)
        return (hi - lo) >= 3

    total_fixed = 0
    tom_to_cym = 0
    cym_to_tom = 0

    for lane in SHARED_LANES:
        # Get indices of hits on this lane, sorted by time
        lane_indices = [i for i, h in enumerate(hits) if h.lane == lane]
        if len(lane_indices) < 5:
            continue

        # ── Stage 1: Fix isolated streaks (iterative) ──
        for _pass in range(3):
            changed = 0

            streaks: list[tuple[bool, int, int]] = []
            s_start = 0
            for j in range(1, len(lane_indices)):
                if hits[lane_indices[j]].is_cymbal != hits[lane_indices[s_start]].is_cymbal:
                    streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, j - 1))
                    s_start = j
            streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, len(lane_indices) - 1))

            for si, (is_cym, j_start, j_end) in enumerate(streaks):
                streak_len = j_end - j_start + 1
                if streak_len > max_isolated_len:
                    continue
                if si == 0 or si == len(streaks) - 1:
                    continue

                prev_is_cym, _, prev_end = streaks[si - 1]
                next_is_cym, next_start, next_end = streaks[si + 1]
                prev_len = prev_end - streaks[si - 1][1] + 1
                next_len = next_end - next_start + 1

                if prev_is_cym == is_cym or next_is_cym == is_cym:
                    continue

                if prev_len < min_neighbor_len or next_len < min_neighbor_len:
                    continue

                target_cymbal = prev_is_cym

                # Skip if this would convert toms-in-fill to cymbals.
                # Stage 1 fires aggressively when a single tom (or pair of toms)
                # appears inside a cymbal groove; a fill of a few toms surrounded
                # by hi-hats matches this pattern but should NOT be flipped.
                if target_cymbal and not is_cym:
                    streak_t0 = hits[lane_indices[j_start]].time_ms
                    streak_t1 = hits[lane_indices[j_end]].time_ms
                    mid_t = (streak_t0 + streak_t1) * 0.5
                    if _in_fill(mid_t):
                        continue

                for j in range(j_start, j_end + 1):
                    idx = lane_indices[j]
                    hits[idx] = replace(hits[idx], is_cymbal=target_cymbal)
                    total_fixed += 1
                    changed += 1
                    if target_cymbal:
                        tom_to_cym += 1
                    else:
                        cym_to_tom += 1

            if changed == 0:
                break

        # ── Stage 2: Fix alternating clusters ──
        # Pattern like ...CCC T C T CCC... where classifier oscillates.
        # Find "noisy zones" = regions with ≥2 type changes within a short span,
        # then resolve by dominant type of the surrounding stable sections.
        streaks = []
        s_start = 0
        for j in range(1, len(lane_indices)):
            if hits[lane_indices[j]].is_cymbal != hits[lane_indices[s_start]].is_cymbal:
                streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, j - 1))
                s_start = j
        streaks.append((hits[lane_indices[s_start]].is_cymbal, s_start, len(lane_indices) - 1))

        if len(streaks) < 4:
            continue

        # Find noisy zones: consecutive short streaks (all ≤2 notes)
        # A noisy zone is 3+ consecutive streaks that are all short.
        i = 0
        while i < len(streaks):
            slen = streaks[i][2] - streaks[i][1] + 1
            if slen > max_isolated_len:
                i += 1
                continue

            # Found a short streak — extend to find full noisy zone
            zone_start = i
            zone_end = i
            while zone_end + 1 < len(streaks):
                next_slen = streaks[zone_end + 1][2] - streaks[zone_end + 1][1] + 1
                if next_slen <= max_isolated_len:
                    zone_end += 1
                else:
                    break

            n_streaks = zone_end - zone_start + 1
            if n_streaks >= 3:
                # Skip if this looks like a real fill (tom-dominated zone),
                # not a classifier oscillation. A fill that ends with a crash
                # is the canonical false-positive: the noisy zone is mostly
                # toms with a single trailing cymbal, but the surrounding
                # context is the cymbal groove that follows.
                zone_tom_count = 0
                zone_cym_count = 0
                for si2 in range(zone_start, zone_end + 1):
                    is_cym_z, j_s, j_e = streaks[si2]
                    n_in_streak = j_e - j_s + 1
                    if is_cym_z:
                        zone_cym_count += n_in_streak
                    else:
                        zone_tom_count += n_in_streak
                # If toms are the majority, leave the fill alone
                if zone_tom_count > zone_cym_count:
                    i = zone_end + 1
                    continue

                # This is a noisy zone — resolve from surrounding context
                # Look at the stable streaks before and after the zone
                before_cym = None
                before_len = 0
                if zone_start > 0:
                    before_cym = streaks[zone_start - 1][0]
                    before_len = streaks[zone_start - 1][2] - streaks[zone_start - 1][1] + 1

                after_cym = None
                after_len = 0
                if zone_end + 1 < len(streaks):
                    after_cym = streaks[zone_end + 1][0]
                    after_len = streaks[zone_end + 1][2] - streaks[zone_end + 1][1] + 1

                # Determine target type from surrounding stable sections
                target = None
                if before_cym is not None and after_cym is not None:
                    if before_cym == after_cym:
                        # Both sides agree — resolve to that type
                        target = before_cym
                    else:
                        # Sides disagree — use the longer side
                        target = before_cym if before_len >= after_len else after_cym
                elif before_cym is not None:
                    target = before_cym
                elif after_cym is not None:
                    target = after_cym

                if target is not None:
                    # Flip all notes in the noisy zone to the target type
                    for si2 in range(zone_start, zone_end + 1):
                        _, j_s, j_e = streaks[si2]
                        for j in range(j_s, j_e + 1):
                            idx = lane_indices[j]
                            if hits[idx].is_cymbal != target:
                                hits[idx] = replace(hits[idx], is_cymbal=target)
                                total_fixed += 1
                                if target:
                                    tom_to_cym += 1
                                else:
                                    cym_to_tom += 1

            i = zone_end + 1

    if total_fixed > 0:
        logger.info(f"  Clean lane flips: {total_fixed} fixed "
                    f"({tom_to_cym} tom→cymbal, {cym_to_tom} cymbal→tom)")

    return replace(chart, hits=hits)


# ══════════════════════════════════════════════════════════════
# 2b. Tom-fill cymbal protection
# ══════════════════════════════════════════════════════════════

def protect_tom_fills(
    chart: DrumChart,
    fill_window_ms: float = 250.0,
    min_toms_for_fill: int = 3,
    keep_last_cymbal_within_ms: float = 80.0,
) -> DrumChart:
    """Strip phantom cymbals from inside tom-roll fills.

    Multiple upstream stages (build_chart cymbal co-fire, phase3 co-occurrence
    rescue, complete_cymbal_patterns) inject cymbals at kick/snare positions.
    Inside a tom roll/fill those cymbals are almost always wrong — either
    classifier confusion (low tom misread as ride) or co-fire false positives.

    A 'fill region' = any tom (lane 2/3/4 + is_cymbal=False) with ≥
    `min_toms_for_fill - 1` other toms within ±fill_window_ms.

    Rule: inside fill regions, drop cymbals on Yellow/Blue/Green (lanes 2/3/4).
    Exception: keep a single CRASH at end-of-fill (cymbal within
    `keep_last_cymbal_within_ms` of the last tom in the fill cluster — drummers
    routinely punctuate fills with a crash). Snare/kick/hihat unaffected.
    """
    if not chart.hits:
        return chart

    tom_times = sorted(
        h.time_ms for h in chart.hits
        if h.lane in (2, 3, 4) and not h.is_cymbal
    )
    if len(tom_times) < min_toms_for_fill:
        return chart

    # Build fill-region intervals by clustering toms within fill_window_ms
    cluster_starts: list[float] = []
    cluster_ends: list[float] = []
    cur_start = tom_times[0]
    cur_end = tom_times[0]
    cur_count = 1
    for t in tom_times[1:]:
        if t - cur_end <= fill_window_ms:
            cur_end = t
            cur_count += 1
        else:
            if cur_count >= min_toms_for_fill:
                cluster_starts.append(cur_start)
                cluster_ends.append(cur_end)
            cur_start = t
            cur_end = t
            cur_count = 1
    if cur_count >= min_toms_for_fill:
        cluster_starts.append(cur_start)
        cluster_ends.append(cur_end)

    if not cluster_starts:
        return chart

    import bisect as _bisect

    def _cluster_idx(t_ms: float) -> int:
        """Return cluster index containing t_ms, or -1."""
        i = _bisect.bisect_right(cluster_starts, t_ms) - 1
        if i >= 0 and t_ms <= cluster_ends[i] + keep_last_cymbal_within_ms:
            return i
        return -1

    kept = []
    dropped = 0
    for hit in chart.hits:
        if hit.lane in (2, 3, 4) and hit.is_cymbal:
            ci = _cluster_idx(hit.time_ms)
            if ci >= 0:
                # End-of-fill crash exception: keep CRASH (lane 4) within
                # keep_last_cymbal_within_ms after cluster end
                end = cluster_ends[ci]
                if (hit.lane == 4
                        and hit.time_ms > end
                        and hit.time_ms - end <= keep_last_cymbal_within_ms):
                    kept.append(hit)
                    continue
                # Inside the fill — drop the cymbal
                dropped += 1
                continue
        kept.append(hit)

    if dropped > 0:
        logger.info(f"  Tom-fill protection: dropped {dropped} cymbals from "
                    f"{len(cluster_starts)} fill region(s)")
    return replace(chart, hits=kept)


# ══════════════════════════════════════════════════════════════
# 3. Hi-hat / Ride pattern completion
# ══════════════════════════════════════════════════════════════

def complete_cymbal_patterns(
    chart: DrumChart,
    min_pattern_hits: int = 8,
    max_gap_ratio: float = 2.5,
    min_fill_confidence: float = 0.7,
) -> DrumChart:
    """Fill gaps in steady hi-hat or ride cymbal patterns.

    Detects when a section has a regular cymbal pattern (e.g. 8th-note hi-hats)
    and fills in missing hits where the pattern should continue.

    Only fills if the surrounding pattern is strong enough (min_pattern_hits
    consecutive hits at roughly equal intervals).
    """
    if not chart.hits or not chart.tempo_events:
        return chart

    tempo_bpm = chart.tempo_events[0].tempo_bpm
    ms_per_beat = 60_000.0 / tempo_bpm

    # Cymbal lanes to check: hi-hat (lane=2, cymbal) and ride (lane=3, cymbal)
    cymbal_configs = [
        (2, True, "HiHat"),
        (3, True, "Ride"),
    ]

    all_hits_set = set()
    for hit in chart.hits:
        all_hits_set.add((round(hit.time_ms, 1), hit.lane, hit.is_cymbal))

    new_hits = list(chart.hits)
    total_added = 0

    for lane, is_cymbal, name in cymbal_configs:
        # Get all hits for this cymbal, sorted by time
        cymbal_hits = sorted(
            [h for h in chart.hits if h.lane == lane and h.is_cymbal == is_cymbal],
            key=lambda h: h.time_ms,
        )
        if len(cymbal_hits) < min_pattern_hits:
            continue

        # Compute intervals between consecutive hits
        times = np.array([h.time_ms for h in cymbal_hits])
        intervals = np.diff(times)

        # Find the dominant interval (most common spacing)
        # Bin intervals into 10ms buckets and find the mode
        if len(intervals) == 0:
            continue

        # Use histogram to find dominant interval
        bins = np.arange(0, ms_per_beat * 2, 10)
        hist, edges = np.histogram(intervals, bins=bins)
        if hist.max() == 0:
            continue

        dominant_bin = hist.argmax()
        dominant_interval = (edges[dominant_bin] + edges[dominant_bin + 1]) / 2

        # Expected subdivision intervals
        expected_intervals = [
            ms_per_beat / 4,   # 16th notes
            ms_per_beat / 3,   # triplets
            ms_per_beat / 2,   # 8th notes
            ms_per_beat,       # quarter notes
        ]

        # Snap dominant interval to nearest expected subdivision
        best_subdiv = min(expected_intervals, key=lambda x: abs(x - dominant_interval))
        if abs(dominant_interval - best_subdiv) > best_subdiv * 0.2:
            continue  # Doesn't match any clean subdivision

        pattern_interval = best_subdiv

        # Count how many intervals are close to the pattern
        close_mask = np.abs(intervals - pattern_interval) < pattern_interval * 0.3
        pattern_strength = close_mask.sum() / len(intervals)

        if pattern_strength < min_fill_confidence:
            continue

        # Find gaps: intervals that are ~2x the pattern (one missing hit)
        added_for_lane = 0
        for i, interval in enumerate(intervals):
            ratio = interval / pattern_interval
            if 1.5 < ratio < max_gap_ratio:
                # There's a gap — fill with interpolated hits
                n_missing = round(ratio) - 1
                for j in range(1, n_missing + 1):
                    fill_time = times[i] + j * pattern_interval
                    fill_key = (round(fill_time, 1), lane, is_cymbal)

                    # Don't add if there's already a hit nearby on the same lane
                    # (check ANY is_cymbal value — same lane = same MIDI base note)
                    already_exists = False
                    for existing_time, existing_lane, existing_cym in all_hits_set:
                        if existing_lane == lane \
                                and abs(existing_time - fill_time) < pattern_interval * 0.3:
                            already_exists = True
                            break

                    if not already_exists:
                        # Snap fill to nearest fine grid (16th/triplet/8th
                        # of pattern_interval) so cymbal rolls remain on a
                        # clean grid even when seed `times[i]` is jittered.
                        snap_grid_ms = pattern_interval / 2  # 32nd-relative
                        snapped_fill = round(fill_time / snap_grid_ms) * snap_grid_ms
                        if abs(snapped_fill - fill_time) < snap_grid_ms / 2:
                            fill_time = snapped_fill
                            fill_key = (round(fill_time, 1), lane, is_cymbal)
                        new_hits.append(DrumHit(
                            time_ms=fill_time,
                            tick=0,
                            lane=lane,
                            is_cymbal=is_cymbal,
                            velocity=90,  # Slightly lower velocity for fills
                        ))
                        all_hits_set.add(fill_key)
                        added_for_lane += 1

        if added_for_lane > 0:
            total_added += added_for_lane
            logger.info(f"  Pattern fill: {name} +{added_for_lane} hits "
                        f"(interval={pattern_interval:.0f}ms, "
                        f"strength={pattern_strength:.0%})")

    if total_added > 0:
        new_hits.sort(key=lambda h: h.time_ms)

    return replace(chart, hits=new_hits)


# ══════════════════════════════════════════════════════════════
# 4. Kick/Snare backbeat reinforcement
# ══════════════════════════════════════════════════════════════

def reinforce_backbeat(
    chart: DrumChart,
    min_section_beats: int = 8,
    min_pattern_strength: float = 0.6,
) -> DrumChart:
    """Detect and fill kick/snare backbeat patterns.

    In most rock/pop, the snare hits on beats 2 and 4 (or 2 & 4 of each bar).
    The kick typically hits on beat 1 (and often beat 3). When these are
    partially detected, fill in the gaps.

    Only fills when there's a strong existing pattern to extrapolate from.
    """
    if not chart.hits or not chart.tempo_events:
        return chart

    tempo_bpm = chart.tempo_events[0].tempo_bpm
    ms_per_beat = 60_000.0 / tempo_bpm
    ms_per_bar = ms_per_beat * 4  # Assume 4/4

    duration_ms = max(h.time_ms for h in chart.hits) if chart.hits else 0
    if duration_ms == 0:
        return chart

    # Break the song into sections of min_section_beats beats
    section_ms = ms_per_bar * (min_section_beats // 4)

    # Build time-indexed hit lookup
    kick_times = sorted(h.time_ms for h in chart.hits if h.lane == 0)
    snare_times = sorted(h.time_ms for h in chart.hits if h.lane == 1)

    new_hits = list(chart.hits)
    added_kicks = 0
    added_snares = 0

    # Process in sections
    section_start = 0.0
    while section_start < duration_ms:
        section_end = section_start + section_ms

        # Count snare hits at each beat position within bars in this section
        # Beat positions: 0 (beat 1), 1 (beat 2), 2 (beat 3), 3 (beat 4)
        snare_at_beat = defaultdict(int)
        total_bars_in_section = 0
        bar_start = section_start

        while bar_start < section_end and bar_start < duration_ms:
            total_bars_in_section += 1
            for beat in range(4):
                beat_time = bar_start + beat * ms_per_beat
                # Check if there's a snare near this beat
                for st in snare_times:
                    if abs(st - beat_time) < ms_per_beat * 0.25:
                        snare_at_beat[beat] += 1
                        break
            bar_start += ms_per_bar

        if total_bars_in_section < 2:
            section_start = section_end
            continue

        # Check for backbeat pattern: snare on beats 2 and 4
        # (or just beat 2 in half-time)
        beat2_ratio = snare_at_beat.get(1, 0) / total_bars_in_section
        beat4_ratio = snare_at_beat.get(3, 0) / total_bars_in_section

        has_backbeat = (beat2_ratio > min_pattern_strength
                        and beat4_ratio > min_pattern_strength)
        has_halftime = (beat2_ratio > min_pattern_strength
                        and beat4_ratio < 0.3)

        if has_backbeat:
            # Fill missing beats 2 and 4
            bar_start = section_start
            while bar_start < section_end and bar_start < duration_ms:
                for beat in [1, 3]:  # beats 2 and 4 (0-indexed)
                    beat_time = bar_start + beat * ms_per_beat
                    # Check if snare already exists near this time
                    has_snare = any(abs(st - beat_time) < ms_per_beat * 0.25
                                   for st in snare_times)
                    if not has_snare and beat_time < duration_ms:
                        new_hits.append(DrumHit(
                            time_ms=beat_time, tick=0, lane=1,
                            is_cymbal=False, velocity=90,
                        ))
                        snare_times.append(beat_time)
                        added_snares += 1
                bar_start += ms_per_bar
            snare_times.sort()

        elif has_halftime:
            # Fill missing beat 2 only
            bar_start = section_start
            while bar_start < section_end and bar_start < duration_ms:
                beat_time = bar_start + 1 * ms_per_beat
                has_snare = any(abs(st - beat_time) < ms_per_beat * 0.25
                               for st in snare_times)
                if not has_snare and beat_time < duration_ms:
                    new_hits.append(DrumHit(
                        time_ms=beat_time, tick=0, lane=1,
                        is_cymbal=False, velocity=90,
                    ))
                    snare_times.append(beat_time)
                    added_snares += 1
                bar_start += ms_per_bar
            snare_times.sort()

        # Kick reinforcement: if kick is on beat 1 most of the time, fill gaps
        kick_at_beat = defaultdict(int)
        bar_start = section_start
        while bar_start < section_end and bar_start < duration_ms:
            for beat in range(4):
                beat_time = bar_start + beat * ms_per_beat
                for kt in kick_times:
                    if abs(kt - beat_time) < ms_per_beat * 0.25:
                        kick_at_beat[beat] += 1
                        break
            bar_start += ms_per_bar

        beat1_ratio = kick_at_beat.get(0, 0) / total_bars_in_section
        if beat1_ratio > min_pattern_strength:
            bar_start = section_start
            while bar_start < section_end and bar_start < duration_ms:
                beat_time = bar_start  # beat 1
                has_kick = any(abs(kt - beat_time) < ms_per_beat * 0.25
                               for kt in kick_times)
                if not has_kick and beat_time < duration_ms:
                    new_hits.append(DrumHit(
                        time_ms=beat_time, tick=0, lane=0,
                        is_cymbal=False, velocity=90,
                    ))
                    kick_times.append(beat_time)
                    added_kicks += 1
                bar_start += ms_per_bar
            kick_times.sort()

        section_start = section_end

    if added_kicks > 0 or added_snares > 0:
        new_hits.sort(key=lambda h: h.time_ms)
        logger.info(f"  Backbeat: +{added_kicks} kicks, +{added_snares} snares")

    return replace(chart, hits=new_hits)


# ══════════════════════════════════════════════════════════════
# 4b. Snare roll gap fill
# ══════════════════════════════════════════════════════════════

def fill_snare_roll_gaps(
    chart: DrumChart,
    min_run_length: int = 8,
    max_gap_ratio: float = 2.5,
    max_fills_per_run: int = 3,
) -> DrumChart:
    """Fill single-hit gaps in detected snare rolls.

    In dense snare sections (16th-note rolls at ~90ms IOI), the onset
    detector sometimes misses 1-2 hits, creating holes in an otherwise
    consistent pattern.  This function detects runs of regularly-spaced
    snare hits and fills single gaps where the interval jumps to ~2x
    the normal spacing (indicating one missed hit).

    Args:
        chart: Input drum chart.
        min_run_length: Minimum number of snare hits in a run to qualify.
        max_gap_ratio: Maximum ratio of gap IOI to median IOI to fill.
        max_fills_per_run: Maximum fills allowed per detected run.
    """
    if not chart.hits:
        return chart

    snare_times = sorted(h.time_ms for h in chart.hits if h.lane == 1)
    if len(snare_times) < min_run_length:
        return chart

    # Find runs of regularly-spaced snares (IOI within 30% of median)
    iois = [snare_times[i + 1] - snare_times[i]
            for i in range(len(snare_times) - 1)]

    added = 0
    new_hits = list(chart.hits)
    existing_times = {h.time_ms for h in chart.hits if h.lane == 1}

    # Sliding window: find dense runs
    i = 0
    while i < len(iois):
        # Collect a run of similar IOIs
        run_start = i
        run_iois = [iois[i]]
        j = i + 1
        while j < len(iois):
            median_ioi = sorted(run_iois)[len(run_iois) // 2]
            if iois[j] < median_ioi * max_gap_ratio and iois[j] > median_ioi * 0.3:
                run_iois.append(iois[j])
                j += 1
            else:
                break
        run_end = j  # exclusive

        # Only process runs with enough hits and fast enough spacing
        run_len = run_end - run_start + 1  # number of hits
        if run_len >= min_run_length:
            # Filter to only the "normal" IOIs (exclude gaps)
            normal_iois = [v for v in run_iois if v < 200]
            if len(normal_iois) >= min_run_length // 2:
                median_ioi = sorted(normal_iois)[len(normal_iois) // 2]
                # Only fill in fast rolls (16th notes or faster, < 150ms)
                if median_ioi < 150:
                    # Scan for gaps: IOI roughly 2x median
                    run_fills = 0
                    for k in range(run_start, run_end):
                        if run_fills >= max_fills_per_run:
                            break
                        if iois[k] > median_ioi * 1.6 and iois[k] < median_ioi * max_gap_ratio:
                            # Fill one hit in the middle
                            fill_time = snare_times[k] + median_ioi
                            # Don't double-add
                            if not any(abs(t - fill_time) < median_ioi * 0.3
                                       for t in existing_times):
                                new_hits.append(DrumHit(
                                    time_ms=fill_time, tick=0, lane=1,
                                    is_cymbal=False, velocity=90,
                                ))
                                existing_times.add(fill_time)
                                added += 1
                                run_fills += 1

        i = max(j, i + 1)

    if added > 0:
        new_hits.sort(key=lambda h: h.time_ms)
        logger.info(f"  Snare roll fill: +{added} snares in dense sections")

    return replace(chart, hits=new_hits)


# ══════════════════════════════════════════════════════════════
# 5. Ghost note cleanup
# ══════════════════════════════════════════════════════════════

def remove_isolated_hits(
    chart: DrumChart,
    isolation_window_ms: float = 2000.0,
    min_neighbors: int = 2,
) -> DrumChart:
    """Remove isolated hits that are likely false positives.

    A hit is "isolated" if it has fewer than min_neighbors hits on ANY
    cymbal/tom lane (2-4) within isolation_window_ms. Cross-lane awareness
    preserves drum fills that sweep across toms (e.g., HiTom→MidTom→FloorTom)
    where each lane only has 1-2 hits but the combined pattern is real.

    Only applies to cymbal/tom lanes (not kick/snare which can legitimately
    have isolated hits like accents).
    """
    if not chart.hits:
        return chart

    # Only filter cymbal/tom lanes (2-4), not kick (0) or snare (1)
    filterable_lanes = {2, 3, 4}

    # Build combined sorted time list across ALL filterable lanes.
    # Fills sweep across toms quickly — a HiTom→MidTom→FloorTom fill has
    # only 1-2 hits per lane but 3+ hits across lanes within 2s.
    all_filterable_times: list[float] = []
    for hit in chart.hits:
        if hit.lane in filterable_lanes:
            all_filterable_times.append(hit.time_ms)
    all_filterable_times.sort()

    kept = []
    removed = 0

    for hit in chart.hits:
        if hit.lane not in filterable_lanes:
            kept.append(hit)
            continue

        # Count cross-lane neighbors (any hit on lanes 2-4 within window)
        neighbors = sum(
            1 for t in all_filterable_times
            if t != hit.time_ms and abs(t - hit.time_ms) < isolation_window_ms
        )

        if neighbors >= min_neighbors:
            kept.append(hit)
        else:
            removed += 1

    if removed > 0:
        logger.info(f"  Ghost cleanup: removed {removed} isolated hits")

    return replace(chart, hits=kept)


# ══════════════════════════════════════════════════════════════
# 6. Bleed compensation — restore missing snares alongside kicks
# ══════════════════════════════════════════════════════════════

def compensate_bleed(chart: DrumChart,
                     window_ms: float = 500.0,
                     min_snares: int = 4,
                     min_snare_ratio: float = 3.0) -> DrumChart:
    """Add missing snare hits alongside kicks in snare-dense sections.

    Demucs separation loses snare transients during rapid kick+snare
    sections — the kick's low-frequency energy dominates after bleed, so
    the classifier sees "kick" but the coexisting snare vanishes.

    For each kick that has no simultaneous snare, checks if the local
    neighbourhood (±window_ms) is snare-dominated.  If so, adds a snare
    at the kick's (already quantized) position.

    Runs AFTER quantization so that added snares stay on the same tick
    as the kick and aren't deduped by coarsening.
    """
    if not chart.hits:
        return chart

    import bisect

    hits = list(chart.hits)
    snare_times_sorted = sorted(h.time_ms for h in hits if h.lane == 1)
    kick_times_sorted = sorted(h.time_ms for h in hits if h.lane == 0)
    snare_time_set = set(snare_times_sorted)

    added = 0
    for h in hits:
        if h.lane != 0:
            continue
        if h.time_ms in snare_time_set:
            continue

        # Count snares within ±window
        lo = bisect.bisect_left(snare_times_sorted, h.time_ms - window_ms)
        hi = bisect.bisect_right(snare_times_sorted, h.time_ms + window_ms)
        nearby_snares = hi - lo
        if nearby_snares < min_snares:
            continue

        # Count kicks within ±window (excluding self)
        lo_k = bisect.bisect_left(kick_times_sorted, h.time_ms - window_ms)
        hi_k = bisect.bisect_right(kick_times_sorted, h.time_ms + window_ms)
        nearby_kicks = hi_k - lo_k - 1

        if nearby_kicks > 0 and nearby_snares / nearby_kicks < min_snare_ratio:
            continue

        hits.append(DrumHit(
            time_ms=h.time_ms, tick=h.tick, lane=1,
            is_cymbal=False, velocity=100,
        ))
        snare_time_set.add(h.time_ms)
        added += 1

    if added > 0:
        hits.sort(key=lambda h: h.time_ms)
        logger.info(f"  Bleed compensation: {added} snares restored alongside kicks")

    return replace(chart, hits=hits)


# ══════════════════════════════════════════════════════════════
# Beat-magnet snap (catches near-beat notes the fine grid missed)
# ══════════════════════════════════════════════════════════════

def snap_to_beat_positions(chart: DrumChart, max_offset_ms: float = 25.0,
                           run_threshold_ms: float = 90.0) -> DrumChart:
    """Pull non-run hits onto the nearest quarter or 8th-note position.

    The fine quantizer only snaps within +/-(grid/2). For a 16th grid at
    120 BPM that's +/-7.8ms, so a hit 12-25ms off a beat (e.g. detector
    drift on a long snare hit) lands on a 32nd grid point and looks
    visibly off the main beat. This pass re-snaps any non-run hit within
    ``max_offset_ms`` of a quarter-note (or 8th-note) position to that beat.

    Hits inside a fast same-lane run (neighbor within ``run_threshold_ms``)
    are left alone so cymbal/snare rolls keep their micro-timing.
    """
    if not chart.hits or not chart.tempo_events:
        return chart

    tempo_bpm = chart.tempo_events[0].tempo_bpm
    ms_per_beat = 60_000.0 / tempo_bpm
    eighth_ms = ms_per_beat / 2

    hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)

    # Build per-lane time lists for SAME-lane run detection
    lane_times: dict[tuple[int, bool], list[float]] = defaultdict(list)
    for h in hits_sorted:
        lane_times[(h.lane, h.is_cymbal)].append(h.time_ms)

    # Also build a global timeline (drum lanes only — exclude kick lane 0
    # which often coincides with hand notes on strong beats and is NOT
    # part of a hand roll). Used to detect cross-lane fast rolls like
    # LowTom→FloorTom→LowTom that would otherwise be magnetized into
    # simultaneous chords.
    drum_times = [h.time_ms for h in hits_sorted if h.lane >= 1]

    import bisect

    def _in_run(t: float, key: tuple[int, bool], lane: int) -> bool:
        # Same-lane run (cymbal/snare rolls)
        times = lane_times[key]
        pos = bisect.bisect_left(times, t)
        if pos > 0 and (t - times[pos - 1]) <= run_threshold_ms:
            return True
        if pos < len(times) - 1 and (times[pos + 1] - t) <= run_threshold_ms:
            return True
        # Cross-lane hand-roll detection: any neighboring hand hit within
        # run_threshold_ms means this hit is part of a fast pattern and
        # must NOT be snapped (would collapse the roll into a chord).
        if lane >= 1 and drum_times:
            gpos = bisect.bisect_left(drum_times, t)
            if gpos > 0 and (t - drum_times[gpos - 1]) <= run_threshold_ms \
                    and (t - drum_times[gpos - 1]) > 0.5:
                return True
            if gpos < len(drum_times) - 1 \
                    and (drum_times[gpos + 1] - t) <= run_threshold_ms \
                    and (drum_times[gpos + 1] - t) > 0.5:
                return True
        return False

    snapped = 0
    new_hits = []
    for hit in hits_sorted:
        if _in_run(hit.time_ms, (hit.lane, hit.is_cymbal), hit.lane):
            new_hits.append(hit)
            continue
        # Try quarter-beat first, then 8th — wider tolerance for the more
        # important beat positions.
        nearest_beat = round(hit.time_ms / ms_per_beat) * ms_per_beat
        nearest_eighth = round(hit.time_ms / eighth_ms) * eighth_ms
        d_beat = abs(hit.time_ms - nearest_beat)
        d_eighth = abs(hit.time_ms - nearest_eighth)
        if d_beat <= max_offset_ms and d_beat <= d_eighth + 2.0:
            target = nearest_beat
        elif d_eighth <= max_offset_ms * 0.65:
            target = nearest_eighth
        else:
            new_hits.append(hit)
            continue
        if abs(hit.time_ms - target) > 0.5:
            snapped += 1
            new_hits.append(replace(hit, time_ms=target))
        else:
            new_hits.append(hit)

    # Dedup
    new_hits.sort(key=lambda h: (h.time_ms, h.lane, not h.is_cymbal))
    deduped = []
    for hit in new_hits:
        if deduped and abs(hit.time_ms - deduped[-1].time_ms) < 1.0 \
                and hit.lane == deduped[-1].lane:
            continue
        deduped.append(hit)

    if snapped > 0:
        logger.info(f"  Beat magnet: {snapped} hits pulled onto nearest beat/8th")

    return replace(chart, hits=deduped)


# ══════════════════════════════════════════════════════════════
# Single-tom-per-onset constraint (drummer physics)
# ══════════════════════════════════════════════════════════════

def enforce_single_tom_per_onset(chart: DrumChart,
                                 window_ms: float = 30.0) -> DrumChart:
    """Allow at most ONE tom hit per onset.

    A drummer cannot strike two toms simultaneously with one stick. When
    multiple tom hits (lanes 2/3/4 with is_cymbal=False) appear within
    ``window_ms`` of each other, keep only the highest-velocity one.
    Cymbal+tom and snare+tom co-occurrences remain valid (different hands
    or hand+foot).

    This is a downstream safety net for cases where the classifier or
    tom-refinement rescue introduces same-onset tom-tom chords (e.g.
    HighTom+FloorTom) that confuse fast-roll patterns into chord stacks.
    """
    if not chart.hits:
        return chart

    hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)
    keep_mask = [True] * len(hits_sorted)
    removed = 0

    i = 0
    while i < len(hits_sorted):
        # Build a window of hits within window_ms of hits_sorted[i]
        j = i
        while j + 1 < len(hits_sorted) and \
                (hits_sorted[j + 1].time_ms - hits_sorted[i].time_ms) <= window_ms:
            j += 1
        # Find tom hits in this window
        tom_idxs = [k for k in range(i, j + 1)
                    if (not hits_sorted[k].is_cymbal
                        and hits_sorted[k].lane in (2, 3, 4)
                        and keep_mask[k])]
        if len(tom_idxs) > 1:
            # Keep the highest-velocity one (proxy for highest-prob)
            best = max(tom_idxs, key=lambda k: hits_sorted[k].velocity)
            for k in tom_idxs:
                if k != best:
                    keep_mask[k] = False
                    removed += 1
        i = j + 1

    if removed > 0:
        logger.info(f"  Single-tom-per-onset: removed {removed} extra tom hits "
                    f"(window={window_ms:.0f}ms)")

    new_hits = [h for k, h in enumerate(hits_sorted) if keep_mask[k]]
    return replace(chart, hits=new_hits)


# ══════════════════════════════════════════════════════════════
# Close-hands cap (catches visually-simultaneous hand stacks the
# 1ms playability grouping misses)
# ══════════════════════════════════════════════════════════════

def cap_close_hands(chart: DrumChart, window_ms: float | None = None,
                    max_window_ms: float = 60.0) -> DrumChart:
    """Cap to 2 hands within a tempo-aware playability window.

    ``resolve_playability`` only groups hits within +/-1ms (post-quantize
    same-tick). After pattern fill / snare-roll fill / spectral reclass
    add new hits that DON'T get re-quantized, two hand notes on adjacent
    grid points (~30ms apart) can survive and look visually simultaneous
    in fast fills, making them unplayable.

    This pass groups hand notes (lanes 1-4) within ``window_ms`` (default
    min(60ms, 1/8 beat)) and keeps at most 2 — preferring cymbals over toms
    on shared lanes (matches resolve_playability semantics).
    Kick (lane 0) is foot-only and never capped.
    """
    if not chart.hits:
        return chart

    if window_ms is None:
        if chart.tempo_events:
            ms_per_beat = 60_000.0 / chart.tempo_events[0].tempo_bpm
            window_ms = min(max_window_ms, ms_per_beat / 8)
        else:
            window_ms = max_window_ms

    hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)
    kept = []
    capped = 0

    i = 0
    while i < len(hits_sorted):
        # Build a window starting at hits_sorted[i]
        window_start = hits_sorted[i].time_ms
        j = i
        while j < len(hits_sorted) and (hits_sorted[j].time_ms - window_start) <= window_ms:
            j += 1
        group = hits_sorted[i:j]

        hand_hits = [h for h in group if h.lane >= 1]
        non_hand = [h for h in group if h.lane < 1]

        if len(hand_hits) > 2:
            capped += 1
            # Same priority as resolve_playability: toms first (drumsep
            # arbiter validated), then by lane.
            hand_hits.sort(key=lambda h: (h.is_cymbal, h.lane))
            hand_hits = hand_hits[:2]

        kept.extend(non_hand)
        kept.extend(hand_hits)
        i = j

    kept.sort(key=lambda h: h.time_ms)
    removed = len(chart.hits) - len(kept)
    if capped > 0 or removed > 0:
        logger.info(f"  Close-hands cap (window={window_ms:.0f}ms): {capped} groups capped, {removed} hits removed")

    return replace(chart, hits=kept)


# ══════════════════════════════════════════════════════════════
# Combined post-processing pipeline
def resolve_playability(chart: DrumChart) -> DrumChart:
    """Final playability cleanup after quantization.

    Fixes issues caused by nearby onsets being snapped to the same grid point:
      1. Lane conflicts: when a cymbal and tom on the same lane share a tick,
         keep only the cymbal (dominant class).
      2. Hand cap: max 2 hand notes per tick (drummer has 2 hands).
         Hand lanes are 1-4 (Snare, Yellow, Blue, Green). Kick (lane 0) is foot.
    """
    if not chart.hits:
        return chart

    from collections import defaultdict

    # Group hits by quantized time. We use a 6ms tolerance so that hits which
    # the various passes added at slightly different sub-tick times still get
    # deduped before export rounds them to the same tick. (At 480 tpb /
    # 120 BPM, one tick = 1.04 ms, so 6ms is ~6 ticks — well below any
    # legitimate grid spacing yet wide enough to catch float-jitter.)
    LANE_DEDUP_MS = 6.0
    groups: list[tuple[float, list[DrumHit]]] = []
    hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)
    for hit in hits_sorted:
        if groups and abs(hit.time_ms - groups[-1][0]) < LANE_DEDUP_MS:
            groups[-1][1].append(hit)
        else:
            groups.append((hit.time_ms, [hit]))

    kept = []
    lane_conflicts = 0
    hand_caps = 0

    for _t, group in groups:
        # 1. Lane dedup: if both cymbal and tom on same lane, keep cymbal
        by_lane: dict[int, list[DrumHit]] = defaultdict(list)
        for hit in group:
            by_lane[hit.lane].append(hit)

        deduped_group = []
        for lane, lane_hits in by_lane.items():
            if len(lane_hits) == 1:
                deduped_group.append(lane_hits[0])
            else:
                # Multiple hits on same lane at the same tick. Prefer TOM:
                # the drumsep arbiter (validated >=98% Tom->Cym fix rate)
                # only adds the tom variant when the physical signal in
                # toms.wav clearly dominates cymbals.wav. If a tom is
                # present here it's an explicit signal we should keep.
                # If only cymbals (no tom), keep the loudest one.
                toms = [h for h in lane_hits if not h.is_cymbal]
                cymbals = [h for h in lane_hits if h.is_cymbal]
                if toms:
                    # Pick the highest-velocity tom (often only one)
                    deduped_group.append(max(toms, key=lambda h: h.velocity))
                    if cymbals:
                        lane_conflicts += 1
                else:
                    deduped_group.append(max(cymbals, key=lambda h: h.velocity))

        # 2. Hand cap: max 2 hand notes (lanes 1-4)
        hand_hits = [h for h in deduped_group if h.lane >= 1]
        non_hand = [h for h in deduped_group if h.lane < 1]

        if len(hand_hits) > 2:
            hand_caps += 1
            # Keep the 2 most musical hands. Toms come first (drumsep
            # arbiter is high-confidence and tom fills are intentional),
            # then by lane to keep ergonomic groupings.
            hand_hits.sort(key=lambda h: (h.is_cymbal, h.lane))
            hand_hits = hand_hits[:2]

        kept.extend(non_hand)
        kept.extend(hand_hits)

    kept.sort(key=lambda h: h.time_ms)

    removed = len(chart.hits) - len(kept)
    logger.info(f"  Playability: {lane_conflicts} lane conflicts, "
                f"{hand_caps} hand caps, {removed} hits removed")

    return replace(chart, hits=kept)


# ══════════════════════════════════════════════════════════════

def _quantize_segment(seg_chart: DrumChart, tempo_bpm: float) -> DrumChart:
    """Run the four-stage quantization stack on a single tempo segment.

    Each segment is quantized using its own BPM and an independently
    computed phase offset. This prevents global drift on songs with
    tempo changes (where a single global grid built on the first
    tempo would get progressively out of phase with the audio in
    later segments).
    """
    ms_per_beat = 60_000.0 / tempo_bpm
    MIN_GRID_MS = 20.0  # V14 onset detector min_distance_ms
    for subdiv in [192, 96, 64, 48, 32, 24, 16, 12, 8, 4]:
        grid_ms = ms_per_beat / (subdiv / 4)
        if grid_ms >= MIN_GRID_MS:
            break
    else:
        subdiv = 4

    fine_grid_ms = ms_per_beat / (subdiv / 4)
    phase_offset = _compute_phase_offset([h.time_ms for h in seg_chart.hits], fine_grid_ms)

    _STANDARD_SUBDIVS = [4, 8, 12, 16, 24, 32, 48, 64, 96, 192]
    if subdiv in _STANDARD_SUBDIVS:
        _fine_idx = _STANDARD_SUBDIVS.index(subdiv)
        coarse_subdiv = _STANDARD_SUBDIVS[max(0, _fine_idx - 1)]
    else:
        coarse_subdiv = 32

    logger.info(
        f"    segment @{tempo_bpm:.2f} BPM: grid={subdiv}th "
        f"({fine_grid_ms:.1f}ms)  coarse={coarse_subdiv}th  "
        f"phase={phase_offset:.1f}ms  hits={len(seg_chart.hits)}"
    )

    seg_chart = quantize_hits(seg_chart, max_subdivision=subdiv,
                              phase_offset_ms=phase_offset)
    seg_chart = regularize_runs(seg_chart, max_subdivision=subdiv,
                                phase_offset_ms=phase_offset)
    seg_chart = coarsen_non_runs(seg_chart, fine_subdivision=subdiv,
                                 coarse_subdivision=coarse_subdiv,
                                 phase_offset_ms=phase_offset)
    seg_chart = align_simultaneous_hits(seg_chart, max_subdivision=subdiv,
                                        coarse_subdivision=coarse_subdiv,
                                        phase_offset_ms=phase_offset)
    return seg_chart


def postprocess_chart(chart: DrumChart) -> DrumChart:
    """Apply all post-processing steps in order.

    Order matters:
      1. Min gap first (remove artifacts before pattern detection)
      2. Quantize (snap to grid before pattern analysis)
      3. Playability (resolve lane conflicts and hand caps from quantization)
      4. Pattern completion (needs clean grid-aligned hits)
      5. Backbeat reinforcement (after pattern fill)
      6. Ghost cleanup last (after everything else has been added)

    Quantization (step 2) is applied PER TEMPO SEGMENT so that songs
    with tempo changes do not accumulate phase drift from using a
    single global BPM/phase across the whole song.
    """
    if not chart.hits:
        return chart

    original_count = len(chart.hits)
    logger.info(f"  Post-processing {original_count} hits...")

    chart = enforce_min_gap(chart, min_gap_ms=20.0)

    # Per-segment quantization. Build segment boundaries from
    # tempo_events: segment i covers [events[i].time_ms, events[i+1].time_ms).
    tempo_events = chart.tempo_events or [TempoEvent(tick=0, tempo_bpm=120.0, time_ms=0.0)]
    n_segments = len(tempo_events)
    if n_segments == 1:
        logger.info(f"  Quantization: single tempo segment @{tempo_events[0].tempo_bpm:.2f} BPM")
    else:
        logger.info(f"  Quantization: {n_segments} tempo segments")

    seg_boundaries = [te.time_ms for te in tempo_events] + [float("inf")]
    new_hits: list[DrumHit] = []
    for seg_idx in range(n_segments):
        lo, hi = seg_boundaries[seg_idx], seg_boundaries[seg_idx + 1]
        seg_hits = [h for h in chart.hits if lo <= h.time_ms < hi]
        if not seg_hits:
            continue
        seg_chart = DrumChart(
            hits=seg_hits,
            tempo_events=[tempo_events[seg_idx]],
            time_signatures=chart.time_signatures,
            ticks_per_beat=chart.ticks_per_beat,
        )
        seg_chart = _quantize_segment(seg_chart, tempo_events[seg_idx].tempo_bpm)
        new_hits.extend(seg_chart.hits)

    chart = DrumChart(
        hits=sorted(new_hits, key=lambda h: h.time_ms),
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )
    chart = resolve_playability(chart)
    # Beat magnet: pull near-beat notes onto the actual beat/8th grid.
    # Runs after the segment quantize stack so it sees post-aligned hits.
    chart = snap_to_beat_positions(chart)
    if PP_LANE_FLIPS:
        chart = clean_isolated_lane_flips(chart)
    if PP_PATTERN_FILL:
        chart = complete_cymbal_patterns(chart)
    if PP_BACKBEAT:
        chart = reinforce_backbeat(chart)
    if PP_SNARE_ROLL:
        chart = fill_snare_roll_gaps(chart)
    if PP_GHOST_KILL:
        chart = remove_isolated_hits(chart)
    # Final close-hands cap: pattern_fill / snare_roll / phase3 can add
    # hits that bypass resolve_playability's 1ms grouping. Run a wider
    # tempo-aware cap so visually-simultaneous hand stacks (~1/8 beat
    # apart) are reduced to 2 hands per group.
    chart = cap_close_hands(chart)

    final_count = len(chart.hits)
    diff = final_count - original_count
    sign = "+" if diff >= 0 else ""
    logger.info(f"  Post-processing done: {original_count} → {final_count} hits "
                f"({sign}{diff})")

    return chart
