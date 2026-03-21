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
from dataclasses import dataclass, replace
from collections import defaultdict

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.parsers.midi_parser import DrumChart, DrumHit, TempoEvent

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 1. Rhythmic quantization
# ══════════════════════════════════════════════════════════════

def quantize_hits(chart: DrumChart, max_subdivision: int = 16) -> DrumChart:
    """Snap hit times to the nearest beat subdivision.

    Args:
        chart: Input drum chart
        max_subdivision: Maximum subdivision to quantize to (8=8th notes,
            16=16th notes, 32=32nd notes). Higher = less quantization.

    Returns:
        New DrumChart with quantized hit times.
    """
    if not chart.hits or not chart.tempo_events:
        return chart

    tempo_bpm = chart.tempo_events[0].tempo_bpm
    ms_per_beat = 60_000.0 / tempo_bpm
    # Grid spacing: one subdivision
    grid_ms = ms_per_beat / (max_subdivision / 4)

    new_hits = []
    for hit in chart.hits:
        # Find nearest grid point
        grid_pos = round(hit.time_ms / grid_ms)
        snapped_ms = grid_pos * grid_ms

        # Only snap if within a reasonable window (half a grid cell)
        if abs(hit.time_ms - snapped_ms) <= grid_ms / 2:
            new_hits.append(replace(hit, time_ms=snapped_ms))
        else:
            new_hits.append(hit)

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

                    # Don't add if there's already a hit nearby
                    already_exists = False
                    for existing_time, existing_lane, existing_cym in all_hits_set:
                        if existing_lane == lane and existing_cym == is_cymbal \
                                and abs(existing_time - fill_time) < pattern_interval * 0.3:
                            already_exists = True
                            break

                    if not already_exists:
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

    # Group hits by quantized time (within 1ms tolerance)
    groups: list[tuple[float, list[DrumHit]]] = []
    hits_sorted = sorted(chart.hits, key=lambda h: h.time_ms)
    for hit in hits_sorted:
        if groups and abs(hit.time_ms - groups[-1][0]) < 1.0:
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
                # Multiple hits on same lane at same tick — keep cymbal if present
                cymbals = [h for h in lane_hits if h.is_cymbal]
                toms = [h for h in lane_hits if not h.is_cymbal]
                if cymbals:
                    deduped_group.append(cymbals[0])
                    if toms:
                        lane_conflicts += 1
                else:
                    deduped_group.append(toms[0])

        # 2. Hand cap: max 2 hand notes (lanes 1-4)
        hand_hits = [h for h in deduped_group if h.lane >= 1]
        non_hand = [h for h in deduped_group if h.lane < 1]

        if len(hand_hits) > 2:
            hand_caps += 1
            # Keep the 2 hands that best fit the song flow:
            # Prioritize cymbal hits (groove) over toms (fills are rarer)
            hand_hits.sort(key=lambda h: (not h.is_cymbal, h.lane))
            hand_hits = hand_hits[:2]

        kept.extend(non_hand)
        kept.extend(hand_hits)

    kept.sort(key=lambda h: h.time_ms)

    removed = len(chart.hits) - len(kept)
    logger.info(f"  Playability: {lane_conflicts} lane conflicts, "
                f"{hand_caps} hand caps, {removed} hits removed")

    return replace(chart, hits=kept)


# ══════════════════════════════════════════════════════════════

def postprocess_chart(chart: DrumChart) -> DrumChart:
    """Apply all post-processing steps in order.

    Order matters:
      1. Min gap first (remove artifacts before pattern detection)
      2. Quantize (snap to grid before pattern analysis)
      3. Playability (resolve lane conflicts and hand caps from quantization)
      4. Pattern completion (needs clean grid-aligned hits)
      5. Backbeat reinforcement (after pattern fill)
      6. Ghost cleanup last (after everything else has been added)
    """
    if not chart.hits:
        return chart

    original_count = len(chart.hits)
    logger.info(f"  Post-processing {original_count} hits...")

    chart = enforce_min_gap(chart, min_gap_ms=20.0)

    # Always use 48th note grid — 32nd is too coarse for snare rolls
    # at moderate tempos (e.g., 83.5 BPM: 32nd=89.8ms but rolls are ~81ms apart,
    # causing 20-30% of roll notes to dedup). 48th (59.9ms) captures them;
    # 64th adds nothing beyond 48th.
    subdiv = 48

    chart = quantize_hits(chart, max_subdivision=subdiv)
    chart = resolve_playability(chart)
    chart = complete_cymbal_patterns(chart)
    chart = reinforce_backbeat(chart)
    chart = remove_isolated_hits(chart)

    final_count = len(chart.hits)
    diff = final_count - original_count
    sign = "+" if diff >= 0 else ""
    logger.info(f"  Post-processing done: {original_count} → {final_count} hits "
                f"({sign}{diff})")

    return chart
