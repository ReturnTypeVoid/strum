"""
MIDI export for STRUM.

Exports drum charts to standard MIDI format compatible with Clone Hero/YARG.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging
import os

import mido

from src.preprocessing.parsers.midi_parser import DrumChart, DrumHit, TempoEvent

logger = logging.getLogger(__name__)


# Pro drums MIDI note mapping (Clone Hero/YARG standard - Expert difficulty)
# Reference: https://github.com/TheNathannator/GuitarGame_ChartFormats
# 
# Clone Hero / YARG expects:
# - Kick: 96
# - Red (Snare): 97  
# - Yellow: 98 (default=CYMBAL, tom marker 110 makes it tom)
# - Blue: 99 (default=CYMBAL, tom marker 111 makes it tom)
# - Green: 100 (default=CYMBAL, tom marker 112 makes it tom)
#
# For cymbals: ONLY output the base note (98/99/100), no tom marker
# For toms: Output BOTH the base note AND the tom marker (e.g., 98 + 110)

PRO_DRUMS_MIDI_MAP = {
    # lane: (base_note, tom_marker)
    0: (96, None),     # Kick - no cymbal variant
    1: (97, None),     # Red/Snare - no cymbal variant
    2: (98, 110),      # Yellow - base note + tom marker 110 for tom
    3: (99, 111),      # Blue - base note + tom marker 111 for tom
    4: (100, 112),     # Green - base note + tom marker 112 for tom
}

# Difficulty note offsets (Clone Hero standard)
# Expert uses the base notes (96-100), other difficulties use offsets
DIFFICULTY_NOTE_OFFSETS = {
    "expert": 0,    # Notes 96-100
    "hard": -12,    # Notes 84-88
    "medium": -24,  # Notes 72-76
    "easy": -36,    # Notes 60-64
}


@dataclass
class MidiExportConfig:
    """Configuration for MIDI export."""
    
    ticks_per_beat: int = 480
    default_tempo_bpm: float = 120.0
    include_tempo_track: bool = True
    track_name: str = "PART DRUMS"


def export_drums_midi(
    chart: DrumChart,
    output_path: Path,
    config: Optional[MidiExportConfig] = None,
) -> None:
    """
    Export a drum chart to MIDI format.
    
    Args:
        chart: DrumChart with hits and tempo info
        output_path: Path to write .mid file
        config: Export configuration
    """
    if config is None:
        config = MidiExportConfig()
    
    # Create MIDI file
    mid = mido.MidiFile(type=1, ticks_per_beat=config.ticks_per_beat)
    
    # Create tempo track
    if config.include_tempo_track:
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.name = "Tempo Track"
        
        # Add tempo events
        if chart.tempo_events:
            prev_tick = 0
            for tempo_event in sorted(chart.tempo_events, key=lambda t: t.tick):
                delta = tempo_event.tick - prev_tick
                tempo_us = int(60_000_000 / tempo_event.tempo_bpm)
                tempo_track.append(mido.MetaMessage(
                    "set_tempo",
                    tempo=tempo_us,
                    time=delta,
                ))
                prev_tick = tempo_event.tick
        else:
            # Add default tempo
            tempo_us = int(60_000_000 / config.default_tempo_bpm)
            tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        
        # Add time signatures
        if chart.time_signatures:
            prev_tick = 0
            for ts in sorted(chart.time_signatures, key=lambda t: t.tick):
                delta = ts.tick - prev_tick
                tempo_track.append(mido.MetaMessage(
                    "time_signature",
                    numerator=ts.numerator,
                    denominator=ts.denominator,
                    time=delta,
                ))
                prev_tick = ts.tick
        else:
            tempo_track.append(mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                time=0,
            ))
        
        tempo_track.append(mido.MetaMessage("end_of_track", time=0))
    
    # Create drums track
    drums_track = mido.MidiTrack()
    mid.tracks.append(drums_track)
    drums_track.name = config.track_name
    
    # Convert hits to ticks if needed
    hits_with_ticks = _compute_hit_ticks(chart, config.ticks_per_beat)
    
    # Sort by tick
    hits_with_ticks.sort(key=lambda h: h.tick)

    # ------------------------------------------------------------------
    # Lane+tick safety dedup. Upstream passes occasionally leak two hits
    # on the same lane at the same export tick (one tom, one cymbal),
    # producing duplicate `note_on` events that show up as a stacked
    # gem in Clone Hero. Prefer the tom variant (drumsep arbiter is
    # high-confidence); fall back to highest-velocity cymbal otherwise.
    # ------------------------------------------------------------------
    seen: dict[tuple[int, int], int] = {}  # (tick, lane) -> index in unique
    unique: list = []
    for h in hits_with_ticks:
        key = (h.tick, h.lane)
        if key not in seen:
            seen[key] = len(unique)
            unique.append(h)
        else:
            existing = unique[seen[key]]
            # Prefer tom over cymbal; among same kind, prefer louder.
            if (not h.is_cymbal and existing.is_cymbal) or (
                h.is_cymbal == existing.is_cymbal and h.velocity > existing.velocity
            ):
                unique[seen[key]] = h
    if len(unique) != len(hits_with_ticks):
        # Brief log so regressions are visible without grepping
        try:
            from src.preprocessing.parsers.midi_parser import logger as _mp_log  # type: ignore
        except Exception:
            _mp_log = None
        msg = f"  Export dedup: removed {len(hits_with_ticks) - len(unique)} same-lane duplicates"
        if _mp_log:
            _mp_log.info(msg)
        else:
            print(msg)
    hits_with_ticks = unique

    # ------------------------------------------------------------------
    # 2x bass pedal detection.
    # Clone Hero / YARG convention: MIDI note 95 = "expert+ kick" (the
    # 2nd pedal). When a player has only a single pedal, those notes are
    # hidden; with two pedals enabled, both note 95 and note 96 play.
    # We promote the SECOND kick of any rapid kick pair (gap < 175 ms,
    # ~ 16th-notes at 100 BPM) to note 95 so the regular 1x bass chart
    # remains playable while a 2x bass version is available.
    # Disable via STRUM_TWO_KICK=0.
    # ------------------------------------------------------------------
    enable_2x = os.environ.get("STRUM_TWO_KICK", "1") == "1"
    two_kick_gap_ms = float(os.environ.get("STRUM_TWO_KICK_GAP_MS", "175"))
    is_two_kick: dict[int, bool] = {}
    if enable_2x:
        prev_kick_ms = -1e9
        for idx, h in enumerate(hits_with_ticks):
            if h.lane != 0:
                continue
            if h.time_ms - prev_kick_ms < two_kick_gap_ms:
                is_two_kick[idx] = True
            prev_kick_ms = h.time_ms

    # Build MIDI events
    events = []
    
    for idx, hit in enumerate(hits_with_ticks):
        base_note = _get_midi_note(hit)
        # Promote 2nd kick of a fast pair to note 95 (2x bass).
        if hit.lane == 0 and is_two_kick.get(idx, False):
            base_note = 95
        tom_marker = _get_tom_marker(hit)
        velocity = hit.velocity
        
        # Note on for the base drum note
        events.append({
            "type": "note_on",
            "tick": hit.tick,
            "note": base_note,
            "velocity": velocity,
        })
        
        # Note off shortly after (1/32 note duration)
        note_duration = config.ticks_per_beat // 8
        events.append({
            "type": "note_off",
            "tick": hit.tick + note_duration,
            "note": base_note,
            "velocity": 0,
        })
        
        # If this is a tom (not cymbal), also output the tom marker
        if tom_marker is not None:
            events.append({
                "type": "note_on",
                "tick": hit.tick,
                "note": tom_marker,
                "velocity": velocity,
            })
            events.append({
                "type": "note_off",
                "tick": hit.tick + note_duration,
                "note": tom_marker,
                "velocity": 0,
            })
    
    # Sort events by tick
    events.sort(key=lambda e: (e["tick"], e["type"] == "note_off"))
    
    # Convert to MIDI messages with delta times
    prev_tick = 0
    for event in events:
        delta = event["tick"] - prev_tick
        
        if event["type"] == "note_on":
            drums_track.append(mido.Message(
                "note_on",
                note=event["note"],
                velocity=event["velocity"],
                time=delta,
            ))
        else:
            drums_track.append(mido.Message(
                "note_off",
                note=event["note"],
                velocity=0,
                time=delta,
            ))
        
        prev_tick = event["tick"]
    
    drums_track.append(mido.MetaMessage("end_of_track", time=0))
    
    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(output_path)
    
    logger.info(f"Exported {len(hits_with_ticks)} drum hits to {output_path}")


def _compute_hit_ticks(
    chart: DrumChart,
    ticks_per_beat: int,
) -> list[DrumHit]:
    """
    Convert hit times from ms to ticks.
    
    Uses tempo events to compute accurate tick positions.
    """
    if not chart.hits:
        return []
    
    # Build tempo map: list of (time_ms, tick, tempo_bpm)
    tempo_map = []
    if chart.tempo_events:
        for te in sorted(chart.tempo_events, key=lambda t: t.time_ms):
            tempo_map.append((te.time_ms, te.tick, te.tempo_bpm))
    else:
        tempo_map.append((0.0, 0, 120.0))
    
    hits_with_ticks = []
    
    for hit in chart.hits:
        # Find tempo segment containing this hit
        segment_idx = 0
        for i, (time_ms, _, _) in enumerate(tempo_map):
            if time_ms <= hit.time_ms:
                segment_idx = i
            else:
                break
        
        segment_start_ms, segment_start_tick, tempo_bpm = tempo_map[segment_idx]
        
        # Compute ticks from segment start
        ms_per_beat = 60_000 / tempo_bpm
        ms_per_tick = ms_per_beat / ticks_per_beat
        
        delta_ms = hit.time_ms - segment_start_ms
        delta_ticks = round(delta_ms / ms_per_tick)
        tick = segment_start_tick + delta_ticks
        
        hits_with_ticks.append(DrumHit(
            time_ms=hit.time_ms,
            tick=tick,
            lane=hit.lane,
            is_cymbal=hit.is_cymbal,
            velocity=hit.velocity,
        ))
    
    return hits_with_ticks


def _get_midi_note(hit: DrumHit) -> int:
    """Get MIDI note number for a drum hit (base note only)."""
    base_note, _ = PRO_DRUMS_MIDI_MAP[hit.lane]
    return base_note


def _get_tom_marker(hit: DrumHit) -> Optional[int]:
    """Get tom marker note if this is a tom hit (not cymbal)."""
    _, tom_marker = PRO_DRUMS_MIDI_MAP[hit.lane]
    
    # Tom marker only needed for yellow/blue/green AND when it's NOT a cymbal
    # (i.e., when is_cymbal=False, we need the tom marker)
    if tom_marker is not None and not hit.is_cymbal:
        return tom_marker
    return None


def generate_difficulties(
    expert_chart: DrumChart,
    output_dir: Path,
    song_name: str,
) -> dict[str, Path]:
    """
    Generate all difficulty levels from expert chart.
    
    Applies reduction rules to create Hard/Medium/Easy versions.
    
    Args:
        expert_chart: Full expert-level drum chart
        output_dir: Directory to write files
        song_name: Base name for output files
        
    Returns:
        Dict mapping difficulty name to output path
    """
    outputs = {}
    
    # Expert: full chart
    expert_path = output_dir / f"{song_name}_expert_drums.mid"
    export_drums_midi(expert_chart, expert_path)
    outputs["expert"] = expert_path
    
    # Hard: remove some ghost notes, simplify fast rolls
    hard_chart = _reduce_to_hard(expert_chart)
    hard_path = output_dir / f"{song_name}_hard_drums.mid"
    export_drums_midi(hard_chart, hard_path)
    outputs["hard"] = hard_path
    
    # Medium: further simplification
    medium_chart = _reduce_to_medium(hard_chart)
    medium_path = output_dir / f"{song_name}_medium_drums.mid"
    export_drums_midi(medium_chart, medium_path)
    outputs["medium"] = medium_path
    
    # Easy: basic pattern only
    easy_chart = _reduce_to_easy(medium_chart)
    easy_path = output_dir / f"{song_name}_easy_drums.mid"
    export_drums_midi(easy_chart, easy_path)
    outputs["easy"] = easy_path
    
    return outputs


def export_all_difficulties(
    expert_chart: DrumChart,
    output_path: Path,
    config: Optional[MidiExportConfig] = None,
) -> None:
    """
    Export drum chart with all 4 difficulty levels in a single MIDI file.
    
    Clone Hero expects all difficulties in one track (PART DRUMS) with
    different note ranges for each difficulty:
    - Expert: 96-100 (base notes)
    - Hard: 84-88 (-12 offset)
    - Medium: 72-76 (-24 offset)
    - Easy: 60-64 (-36 offset)
    
    Args:
        expert_chart: Expert-level drum chart
        output_path: Path to write .mid file
        config: Export configuration
    """
    if config is None:
        config = MidiExportConfig()
    
    # Generate reduced charts for each difficulty
    hard_chart = _reduce_to_hard(expert_chart)
    medium_chart = _reduce_to_medium(hard_chart)
    easy_chart = _reduce_to_easy(medium_chart)
    
    difficulty_charts = {
        "expert": expert_chart,
        "hard": hard_chart,
        "medium": medium_chart,
        "easy": easy_chart,
    }
    
    # Create MIDI file
    mid = mido.MidiFile(type=1, ticks_per_beat=config.ticks_per_beat)
    
    # Create tempo track
    if config.include_tempo_track:
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.name = "Tempo Track"
        
        # Add tempo events
        if expert_chart.tempo_events:
            prev_tick = 0
            for tempo_event in sorted(expert_chart.tempo_events, key=lambda t: t.tick):
                delta = tempo_event.tick - prev_tick
                tempo_us = int(60_000_000 / tempo_event.tempo_bpm)
                tempo_track.append(mido.MetaMessage(
                    "set_tempo",
                    tempo=tempo_us,
                    time=delta,
                ))
                prev_tick = tempo_event.tick
        else:
            tempo_us = int(60_000_000 / config.default_tempo_bpm)
            tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        
        # Add time signatures
        if expert_chart.time_signatures:
            prev_tick = 0
            for ts in sorted(expert_chart.time_signatures, key=lambda t: t.tick):
                delta = ts.tick - prev_tick
                tempo_track.append(mido.MetaMessage(
                    "time_signature",
                    numerator=ts.numerator,
                    denominator=ts.denominator,
                    time=delta,
                ))
                prev_tick = ts.tick
        else:
            tempo_track.append(mido.MetaMessage(
                "time_signature",
                numerator=4,
                denominator=4,
                time=0,
            ))
        
        tempo_track.append(mido.MetaMessage("end_of_track", time=0))
    
    # Create drums track with all difficulties
    drums_track = mido.MidiTrack()
    mid.tracks.append(drums_track)
    drums_track.name = config.track_name
    
    # Build events for all difficulties
    events = []
    
    # Track tom markers separately — they are global flags (110/111/112)
    # that apply across ALL difficulties and must NOT be offset.
    # Deduplicate: only emit each (tick, marker) once.
    tom_marker_events: set[tuple[int, int]] = set()

    for difficulty, chart in difficulty_charts.items():
        note_offset = DIFFICULTY_NOTE_OFFSETS[difficulty]
        hits_with_ticks = _compute_hit_ticks(chart, config.ticks_per_beat)
        hits_with_ticks.sort(key=lambda h: h.tick)

        # 2x bass detection — Expert track only.  Promote 2nd kick of any
        # rapid pair (gap < STRUM_TWO_KICK_GAP_MS, default 175 ms ⇒ ~16ths
        # at 100 BPM) to MIDI note 95.  Clone Hero / YARG render note 95
        # only when the player has 2x bass enabled, leaving a clean 1x
        # kick chart (note 96) for single-pedal players.  Disable via
        # STRUM_TWO_KICK=0.
        is_two_kick: dict[int, bool] = {}
        if difficulty == "expert" and os.environ.get("STRUM_TWO_KICK", "1") == "1":
            two_kick_gap_ms = float(os.environ.get("STRUM_TWO_KICK_GAP_MS", "175"))
            prev_kick_ms = -1e9
            for idx, h in enumerate(hits_with_ticks):
                if h.lane != 0:
                    continue
                if h.time_ms - prev_kick_ms < two_kick_gap_ms:
                    is_two_kick[idx] = True
                prev_kick_ms = h.time_ms

        for idx, hit in enumerate(hits_with_ticks):
            base_note = _get_midi_note(hit) + note_offset
            # Promote 2nd kick of a fast pair to note 95 on Expert.
            if hit.lane == 0 and is_two_kick.get(idx, False):
                base_note = 95
            tom_marker = _get_tom_marker(hit)  # None or 110/111/112
            velocity = hit.velocity
            
            note_duration = config.ticks_per_beat // 8
            
            # Note on/off for base note (offset per difficulty)
            events.append({
                "type": "note_on",
                "tick": hit.tick,
                "note": base_note,
                "velocity": velocity,
            })
            events.append({
                "type": "note_off",
                "tick": hit.tick + note_duration,
                "note": base_note,
                "velocity": 0,
            })
            
            # Collect tom marker (no offset, deduplicated across difficulties)
            if tom_marker is not None:
                tom_marker_events.add((hit.tick, tom_marker))

    # Emit deduplicated tom markers
    note_duration = config.ticks_per_beat // 8
    for tick, marker in tom_marker_events:
        events.append({
            "type": "note_on",
            "tick": tick,
            "note": marker,
            "velocity": 100,
        })
        events.append({
            "type": "note_off",
            "tick": tick + note_duration,
            "note": marker,
            "velocity": 0,
        })
    
    # Deduplicate base note events: same (tick, note, type) should only
    # appear once.  This can happen when a tom and cymbal hit share the
    # same lane/tick (same base note but different is_cymbal flag).
    seen_events: set[tuple[int, int, str]] = set()
    deduped_events = []
    for event in events:
        key = (event["tick"], event["note"], event["type"])
        if key not in seen_events:
            seen_events.add(key)
            deduped_events.append(event)
    events = deduped_events

    # Sort events by tick
    events.sort(key=lambda e: (e["tick"], e["type"] == "note_off"))
    
    # Convert to MIDI messages with delta times
    prev_tick = 0
    for event in events:
        delta = event["tick"] - prev_tick
        
        if event["type"] == "note_on":
            drums_track.append(mido.Message(
                "note_on",
                note=event["note"],
                velocity=event["velocity"],
                time=delta,
            ))
        else:
            drums_track.append(mido.Message(
                "note_off",
                note=event["note"],
                velocity=0,
                time=delta,
            ))
        
        prev_tick = event["tick"]
    
    drums_track.append(mido.MetaMessage("end_of_track", time=0))
    
    # Create EVENTS track (required by Clone Hero / YARG)
    events_track = mido.MidiTrack()
    mid.tracks.append(events_track)
    events_track.append(mido.MetaMessage("track_name", name="EVENTS", time=0))
    events_track.append(mido.MetaMessage("text", text="[music_start]", time=0))
    # Place [end] at the last note tick + 1 beat
    last_tick = max((e["tick"] for e in events), default=0) + config.ticks_per_beat
    events_track.append(mido.MetaMessage("text", text="[end]", time=last_tick))
    events_track.append(mido.MetaMessage("end_of_track", time=0))
    
    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(output_path)
    
    # Count hits per difficulty
    logger.info(f"Exported multi-difficulty chart to {output_path}")
    for diff, chart in difficulty_charts.items():
        logger.info(f"  {diff.capitalize()}: {len(chart.hits)} hits")


def _reduce_to_hard(chart: DrumChart) -> DrumChart:
    """
    Reduce expert chart to hard difficulty.
    
    Hard (~75% of expert notes):
    - Remove ghost notes (velocity < 60)
    - Remove rapid consecutive hits on same lane (< 80ms)
    - Thin out fast hi-hat patterns
    """
    hits = []
    last_hit_time = {i: -1000.0 for i in range(5)}
    hihat_count = 0
    prev_hihat_time = -1000.0
    
    for hit in sorted(chart.hits, key=lambda h: h.time_ms):
        # Skip ghost notes
        if hit.velocity < 60:
            continue
        
        # Skip very fast repetitions on same lane
        min_gap = 80 if hit.lane != 2 else 100  # Stricter for hi-hat
        if hit.time_ms - last_hit_time[hit.lane] < min_gap:
            continue
        
        # Thin hi-hat: skip every 3rd hit in fast patterns
        if hit.lane == 2:
            if hit.time_ms - prev_hihat_time < 150:
                hihat_count += 1
                if hihat_count % 3 == 0:
                    continue
            else:
                hihat_count = 0
            prev_hihat_time = hit.time_ms
        
        hits.append(hit)
        last_hit_time[hit.lane] = hit.time_ms
    
    return DrumChart(
        hits=hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def _reduce_to_medium(chart: DrumChart) -> DrumChart:
    """
    Reduce hard chart to medium difficulty.
    
    Medium (~50% of expert notes):
    - Simplify kick to basic pulse (no double kicks)
    - Thin hi-hat to half (every other hit)
    - Remove most tom fills (keep only loud accents)
    - Keep snare on backbeats
    """
    hits = []
    prev_kick_time = -1000.0
    prev_snare_time = -1000.0
    prev_hihat_time = -1000.0
    hihat_count = 0
    
    for hit in sorted(chart.hits, key=lambda h: h.time_ms):
        # Kick: minimum 200ms gap (no double kicks)
        if hit.lane == 0:
            if hit.time_ms - prev_kick_time < 200:
                continue
            prev_kick_time = hit.time_ms
        
        # Snare: minimum 150ms gap
        elif hit.lane == 1:
            if hit.time_ms - prev_snare_time < 150:
                continue
            prev_snare_time = hit.time_ms
        
        # Hi-hat: keep every other hit, minimum 180ms gap
        elif hit.lane == 2:
            hihat_count += 1
            if hihat_count % 2 == 0:
                continue
            if hit.time_ms - prev_hihat_time < 180:
                continue
            prev_hihat_time = hit.time_ms
        
        # Toms (lanes 3-4): only keep loud accents (velocity > 80)
        elif hit.lane >= 3:
            if hit.velocity < 80:
                continue
        
        hits.append(hit)
    
    return DrumChart(
        hits=hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def _reduce_to_easy(chart: DrumChart) -> DrumChart:
    """
    Reduce medium chart to easy difficulty.
    
    Easy (~25-30% of expert notes):
    - Only kick and snare (no cymbals, no toms)
    - Very sparse pattern (minimum 300ms between any hits)
    - Focus on main backbeat feel
    """
    hits = []
    last_any_hit_time = -1000.0
    last_lane_time = {i: -1000.0 for i in range(5)}
    
    for hit in sorted(chart.hits, key=lambda h: h.time_ms):
        # Only keep kick and snare
        if hit.lane > 1:
            continue
        
        # Minimum 250ms between hits on same lane
        if hit.time_ms - last_lane_time[hit.lane] < 250:
            continue
        
        # Minimum 150ms between ANY hits (no simultaneous kick+snare in easy)
        if hit.time_ms - last_any_hit_time < 150:
            continue
        
        hits.append(DrumHit(
            time_ms=hit.time_ms,
            tick=hit.tick,
            lane=hit.lane,
            is_cymbal=False,
            velocity=hit.velocity,
        ))
        last_lane_time[hit.lane] = hit.time_ms
        last_any_hit_time = hit.time_ms
    
    return DrumChart(
        hits=hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )
