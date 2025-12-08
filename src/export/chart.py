"""
.chart format export for STRUM.

Exports charts to the .chart text format used by Clone Hero/YARG.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

from src.preprocessing.parsers.midi_parser import DrumChart, DrumHit, TempoEvent, TimeSignature

logger = logging.getLogger(__name__)


# Clone Hero .chart drum note values
# In .chart format, drums use note values 0-4 for the 5 lanes
# Cymbals are represented by additional flags (66 = yellow cymbal, 67 = blue, 68 = green)
CHART_DRUM_NOTES = {
    0: 0,   # Kick
    1: 1,   # Red (snare)
    2: 2,   # Yellow
    3: 3,   # Blue
    4: 4,   # Green (or Orange for 5-lane)
}

CHART_CYMBAL_FLAGS = {
    2: 66,  # Yellow cymbal
    3: 67,  # Blue cymbal
    4: 68,  # Green cymbal
}


@dataclass
class ChartExportConfig:
    """Configuration for .chart export."""
    
    resolution: int = 192  # Ticks per beat
    offset_ms: float = 0.0  # Audio offset
    name: str = "Unknown"
    artist: str = "Unknown"
    charter: str = "STRUM"
    difficulty: int = 0  # 0-6 difficulty rating


def export_drums_chart(
    chart: DrumChart,
    output_path: Path,
    config: Optional[ChartExportConfig] = None,
    difficulty: str = "expert",
) -> None:
    """
    Export a drum chart to .chart format.
    
    Args:
        chart: DrumChart with hits and tempo info
        output_path: Path to write .chart file
        config: Export configuration
        difficulty: Difficulty level (expert/hard/medium/easy)
    """
    if config is None:
        config = ChartExportConfig()
    
    lines = []
    
    # [Song] section
    lines.append("[Song]")
    lines.append("{")
    lines.append(f'  Name = "{config.name}"')
    lines.append(f'  Artist = "{config.artist}"')
    lines.append(f'  Charter = "{config.charter}"')
    lines.append(f"  Offset = {config.offset_ms / 1000:.3f}")
    lines.append(f"  Resolution = {config.resolution}")
    lines.append(f"  Difficulty = {config.difficulty}")
    lines.append('  PreviewStart = 0')
    lines.append('  PreviewEnd = 0')
    lines.append('  Genre = "Unknown"')
    lines.append('  MediaType = "cd"')
    lines.append("}")
    lines.append("")
    
    # [SyncTrack] section
    lines.append("[SyncTrack]")
    lines.append("{")
    
    # Add tempo events
    if chart.tempo_events:
        for te in sorted(chart.tempo_events, key=lambda t: t.tick):
            # Convert BPM to .chart format (BPM * 1000)
            bpm_value = int(te.tempo_bpm * 1000)
            tick = _convert_tick(te.tick, chart.ticks_per_beat, config.resolution)
            lines.append(f"  {tick} = B {bpm_value}")
    else:
        lines.append("  0 = B 120000")  # Default 120 BPM
    
    # Add time signatures
    if chart.time_signatures:
        for ts in sorted(chart.time_signatures, key=lambda t: t.tick):
            tick = _convert_tick(ts.tick, chart.ticks_per_beat, config.resolution)
            lines.append(f"  {tick} = TS {ts.numerator}")
    else:
        lines.append("  0 = TS 4")  # Default 4/4
    
    lines.append("}")
    lines.append("")
    
    # [Events] section (empty for now)
    lines.append("[Events]")
    lines.append("{")
    lines.append("}")
    lines.append("")
    
    # Drum track section
    section_name = _get_chart_section_name(difficulty)
    lines.append(f"[{section_name}]")
    lines.append("{")
    
    # Convert hits to .chart format
    hits_with_ticks = _compute_hit_ticks_chart(chart, config.resolution)
    
    for hit in sorted(hits_with_ticks, key=lambda h: h.tick):
        note_value = CHART_DRUM_NOTES[hit.lane]
        tick = _convert_tick(hit.tick, chart.ticks_per_beat, config.resolution)
        
        # Note event: tick = N note_value 0 (0 = note length, drums are instant)
        lines.append(f"  {tick} = N {note_value} 0")
        
        # Add cymbal flag if applicable
        if hit.is_cymbal and hit.lane in CHART_CYMBAL_FLAGS:
            cymbal_flag = CHART_CYMBAL_FLAGS[hit.lane]
            lines.append(f"  {tick} = N {cymbal_flag} 0")
    
    lines.append("}")
    
    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Exported .chart with {len(hits_with_ticks)} drum hits to {output_path}")


def _get_chart_section_name(difficulty: str) -> str:
    """Get the .chart section name for a difficulty level."""
    mapping = {
        "expert": "ExpertDrums",
        "hard": "HardDrums",
        "medium": "MediumDrums",
        "easy": "EasyDrums",
    }
    return mapping.get(difficulty.lower(), "ExpertDrums")


def _convert_tick(tick: int, source_tpb: int, target_resolution: int) -> int:
    """Convert tick from source ticks-per-beat to target resolution."""
    if source_tpb == target_resolution:
        return tick
    return int(tick * target_resolution / source_tpb)


def _compute_hit_ticks_chart(
    chart: DrumChart,
    resolution: int,
) -> list[DrumHit]:
    """
    Convert hit times from ms to ticks for .chart export.
    """
    if not chart.hits:
        return []
    
    # Build tempo map
    tempo_map = []
    if chart.tempo_events:
        for te in sorted(chart.tempo_events, key=lambda t: t.time_ms):
            # Convert original tick to target resolution
            tick = _convert_tick(te.tick, chart.ticks_per_beat, resolution)
            tempo_map.append((te.time_ms, tick, te.tempo_bpm))
    else:
        tempo_map.append((0.0, 0, 120.0))
    
    hits_with_ticks = []
    
    for hit in chart.hits:
        # Find tempo segment
        segment_idx = 0
        for i, (time_ms, _, _) in enumerate(tempo_map):
            if time_ms <= hit.time_ms:
                segment_idx = i
            else:
                break
        
        segment_start_ms, segment_start_tick, tempo_bpm = tempo_map[segment_idx]
        
        # Compute ticks
        ms_per_beat = 60_000 / tempo_bpm
        ms_per_tick = ms_per_beat / resolution
        
        delta_ms = hit.time_ms - segment_start_ms
        delta_ticks = int(delta_ms / ms_per_tick)
        tick = segment_start_tick + delta_ticks
        
        hits_with_ticks.append(DrumHit(
            time_ms=hit.time_ms,
            tick=tick,
            lane=hit.lane,
            is_cymbal=hit.is_cymbal,
            velocity=hit.velocity,
        ))
    
    return hits_with_ticks


def export_full_chart_file(
    drums_chart: DrumChart,
    output_path: Path,
    song_info: Optional[dict] = None,
) -> None:
    """
    Export a complete .chart file with all difficulties.
    
    Args:
        drums_chart: Expert-level drum chart
        output_path: Output path for .chart file
        song_info: Optional dict with name, artist, etc.
    """
    from src.export.midi import _reduce_to_hard, _reduce_to_medium, _reduce_to_easy
    
    song_info = song_info or {}
    
    config = ChartExportConfig(
        name=song_info.get("name", "Unknown"),
        artist=song_info.get("artist", "Unknown"),
        charter=song_info.get("charter", "STRUM"),
    )
    
    lines = []
    
    # [Song] section
    lines.append("[Song]")
    lines.append("{")
    lines.append(f'  Name = "{config.name}"')
    lines.append(f'  Artist = "{config.artist}"')
    lines.append(f'  Charter = "{config.charter}"')
    lines.append(f"  Offset = 0")
    lines.append(f"  Resolution = {config.resolution}")
    lines.append(f"  Difficulty = 0")
    lines.append('  PreviewStart = 0')
    lines.append('  PreviewEnd = 0')
    lines.append('  Genre = "Unknown"')
    lines.append('  MediaType = "cd"')
    lines.append("}")
    lines.append("")
    
    # [SyncTrack]
    lines.append("[SyncTrack]")
    lines.append("{")
    if drums_chart.tempo_events:
        for te in sorted(drums_chart.tempo_events, key=lambda t: t.tick):
            bpm_value = int(te.tempo_bpm * 1000)
            tick = _convert_tick(te.tick, drums_chart.ticks_per_beat, config.resolution)
            lines.append(f"  {tick} = B {bpm_value}")
    else:
        lines.append("  0 = B 120000")
    
    if drums_chart.time_signatures:
        for ts in sorted(drums_chart.time_signatures, key=lambda t: t.tick):
            tick = _convert_tick(ts.tick, drums_chart.ticks_per_beat, config.resolution)
            lines.append(f"  {tick} = TS {ts.numerator}")
    else:
        lines.append("  0 = TS 4")
    lines.append("}")
    lines.append("")
    
    # [Events]
    lines.append("[Events]")
    lines.append("{")
    lines.append("}")
    lines.append("")
    
    # Generate all difficulties
    difficulties = [
        ("ExpertDrums", drums_chart),
        ("HardDrums", _reduce_to_hard(drums_chart)),
        ("MediumDrums", _reduce_to_medium(_reduce_to_hard(drums_chart))),
        ("EasyDrums", _reduce_to_easy(_reduce_to_medium(_reduce_to_hard(drums_chart)))),
    ]
    
    for section_name, diff_chart in difficulties:
        lines.append(f"[{section_name}]")
        lines.append("{")
        
        hits_with_ticks = _compute_hit_ticks_chart(diff_chart, config.resolution)
        
        for hit in sorted(hits_with_ticks, key=lambda h: h.tick):
            note_value = CHART_DRUM_NOTES[hit.lane]
            tick = _convert_tick(hit.tick, diff_chart.ticks_per_beat, config.resolution)
            lines.append(f"  {tick} = N {note_value} 0")
            
            if hit.is_cymbal and hit.lane in CHART_CYMBAL_FLAGS:
                cymbal_flag = CHART_CYMBAL_FLAGS[hit.lane]
                lines.append(f"  {tick} = N {cymbal_flag} 0")
        
        lines.append("}")
        lines.append("")
    
    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    logger.info(f"Exported full .chart file to {output_path}")
