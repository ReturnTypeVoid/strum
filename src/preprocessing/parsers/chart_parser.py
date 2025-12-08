"""
Chart Parser for Clone Hero .chart format files.

Parses the text-based .chart format used by Clone Hero and YARG.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging
import re

from src.preprocessing.parsers.midi_parser import DrumHit, DrumChart, TempoEvent, TimeSignature

logger = logging.getLogger(__name__)


@dataclass
class ChartMetadata:
    """Song metadata from [Song] section."""
    name: str = ""
    artist: str = ""
    charter: str = ""
    album: str = ""
    year: str = ""
    offset: float = 0.0
    resolution: int = 192
    difficulty: int = 0
    preview_start: float = 0.0
    genre: str = ""


class ChartParser:
    """
    Parser for Clone Hero .chart format.
    
    .chart files are text-based with sections like:
    [Song] - Metadata
    [SyncTrack] - Tempo and time signatures
    [ExpertDrums] - Expert difficulty drum notes
    [HardDrums] - Hard difficulty drum notes
    etc.
    """
    
    # Note types in .chart format
    # N <fret> <length> - Regular note
    # S <type> <length> - Special phrase (star power, etc.)
    # E <text> - Event
    
    # Drum lanes in .chart (different from MIDI!)
    CHART_KICK = 0
    CHART_RED = 1
    CHART_YELLOW = 2
    CHART_BLUE = 3
    CHART_GREEN = 4
    
    # Cymbal markers (notes 66, 67, 68 in some versions, or modifier flags)
    CHART_CYMBAL_YELLOW = 66
    CHART_CYMBAL_BLUE = 67
    CHART_CYMBAL_GREEN = 68
    
    def __init__(self) -> None:
        self._section_pattern = re.compile(r"\[(.+?)\]")
        self._note_pattern = re.compile(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
        self._tempo_pattern = re.compile(r"(\d+)\s*=\s*B\s+(\d+)")
        self._ts_pattern = re.compile(r"(\d+)\s*=\s*TS\s+(\d+)(?:\s+(\d+))?")
        self._metadata_pattern = re.compile(r"(\w+)\s*=\s*(.+)")
    
    def parse(
        self,
        chart_path: Path,
        difficulty: str = "Expert",
    ) -> DrumChart:
        """
        Parse a .chart file and extract drum chart.
        
        Args:
            chart_path: Path to the .chart file
            difficulty: Difficulty level to parse (Expert, Hard, Medium, Easy)
            
        Returns:
            DrumChart with all parsed events
        """
        chart_path = Path(chart_path)
        if not chart_path.exists():
            raise FileNotFoundError(f"Chart file not found: {chart_path}")
        
        logger.info(f"Parsing .chart file: {chart_path}")
        
        # Read file content
        with open(chart_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        
        # Parse sections
        sections = self._split_sections(content)
        
        # Parse metadata
        metadata = self._parse_metadata(sections.get("Song", ""))
        resolution = metadata.resolution
        
        # Parse sync track (tempo and time signatures)
        tempo_events, time_signatures = self._parse_sync_track(
            sections.get("SyncTrack", ""),
            resolution,
        )
        
        # Parse drum track
        drum_section_name = f"{difficulty}Drums"
        drum_content = sections.get(drum_section_name, "")
        
        if not drum_content:
            logger.warning(f"No {drum_section_name} section found in {chart_path}")
            return DrumChart(
                tempo_events=tempo_events,
                time_signatures=time_signatures,
                ticks_per_beat=resolution,
            )
        
        hits = self._parse_drum_section(drum_content, resolution, tempo_events)
        
        logger.info(f"Parsed {len(hits)} drum hits from {chart_path}")
        
        return DrumChart(
            hits=hits,
            tempo_events=tempo_events,
            time_signatures=time_signatures,
            ticks_per_beat=resolution,
        )
    
    def _split_sections(self, content: str) -> dict[str, str]:
        """Split .chart content into sections."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split("\n"):
            line = line.strip()
            
            # Check for section header
            match = self._section_pattern.match(line)
            if match:
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                
                current_section = match.group(1)
                current_content = []
            elif line == "{":
                continue
            elif line == "}":
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                current_section = None
                current_content = []
            elif current_section:
                current_content.append(line)
        
        return sections
    
    def _parse_metadata(self, content: str) -> ChartMetadata:
        """Parse [Song] section for metadata."""
        metadata = ChartMetadata()
        
        for line in content.split("\n"):
            match = self._metadata_pattern.match(line.strip())
            if match:
                key = match.group(1).lower()
                value = match.group(2).strip().strip('"')
                
                if key == "name":
                    metadata.name = value
                elif key == "artist":
                    metadata.artist = value
                elif key == "charter":
                    metadata.charter = value
                elif key == "album":
                    metadata.album = value
                elif key == "year":
                    metadata.year = value
                elif key == "offset":
                    metadata.offset = float(value)
                elif key == "resolution":
                    metadata.resolution = int(value)
                elif key == "difficulty":
                    metadata.difficulty = int(value)
                elif key == "previewstart":
                    metadata.preview_start = float(value)
                elif key == "genre":
                    metadata.genre = value
        
        return metadata
    
    def _parse_sync_track(
        self,
        content: str,
        resolution: int,
    ) -> tuple[list[TempoEvent], list[TimeSignature]]:
        """Parse [SyncTrack] for tempo and time signatures."""
        tempo_events = []
        time_signatures = []
        
        for line in content.split("\n"):
            line = line.strip()
            
            # Parse tempo (B = BPM * 1000)
            tempo_match = self._tempo_pattern.match(line)
            if tempo_match:
                tick = int(tempo_match.group(1))
                tempo_millibpm = int(tempo_match.group(2))
                tempo_bpm = tempo_millibpm / 1000.0
                
                tempo_events.append(TempoEvent(
                    tick=tick,
                    tempo_bpm=tempo_bpm,
                ))
            
            # Parse time signature
            ts_match = self._ts_pattern.match(line)
            if ts_match:
                tick = int(ts_match.group(1))
                numerator = int(ts_match.group(2))
                # Denominator is optional, defaults to 4
                denominator = int(ts_match.group(3)) if ts_match.group(3) else 4
                denominator = 2 ** denominator  # Convert from power of 2
                
                time_signatures.append(TimeSignature(
                    tick=tick,
                    numerator=numerator,
                    denominator=denominator,
                ))
        
        # Sort and set defaults
        tempo_events.sort(key=lambda e: e.tick)
        time_signatures.sort(key=lambda e: e.tick)
        
        if not tempo_events:
            tempo_events.append(TempoEvent(tick=0, tempo_bpm=120.0))
        
        if not time_signatures:
            time_signatures.append(TimeSignature(tick=0, numerator=4, denominator=4))
        
        # Calculate time_ms for each event
        for i, event in enumerate(tempo_events):
            event.time_ms = self._ticks_to_ms(event.tick, resolution, tempo_events[:i] or tempo_events[:1])
        
        for event in time_signatures:
            event.time_ms = self._ticks_to_ms(event.tick, resolution, tempo_events)
        
        return tempo_events, time_signatures
    
    def _parse_drum_section(
        self,
        content: str,
        resolution: int,
        tempo_events: list[TempoEvent],
    ) -> list[DrumHit]:
        """Parse drum notes from a difficulty section."""
        hits = []
        cymbal_ticks: set[tuple[int, int]] = set()  # (tick, lane) pairs with cymbal markers
        
        # First pass: collect cymbal markers
        for line in content.split("\n"):
            line = line.strip()
            match = self._note_pattern.match(line)
            if match:
                tick = int(match.group(1))
                note = int(match.group(2))
                
                # Cymbal markers
                if note == self.CHART_CYMBAL_YELLOW:
                    cymbal_ticks.add((tick, DrumHit.YELLOW))
                elif note == self.CHART_CYMBAL_BLUE:
                    cymbal_ticks.add((tick, DrumHit.BLUE))
                elif note == self.CHART_CYMBAL_GREEN:
                    cymbal_ticks.add((tick, DrumHit.GREEN))
        
        # Second pass: parse actual drum notes
        for line in content.split("\n"):
            line = line.strip()
            match = self._note_pattern.match(line)
            if match:
                tick = int(match.group(1))
                note = int(match.group(2))
                
                # Skip cymbal markers
                if note in (self.CHART_CYMBAL_YELLOW, self.CHART_CYMBAL_BLUE, self.CHART_CYMBAL_GREEN):
                    continue
                
                # Map note to lane
                if note == self.CHART_KICK:
                    lane = DrumHit.KICK
                elif note == self.CHART_RED:
                    lane = DrumHit.RED
                elif note == self.CHART_YELLOW:
                    lane = DrumHit.YELLOW
                elif note == self.CHART_BLUE:
                    lane = DrumHit.BLUE
                elif note == self.CHART_GREEN:
                    lane = DrumHit.GREEN
                else:
                    continue  # Unknown note, skip
                
                time_ms = self._ticks_to_ms(tick, resolution, tempo_events)
                is_cymbal = (tick, lane) in cymbal_ticks
                
                hits.append(DrumHit(
                    time_ms=time_ms,
                    tick=tick,
                    lane=lane,
                    is_cymbal=is_cymbal,
                    velocity=100,  # .chart doesn't have velocity, default to 100
                ))
        
        # Sort by time
        hits.sort(key=lambda h: h.time_ms)
        
        return hits
    
    def _ticks_to_ms(
        self,
        tick: int,
        resolution: int,
        tempo_events: list[TempoEvent],
    ) -> float:
        """Convert tick position to milliseconds using tempo map."""
        if not tempo_events:
            # Default 120 BPM
            return (tick / resolution) * 500.0
        
        time_ms = 0.0
        prev_tick = 0
        prev_tempo_bpm = tempo_events[0].tempo_bpm
        
        for tempo_event in tempo_events:
            if tempo_event.tick >= tick:
                break
            
            # Add time from previous tempo region
            ticks_in_region = tempo_event.tick - prev_tick
            ms_per_tick = (60_000 / prev_tempo_bpm) / resolution
            time_ms += ticks_in_region * ms_per_tick
            
            prev_tick = tempo_event.tick
            prev_tempo_bpm = tempo_event.tempo_bpm
        
        # Add remaining ticks at current tempo
        remaining_ticks = tick - prev_tick
        ms_per_tick = (60_000 / prev_tempo_bpm) / resolution
        time_ms += remaining_ticks * ms_per_tick
        
        return time_ms
