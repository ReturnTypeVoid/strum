"""
MIDI Parser for Clone Hero / YARG chart files.

Parses Standard MIDI Files (SMF) and extracts drum tracks with pro drums mapping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

import mido

logger = logging.getLogger(__name__)


@dataclass
class TempoEvent:
    """Tempo change event."""
    tick: int
    tempo_bpm: float
    time_ms: float = 0.0


@dataclass
class TimeSignature:
    """Time signature event."""
    tick: int
    numerator: int
    denominator: int
    time_ms: float = 0.0


@dataclass
class DrumHit:
    """Single drum hit event."""
    time_ms: float
    tick: int
    lane: int  # 0=kick, 1=red, 2=yellow, 3=blue, 4=green
    is_cymbal: bool
    velocity: int
    
    # Lane constants
    KICK = 0
    RED = 1      # Snare
    YELLOW = 2   # Hi-hat / High tom
    BLUE = 3     # Ride / Mid tom
    GREEN = 4    # Crash / Floor tom


@dataclass
class DrumChart:
    """Parsed drum chart with all events."""
    hits: list[DrumHit] = field(default_factory=list)
    tempo_events: list[TempoEvent] = field(default_factory=list)
    time_signatures: list[TimeSignature] = field(default_factory=list)
    ticks_per_beat: int = 480
    
    def get_duration_ms(self) -> float:
        """Get total duration in milliseconds."""
        if not self.hits:
            return 0.0
        return max(hit.time_ms for hit in self.hits)


class MidiParser:
    """
    Parser for MIDI chart files.
    
    Supports two drum formats:
    
    1. Rock Band Pro Drums (Expert difficulty):
       - Kick: 96
       - Red (Snare): 97
       - Yellow: 98 (default=cymbal, tom with 110 marker)
       - Blue: 99 (default=cymbal, tom with 111 marker)
       - Green: 100 (default=cymbal, tom with 112 marker)
       Note: Yellow/Blue/Green default to CYMBALS. The 110/111/112 markers
       indicate TOMS (inverted from what you might expect).
    
    2. Guitar Hero 5-Lane Drums (World Tour, GH5, etc.):
       - Kick: 96
       - Red (Snare): 97
       - Yellow: 98 (Hi-hat = CYMBAL)
       - Blue: 99 (Tom = TOM)
       - Green: 100 (Floor tom = TOM)
       - Orange: 101 (Crash cymbal = CYMBAL, maps to Green lane)
       Note: In 5-lane mode, cymbal/tom is determined by the lane itself,
       NOT by presence/absence of 110-112 markers.
    """
    
    # Pro drums MIDI note mapping (Expert difficulty)
    NOTE_KICK = 96
    NOTE_RED = 97
    NOTE_YELLOW = 98
    NOTE_BLUE = 99
    NOTE_GREEN = 100
    NOTE_ORANGE = 101  # GH 5-lane crash cymbal
    
    # Tom markers - presence indicates TOM, absence indicates CYMBAL (Rock Band style)
    TOM_YELLOW = 110
    TOM_BLUE = 111
    TOM_GREEN = 112
    
    # Track names to search for drums
    DRUM_TRACK_NAMES = [
        "PART DRUMS",
        "PART REAL_DRUMS_PS",
        "drums",
        "Drums",
        "DRUMS",
    ]
    
    def __init__(self) -> None:
        self._note_to_lane = {
            self.NOTE_KICK: DrumHit.KICK,
            self.NOTE_RED: DrumHit.RED,
            self.NOTE_YELLOW: DrumHit.YELLOW,
            self.NOTE_BLUE: DrumHit.BLUE,
            self.NOTE_GREEN: DrumHit.GREEN,
            self.NOTE_ORANGE: DrumHit.GREEN,  # Orange maps to Green lane
        }
        self._tom_markers = {
            self.TOM_YELLOW,
            self.TOM_BLUE,
            self.TOM_GREEN,
        }
        self._tom_to_lane = {
            self.TOM_YELLOW: DrumHit.YELLOW,
            self.TOM_BLUE: DrumHit.BLUE,
            self.TOM_GREEN: DrumHit.GREEN,
        }
    
    def parse(self, midi_path: Path, five_lane_drums: bool = False) -> DrumChart:
        """
        Parse a MIDI file and extract drum chart.
        
        Args:
            midi_path: Path to the MIDI file
            five_lane_drums: If True, use GH 5-lane mapping where Yellow=cymbal,
                           Blue/Green=tom, Orange=cymbal. If False, use Rock Band
                           pro drums mapping with 110-112 tom markers.
            
        Returns:
            DrumChart with all parsed events
        """
        midi_path = Path(midi_path)
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        logger.info(f"Parsing MIDI file: {midi_path} (five_lane={five_lane_drums})")
        
        midi = mido.MidiFile(midi_path)
        ticks_per_beat = midi.ticks_per_beat
        
        # Extract tempo and time signature events
        tempo_events = self._extract_tempo_events(midi, ticks_per_beat)
        time_signatures = self._extract_time_signatures(midi, ticks_per_beat, tempo_events)
        
        # Find and parse drum track
        drum_track = self._find_drum_track(midi)
        if drum_track is None:
            logger.warning(f"No drum track found in {midi_path}")
            return DrumChart(
                tempo_events=tempo_events,
                time_signatures=time_signatures,
                ticks_per_beat=ticks_per_beat,
            )
        
        # Parse drum hits
        hits = self._parse_drum_track(drum_track, ticks_per_beat, tempo_events, five_lane_drums)
        
        logger.info(f"Parsed {len(hits)} drum hits from {midi_path}")
        
        return DrumChart(
            hits=hits,
            tempo_events=tempo_events,
            time_signatures=time_signatures,
            ticks_per_beat=ticks_per_beat,
        )
    
    def _extract_tempo_events(
        self,
        midi: mido.MidiFile,
        ticks_per_beat: int,
    ) -> list[TempoEvent]:
        """Extract all tempo change events."""
        tempo_events = []
        current_tick = 0
        current_time_ms = 0.0
        current_tempo_us = 500000  # Default: 120 BPM
        
        # Merge all tracks for tempo events
        for track in midi.tracks:
            tick = 0
            for msg in track:
                tick += msg.time
                if msg.type == "set_tempo":
                    # Calculate time up to this point
                    time_ms = self._ticks_to_ms(tick, ticks_per_beat, tempo_events)
                    tempo_bpm = 60_000_000 / msg.tempo
                    
                    tempo_events.append(TempoEvent(
                        tick=tick,
                        tempo_bpm=tempo_bpm,
                        time_ms=time_ms,
                    ))
        
        # Sort by tick and remove duplicates
        tempo_events.sort(key=lambda e: e.tick)
        
        # If no tempo events, add default
        if not tempo_events:
            tempo_events.append(TempoEvent(tick=0, tempo_bpm=120.0, time_ms=0.0))
        
        return tempo_events
    
    def _extract_time_signatures(
        self,
        midi: mido.MidiFile,
        ticks_per_beat: int,
        tempo_events: list[TempoEvent],
    ) -> list[TimeSignature]:
        """Extract all time signature events."""
        time_sigs = []
        
        for track in midi.tracks:
            tick = 0
            for msg in track:
                tick += msg.time
                if msg.type == "time_signature":
                    time_ms = self._ticks_to_ms(tick, ticks_per_beat, tempo_events)
                    time_sigs.append(TimeSignature(
                        tick=tick,
                        numerator=msg.numerator,
                        denominator=msg.denominator,
                        time_ms=time_ms,
                    ))
        
        # Sort and default to 4/4
        time_sigs.sort(key=lambda e: e.tick)
        if not time_sigs:
            time_sigs.append(TimeSignature(tick=0, numerator=4, denominator=4, time_ms=0.0))
        
        return time_sigs
    
    def _find_drum_track(self, midi: mido.MidiFile) -> Optional[mido.MidiTrack]:
        """Find the drum track by name or content."""
        # First, search by track name
        for track in midi.tracks:
            if track.name in self.DRUM_TRACK_NAMES:
                return track
        
        # Fallback: search for track with drum notes
        for track in midi.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.note in self._note_to_lane:
                    return track
        
        return None
    
    def _parse_drum_track(
        self,
        track: mido.MidiTrack,
        ticks_per_beat: int,
        tempo_events: list[TempoEvent],
        five_lane_drums: bool = False,
    ) -> list[DrumHit]:
        """
        Parse drum hits from a track.
        
        Args:
            track: MIDI track to parse
            ticks_per_beat: MIDI resolution
            tempo_events: Tempo map for timing conversion
            five_lane_drums: If True, use GH 5-lane cymbal/tom mapping
        """
        from collections import defaultdict
        
        hits = []
        
        # First pass: collect all note_on events by tick
        events_by_tick: dict[int, list[tuple[int, int]]] = defaultdict(list)  # tick -> [(note, velocity)]
        current_tick = 0
        
        for msg in track:
            current_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                events_by_tick[current_tick].append((msg.note, msg.velocity))
        
        # Second pass: process each tick, looking for drum notes and tom markers together
        for tick in sorted(events_by_tick.keys()):
            notes_at_tick = events_by_tick[tick]
            note_values = {n for n, v in notes_at_tick}
            
            # Check which tom markers are active at this tick (Rock Band style)
            # Tom markers mean TOM, absence means CYMBAL
            has_yellow_tom = self.TOM_YELLOW in note_values
            has_blue_tom = self.TOM_BLUE in note_values
            has_green_tom = self.TOM_GREEN in note_values
            
            time_ms = self._ticks_to_ms(tick, ticks_per_beat, tempo_events)
            
            for note, velocity in notes_at_tick:
                if note not in self._note_to_lane:
                    continue
                
                lane = self._note_to_lane[note]
                
                # Determine cymbal vs tom based on format
                is_cymbal = False
                
                if five_lane_drums:
                    # GH 5-Lane format:
                    # - Yellow (98) = Hi-hat = ALWAYS CYMBAL
                    # - Blue (99) = Tom = ALWAYS TOM
                    # - Green (100) = Floor tom = ALWAYS TOM
                    # - Orange (101) = Crash = ALWAYS CYMBAL
                    if note == self.NOTE_YELLOW:
                        is_cymbal = True
                    elif note == self.NOTE_BLUE:
                        is_cymbal = False
                    elif note == self.NOTE_GREEN:
                        is_cymbal = False
                    elif note == self.NOTE_ORANGE:
                        is_cymbal = True  # Orange is always cymbal
                else:
                    # Rock Band Pro Drums format:
                    # - Kick (lane 0) and Red/Snare (lane 1): never cymbals
                    # - Yellow/Blue/Green (lanes 2-4): cymbal by DEFAULT, tom if marker present
                    if lane == DrumHit.YELLOW:
                        is_cymbal = not has_yellow_tom  # No marker = cymbal
                    elif lane == DrumHit.BLUE:
                        is_cymbal = not has_blue_tom
                    elif lane == DrumHit.GREEN:
                        is_cymbal = not has_green_tom
                
                hits.append(DrumHit(
                    time_ms=time_ms,
                    tick=tick,
                    lane=lane,
                    is_cymbal=is_cymbal,
                    velocity=velocity,
                ))
        
        # Sort by time
        hits.sort(key=lambda h: h.time_ms)
        
        return hits
    
    def _ticks_to_ms(
        self,
        tick: int,
        ticks_per_beat: int,
        tempo_events: list[TempoEvent],
    ) -> float:
        """Convert tick position to milliseconds using tempo map."""
        if not tempo_events:
            # Default 120 BPM
            return (tick / ticks_per_beat) * 500.0
        
        time_ms = 0.0
        prev_tick = 0
        prev_tempo_bpm = tempo_events[0].tempo_bpm
        
        for tempo_event in tempo_events:
            if tempo_event.tick >= tick:
                break
            
            # Add time from previous tempo region
            ticks_in_region = tempo_event.tick - prev_tick
            ms_per_tick = (60_000 / prev_tempo_bpm) / ticks_per_beat
            time_ms += ticks_in_region * ms_per_tick
            
            prev_tick = tempo_event.tick
            prev_tempo_bpm = tempo_event.tempo_bpm
        
        # Add remaining ticks at current tempo
        remaining_ticks = tick - prev_tick
        ms_per_tick = (60_000 / prev_tempo_bpm) / ticks_per_beat
        time_ms += remaining_ticks * ms_per_tick
        
        return time_ms
