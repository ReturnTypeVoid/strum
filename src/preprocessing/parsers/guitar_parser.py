"""
Guitar/Bass MIDI Parser for Clone Hero / YARG chart files.

Parses guitar and bass tracks from Standard MIDI Files.

Clone Hero Guitar/Bass note mapping (Expert):
- Green: 96
- Red: 97
- Yellow: 98
- Blue: 99
- Orange: 100

Other difficulties use offsets:
- Hard: 84-88
- Medium: 72-76  
- Easy: 60-64

Additional markers:
- Open note (strum without fret): varies by game
- Star Power: 116
- Solo marker: 103
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

import mido

logger = logging.getLogger(__name__)


@dataclass
class GuitarNote:
    """Single guitar/bass note event."""
    time_ms: float
    tick: int
    fret: int  # 0=Green, 1=Red, 2=Yellow, 3=Blue, 4=Orange
    duration_ms: float = 0.0
    duration_ticks: int = 0
    velocity: int = 100
    is_hopo: bool = False  # Hammer-on/Pull-off
    
    # Fret constants
    GREEN = 0
    RED = 1
    YELLOW = 2
    BLUE = 3
    ORANGE = 4


@dataclass
class GuitarChart:
    """Parsed guitar or bass chart."""
    notes: list[GuitarNote] = field(default_factory=list)
    tempo_events: list = field(default_factory=list)
    time_signatures: list = field(default_factory=list)
    ticks_per_beat: int = 480
    instrument: str = "guitar"  # "guitar" or "bass"
    
    def get_duration_ms(self) -> float:
        """Get total duration in milliseconds."""
        if not self.notes:
            return 0.0
        return max(n.time_ms + n.duration_ms for n in self.notes)


class GuitarParser:
    """
    Parser for guitar/bass MIDI tracks.
    
    Clone Hero Expert guitar mapping:
    - Green: 96
    - Red: 97
    - Yellow: 98
    - Blue: 99
    - Orange: 100
    """
    
    # Expert difficulty note mapping
    NOTE_GREEN = 96
    NOTE_RED = 97
    NOTE_YELLOW = 98
    NOTE_BLUE = 99
    NOTE_ORANGE = 100
    
    # Track names
    GUITAR_TRACK_NAMES = [
        "PART GUITAR",
        "T1 GEMS",  # Legacy GH format
        "PART GUITAR GHL",  # Guitar Hero Live
    ]
    
    BASS_TRACK_NAMES = [
        "PART BASS",
        "PART RHYTHM",  # Some charts use this
    ]
    
    def __init__(self) -> None:
        self._note_to_fret = {
            self.NOTE_GREEN: GuitarNote.GREEN,
            self.NOTE_RED: GuitarNote.RED,
            self.NOTE_YELLOW: GuitarNote.YELLOW,
            self.NOTE_BLUE: GuitarNote.BLUE,
            self.NOTE_ORANGE: GuitarNote.ORANGE,
        }
    
    def parse(self, midi_path: Path, instrument: str = "guitar") -> GuitarChart:
        """
        Parse a MIDI file and extract guitar or bass chart.
        
        Args:
            midi_path: Path to the MIDI file
            instrument: "guitar" or "bass"
            
        Returns:
            GuitarChart with all parsed events
        """
        midi_path = Path(midi_path)
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        midi = mido.MidiFile(midi_path)
        ticks_per_beat = midi.ticks_per_beat
        
        # Extract tempo and time signature events
        tempo_events = self._extract_tempo_events(midi, ticks_per_beat)
        time_signatures = self._extract_time_signatures(midi, ticks_per_beat, tempo_events)
        
        # Find appropriate track
        track_names = self.GUITAR_TRACK_NAMES if instrument == "guitar" else self.BASS_TRACK_NAMES
        track = self._find_track(midi, track_names)
        
        if track is None:
            logger.warning(f"No {instrument} track found in {midi_path}")
            return GuitarChart(
                tempo_events=tempo_events,
                time_signatures=time_signatures,
                ticks_per_beat=ticks_per_beat,
                instrument=instrument,
            )
        
        # Parse notes
        notes = self._parse_track(track, ticks_per_beat, tempo_events)
        
        logger.info(f"Parsed {len(notes)} {instrument} notes from {midi_path}")
        
        return GuitarChart(
            notes=notes,
            tempo_events=tempo_events,
            time_signatures=time_signatures,
            ticks_per_beat=ticks_per_beat,
            instrument=instrument,
        )
    
    def _extract_tempo_events(self, midi: mido.MidiFile, ticks_per_beat: int) -> list:
        """Extract all tempo change events."""
        from src.preprocessing.parsers.midi_parser import TempoEvent
        
        tempo_events = []
        
        for track in midi.tracks:
            tick = 0
            for msg in track:
                tick += msg.time
                if msg.type == "set_tempo":
                    time_ms = self._ticks_to_ms(tick, ticks_per_beat, tempo_events)
                    tempo_bpm = 60_000_000 / msg.tempo
                    tempo_events.append(TempoEvent(
                        tick=tick,
                        tempo_bpm=tempo_bpm,
                        time_ms=time_ms,
                    ))
        
        tempo_events.sort(key=lambda e: e.tick)
        
        if not tempo_events:
            from src.preprocessing.parsers.midi_parser import TempoEvent
            tempo_events.append(TempoEvent(tick=0, tempo_bpm=120.0, time_ms=0.0))
        
        return tempo_events
    
    def _extract_time_signatures(self, midi: mido.MidiFile, ticks_per_beat: int, tempo_events: list) -> list:
        """Extract all time signature events."""
        from src.preprocessing.parsers.midi_parser import TimeSignature
        
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
        
        time_sigs.sort(key=lambda e: e.tick)
        
        if not time_sigs:
            time_sigs.append(TimeSignature(tick=0, numerator=4, denominator=4, time_ms=0.0))
        
        return time_sigs
    
    def _find_track(self, midi: mido.MidiFile, track_names: list) -> Optional[mido.MidiTrack]:
        """Find track by name."""
        for track in midi.tracks:
            if track.name in track_names:
                return track
        
        # Fallback: search for track with guitar notes
        for track in midi.tracks:
            for msg in track:
                if msg.type == "note_on" and msg.note in self._note_to_fret:
                    return track
        
        return None
    
    def _parse_track(
        self,
        track: mido.MidiTrack,
        ticks_per_beat: int,
        tempo_events: list,
    ) -> list[GuitarNote]:
        """Parse guitar notes from track."""
        notes = []
        active_notes = {}  # note_num -> (start_tick, velocity)
        current_tick = 0
        
        for msg in track:
            current_tick += msg.time
            
            if msg.type == "note_on" and msg.velocity > 0:
                if msg.note in self._note_to_fret:
                    active_notes[msg.note] = (current_tick, msg.velocity)
                    
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active_notes:
                    start_tick, velocity = active_notes.pop(msg.note)
                    
                    fret = self._note_to_fret[msg.note]
                    start_ms = self._ticks_to_ms(start_tick, ticks_per_beat, tempo_events)
                    end_ms = self._ticks_to_ms(current_tick, ticks_per_beat, tempo_events)
                    duration_ms = end_ms - start_ms
                    duration_ticks = current_tick - start_tick
                    
                    notes.append(GuitarNote(
                        time_ms=start_ms,
                        tick=start_tick,
                        fret=fret,
                        duration_ms=duration_ms,
                        duration_ticks=duration_ticks,
                        velocity=velocity,
                    ))
        
        # Sort by time
        notes.sort(key=lambda n: (n.time_ms, n.fret))
        
        return notes
    
    def _ticks_to_ms(self, tick: int, ticks_per_beat: int, tempo_events: list) -> float:
        """Convert tick position to milliseconds."""
        if not tempo_events:
            return (tick / ticks_per_beat) * 500.0
        
        time_ms = 0.0
        prev_tick = 0
        prev_tempo_bpm = tempo_events[0].tempo_bpm
        
        for tempo_event in tempo_events:
            if tempo_event.tick >= tick:
                break
            
            ticks_in_region = tempo_event.tick - prev_tick
            ms_per_tick = (60_000 / prev_tempo_bpm) / ticks_per_beat
            time_ms += ticks_in_region * ms_per_tick
            
            prev_tick = tempo_event.tick
            prev_tempo_bpm = tempo_event.tempo_bpm
        
        remaining_ticks = tick - prev_tick
        ms_per_tick = (60_000 / prev_tempo_bpm) / ticks_per_beat
        time_ms += remaining_ticks * ms_per_tick
        
        return time_ms
