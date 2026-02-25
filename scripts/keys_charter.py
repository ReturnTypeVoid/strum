"""
Keys charter for Clone Hero/YARG - generates 5-lane keys and Pro Keys tracks.

Only generates keys tracks when actual keyboard instruments (piano, synth, organ)
are detected in the audio to avoid duplicating guitar content.

Clone Hero/YARG Keys Format:
- PART KEYS: 5-lane simplified keys (like Guitar Hero)
- PART REAL_KEYS_X: Pro Keys with 25-key range (X = E/M/H/X difficulty)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List
import tempfile

import numpy as np
import librosa
import mido

logger = logging.getLogger(__name__)


@dataclass
class KeyNote:
    """A single key note."""
    start_time: float  # seconds
    end_time: float    # seconds  
    midi_pitch: int    # Original MIDI pitch (21-108 for piano)
    velocity: int = 100


class KeysCharter:
    """
    Transcribes keyboard instruments from audio.
    
    Features:
    - Keyboard detection: Only generates tracks if keys are detected
    - Uses Basic Pitch for note transcription
    - Generates both 5-lane and Pro Keys formats
    """
    
    # Pro Keys range: 25 keys (2 octaves + 1)
    PRO_KEYS_MIN = 48  # C3
    PRO_KEYS_MAX = 72  # C5
    PRO_KEYS_RANGE = 25
    
    # 5-lane mapping thresholds
    FIVE_LANE_RANGES = [
        (0, 48),    # Lane 0: Low bass (below C3)
        (48, 54),   # Lane 1: C3-F3
        (54, 60),   # Lane 2: F#3-B3
        (60, 66),   # Lane 3: C4-F4
        (66, 108),  # Lane 4: F#4 and above
    ]
    
    def __init__(
        self,
        detection_threshold: float = 0.3,  # Minimum keyboard presence ratio
        min_notes: int = 20,  # Minimum notes to consider as keyboard track
    ):
        self.detection_threshold = detection_threshold
        self.min_notes = min_notes
    
    def _transcribe_notes_librosa(self, audio_path: str) -> List[KeyNote]:
        """
        Transcribe notes using librosa's onset detection and pitch tracking.
        
        This is a simpler alternative to Basic Pitch that works with Python 3.12.
        """
        logger.info(f"Transcribing with librosa: {audio_path}")
        
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=27.5, fmax=4186)
        
        notes = []
        
        for i, onset_time in enumerate(onset_times):
            # Get the frame for this onset
            onset_frame = onset_frames[i]
            
            # Look at a window after the onset for pitch
            start_frame = onset_frame
            end_frame = min(onset_frame + 10, pitches.shape[1] - 1)
            
            # Find dominant pitch in window
            best_pitch = 0
            best_mag = 0
            
            for frame in range(start_frame, end_frame):
                idx = magnitudes[:, frame].argmax()
                pitch = pitches[idx, frame]
                mag = magnitudes[idx, frame]
                
                if pitch > 0 and mag > best_mag:
                    best_pitch = pitch
                    best_mag = mag
            
            if best_pitch > 0:
                midi_pitch = int(round(librosa.hz_to_midi(best_pitch)))
                
                # Clamp to piano range
                midi_pitch = max(21, min(108, midi_pitch))
                
                # Calculate duration (until next onset or 0.5s max)
                if i + 1 < len(onset_times):
                    duration = min(onset_times[i + 1] - onset_time, 1.0)
                else:
                    duration = 0.5
                
                duration = max(0.1, duration)  # Minimum 100ms
                
                notes.append(KeyNote(
                    start_time=float(onset_time),
                    end_time=float(onset_time + duration),
                    midi_pitch=midi_pitch,
                    velocity=min(127, int(best_mag * 127 / magnitudes.max())),
                ))
        
        return notes
    
    def detect_keyboard_presence(self, audio_path: str) -> Tuple[bool, float, dict]:
        """
        Detect if the audio contains keyboard instruments.
        
        Uses spectral analysis to distinguish keyboard from guitar:
        - Keyboards have more sustained notes with consistent harmonics
        - Keyboards often have cleaner attack transients
        - Piano has characteristic hammer attack + decay envelope
        - Synths have more constant amplitude during sustain
        
        Args:
            audio_path: Path to audio file (typically 'other' stem)
            
        Returns:
            (has_keys, confidence, details) tuple
        """
        logger.info(f"Analyzing audio for keyboard presence: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        duration = len(y) / sr
        
        if duration < 5:
            logger.info("Audio too short for keyboard detection")
            return False, 0.0, {"reason": "too_short"}
        
        # Feature 1: Spectral flatness (keyboards often have cleaner harmonics)
        # Lower flatness = more tonal/harmonic content
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        avg_flatness = np.mean(flatness)
        
        # Feature 2: Note sustain analysis using onset/offset detection
        # Keyboards typically have longer, more sustained notes than guitar
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
        
        # Feature 3: Harmonic content ratio
        harmonic, percussive = librosa.effects.hpss(y)
        harmonic_ratio = np.sum(np.abs(harmonic)) / (np.sum(np.abs(y)) + 1e-8)
        
        # Feature 4: Pitch range analysis - keyboards often span wider range
        # Use pitch detection to see note distribution
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        pitch_range = 0
        if pitch_values:
            pitch_midi = [librosa.hz_to_midi(p) for p in pitch_values if p > 0]
            if pitch_midi:
                pitch_range = max(pitch_midi) - min(pitch_midi)
        
        # Feature 5: Check for typical keyboard frequency bands
        # Piano fundamental range: ~27.5 Hz (A0) to ~4186 Hz (C8)
        # Most keyboard content: 100 Hz - 2000 Hz
        stft = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Energy in keyboard-typical range (100-2000 Hz)
        keyboard_band = (freqs >= 100) & (freqs <= 2000)
        keyboard_energy = np.mean(stft[keyboard_band, :])
        total_energy = np.mean(stft) + 1e-8
        keyboard_ratio = keyboard_energy / total_energy
        
        # Scoring
        score = 0.0
        
        # Low flatness indicates tonal content (good for keys)
        if avg_flatness < 0.1:
            score += 0.25
        elif avg_flatness < 0.2:
            score += 0.15
        
        # High harmonic ratio indicates sustained tonal content
        if harmonic_ratio > 0.7:
            score += 0.25
        elif harmonic_ratio > 0.5:
            score += 0.15
        
        # Wide pitch range suggests keyboard
        if pitch_range > 24:  # More than 2 octaves
            score += 0.25
        elif pitch_range > 12:  # More than 1 octave
            score += 0.15
        
        # Strong keyboard frequency band
        if keyboard_ratio > 1.5:
            score += 0.25
        elif keyboard_ratio > 1.0:
            score += 0.15
        
        details = {
            "spectral_flatness": float(avg_flatness),
            "harmonic_ratio": float(harmonic_ratio),
            "pitch_range_semitones": float(pitch_range),
            "keyboard_band_ratio": float(keyboard_ratio),
            "num_onsets": len(onsets),
            "score": float(score),
        }
        
        has_keys = score >= self.detection_threshold
        logger.info(f"Keyboard detection: {'FOUND' if has_keys else 'NOT FOUND'} "
                   f"(score={score:.2f}, threshold={self.detection_threshold})")
        logger.info(f"  Details: flatness={avg_flatness:.3f}, harmonic={harmonic_ratio:.2f}, "
                   f"pitch_range={pitch_range:.1f}, kb_ratio={keyboard_ratio:.2f}")
        
        return has_keys, score, details
    
    def transcribe(
        self,
        audio_path: str,
        force: bool = False,  # Generate even if keyboard not detected
    ) -> Tuple[Optional[List[KeyNote]], dict]:
        """
        Transcribe keyboard content from audio.
        
        Args:
            audio_path: Path to audio file
            force: If True, generate keys even if not detected
            
        Returns:
            (notes, details) - notes is None if no keyboard detected and not forced
        """
        # Step 1: Detect keyboard presence
        has_keys, confidence, detection_details = self.detect_keyboard_presence(audio_path)
        
        if not has_keys and not force:
            logger.info("No keyboard instruments detected, skipping keys track")
            return None, {"detected": False, **detection_details}
        
        if not has_keys and force:
            logger.warning("Forcing keys transcription despite low keyboard confidence")
        
        # Step 2: Transcribe using librosa
        logger.info("Transcribing keyboard notes...")
        notes = self._transcribe_notes_librosa(audio_path)
        
        logger.info(f"Transcribed {len(notes)} keyboard notes")
        
        if len(notes) < self.min_notes and not force:
            logger.info(f"Too few notes ({len(notes)} < {self.min_notes}), skipping keys track")
            return None, {"detected": True, "notes": len(notes), "reason": "too_few_notes"}
        
        # Sort by start time
        notes.sort(key=lambda n: n.start_time)
        
        # Filter tiny notes (< 50ms)
        notes = [n for n in notes if n.end_time - n.start_time >= 0.05]
        
        details = {
            "detected": True,
            "confidence": confidence,
            "notes": len(notes),
            **detection_details,
        }
        
        return notes, details
    
    def notes_to_5lane(self, notes: List[KeyNote]) -> List[Tuple[float, float, int]]:
        """
        Convert notes to 5-lane format.
        
        Returns list of (start_time, end_time, lane) tuples.
        """
        result = []
        
        for note in notes:
            # Map pitch to lane
            lane = 2  # Default to middle
            for i, (low, high) in enumerate(self.FIVE_LANE_RANGES):
                if low <= note.midi_pitch < high:
                    lane = i
                    break
            
            result.append((note.start_time, note.end_time, lane))
        
        return result
    
    def notes_to_prokeys(self, notes: List[KeyNote]) -> List[Tuple[float, float, int]]:
        """
        Convert notes to Pro Keys format (25-key range).
        
        Transposes notes to fit within C3-C5 range.
        
        Returns list of (start_time, end_time, midi_pitch) tuples.
        """
        result = []
        
        for note in notes:
            pitch = note.midi_pitch
            
            # Transpose to Pro Keys range if needed
            while pitch < self.PRO_KEYS_MIN:
                pitch += 12
            while pitch > self.PRO_KEYS_MAX:
                pitch -= 12
            
            result.append((note.start_time, note.end_time, pitch))
        
        return result
    
    def export_midi(
        self,
        notes: List[KeyNote],
        output_path: str,
        tempo_bpm: float = 120.0,
        ticks_per_beat: int = 480,
    ) -> mido.MidiFile:
        """
        Export notes to MIDI file with PART KEYS and PART REAL_KEYS_X tracks.
        """
        midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
        ticks_per_sec = ticks_per_beat * tempo_bpm / 60
        
        # 5-lane Keys track (PART KEYS)
        five_lane_notes = self.notes_to_5lane(notes)
        keys_track = self._create_5lane_track(five_lane_notes, ticks_per_sec)
        midi.tracks.append(keys_track)
        
        # Pro Keys tracks (multiple difficulties)
        prokeys_notes = self.notes_to_prokeys(notes)
        for difficulty in ['E', 'M', 'H', 'X']:
            prokeys_track = self._create_prokeys_track(
                prokeys_notes, ticks_per_sec, difficulty
            )
            midi.tracks.append(prokeys_track)
        
        if output_path:
            midi.save(output_path)
            logger.info(f"Saved keys MIDI to {output_path}")
        
        return midi
    
    def _create_5lane_track(
        self,
        notes: List[Tuple[float, float, int]],
        ticks_per_sec: float,
    ) -> mido.MidiTrack:
        """Create 5-lane keys track."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name='PART KEYS', time=0))
        
        # 5-lane note mapping (same as guitar)
        # Clone Hero difficulty note ranges:
        LANE_NOTES = {
            'E': [60, 61, 62, 63, 64],  # Easy
            'M': [72, 73, 74, 75, 76],  # Medium
            'H': [84, 85, 86, 87, 88],  # Hard
            'X': [96, 97, 98, 99, 100], # Expert
        }
        
        # Build events for all difficulties
        events = []
        
        for diff, base_notes in LANE_NOTES.items():
            for start_time, end_time, lane in notes:
                start_tick = int(start_time * ticks_per_sec)
                end_tick = int(end_time * ticks_per_sec)
                
                note = base_notes[lane]
                events.append((start_tick, 'on', note, 100))
                events.append((end_tick, 'off', note, 0))
        
        # Sort and convert to delta times
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'on' else 1))
        
        current_tick = 0
        for tick, event_type, note, velocity in events:
            delta = max(0, tick - current_tick)
            
            if event_type == 'on':
                track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta))
            else:
                track.append(mido.Message('note_off', note=note, velocity=0, time=delta))
            
            current_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def _create_prokeys_track(
        self,
        notes: List[Tuple[float, float, int]],
        ticks_per_sec: float,
        difficulty: str,
    ) -> mido.MidiTrack:
        """
        Create Pro Keys track for a given difficulty.
        
        Pro Keys uses actual MIDI pitches (48-72 = C3-C5) within a 25-key range.
        The visible keyboard shows 17 keys at a time.
        Range Shift markers (note 9) tell the game where to position the 17-key view.
        
        Range positions (note 9 velocity):
        - 0: C3-E4 (48-64)
        - 1: D3-F4 (50-65) 
        - 2: E3-G4 (52-67)
        - 3: F3-A4 (53-69)
        - 4: G3-B4 (55-71)
        - 5: A3-C5 (57-72)
        """
        track = mido.MidiTrack()
        track_name = f'PART REAL_KEYS_{difficulty}'
        track.append(mido.MetaMessage('track_name', name=track_name, time=0))
        
        # For easier difficulties, reduce note density
        note_skip = {'E': 4, 'M': 2, 'H': 1, 'X': 1}[difficulty]
        
        # Filter notes first
        filtered_notes = []
        for i, (start_time, end_time, pitch) in enumerate(notes):
            if i % note_skip != 0:
                continue
            filtered_notes.append((start_time, end_time, pitch))
        
        if not filtered_notes:
            track.append(mido.MetaMessage('end_of_track', time=0))
            return track
        
        # Calculate range shifts based on note positions
        # Range position defines the lowest visible key: pos 0 = C3(48), pos 5 = A3(57)
        RANGE_POSITIONS = [48, 50, 52, 53, 55, 57]  # Lowest key for each position
        VISIBLE_RANGE = 17  # 17 keys visible
        
        def get_best_range_position(pitch: int) -> int:
            """Find the range position that best shows this pitch."""
            for pos in range(len(RANGE_POSITIONS) - 1, -1, -1):
                low = RANGE_POSITIONS[pos]
                high = low + VISIBLE_RANGE - 1
                if low <= pitch <= high:
                    return pos
            return 0  # Default to leftmost
        
        events = []
        current_range_pos = -1
        
        # Group notes by time and determine range shifts
        notes_by_time = {}
        for start_time, end_time, pitch in filtered_notes:
            start_tick = int(start_time * ticks_per_sec)
            if start_tick not in notes_by_time:
                notes_by_time[start_tick] = []
            notes_by_time[start_tick].append((start_time, end_time, pitch))
        
        for start_tick in sorted(notes_by_time.keys()):
            tick_notes = notes_by_time[start_tick]
            
            # Find the range that covers all notes at this time
            pitches = [p for _, _, p in tick_notes]
            min_pitch = min(pitches)
            max_pitch = max(pitches)
            
            # Find best range position for these notes
            best_pos = None
            for pos in range(len(RANGE_POSITIONS)):
                low = RANGE_POSITIONS[pos]
                high = low + VISIBLE_RANGE - 1
                if low <= min_pitch and max_pitch <= high:
                    best_pos = pos
                    break
            
            if best_pos is None:
                # Notes span too wide; pick position that shows the most
                best_pos = get_best_range_position((min_pitch + max_pitch) // 2)
            
            # Add range shift if position changed
            if best_pos != current_range_pos:
                # Note 9 is the range shift marker, velocity is the position
                shift_tick = max(0, start_tick - 480)  # Place shift 1 beat before
                events.append((shift_tick, 'on', 9, best_pos))
                events.append((shift_tick + 1, 'off', 9, 0))
                current_range_pos = best_pos
            
            # Add the actual notes
            for start_time, end_time, pitch in tick_notes:
                end_tick = int(end_time * ticks_per_sec)
                events.append((start_tick, 'on', pitch, 100))
                events.append((end_tick, 'off', pitch, 0))
        
        # Sort events
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'on' else 1, e[2]))
        
        # Convert to delta times
        current_tick = 0
        for tick, event_type, note, velocity in events:
            delta = max(0, tick - current_tick)
            
            if event_type == 'on':
                track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta))
            else:
                track.append(mido.Message('note_off', note=note, velocity=0, time=delta))
            
            current_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Keys charter for Clone Hero/YARG")
    parser.add_argument("audio", help="Path to audio file (usually 'other' stem)")
    parser.add_argument("-o", "--output", help="Output MIDI path")
    parser.add_argument("--force", action="store_true", help="Generate keys even if not detected")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    
    args = parser.parse_args()
    
    charter = KeysCharter(detection_threshold=args.threshold)
    notes, details = charter.transcribe(args.audio, force=args.force)
    
    if notes:
        output = args.output or args.audio.replace('.wav', '_keys.mid')
        charter.export_midi(notes, output)
        print(f"Generated {len(notes)} key notes")
    else:
        print("No keyboard content detected")
        print(f"Details: {details}")
