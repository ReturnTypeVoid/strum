"""
Unified Inference Pipeline

End-to-end chart generation: Audio → Demucs → Drums/Guitar/Bass → Combined Chart

Usage:
    python -m src.inference.unified song.mp3 -o chart.mid
    python -m src.inference.unified song.mp3 --stems-dir /path/to/stems  # pre-separated
"""

import argparse
import sys
from pathlib import Path
import logging
import tempfile
import shutil
from typing import Optional, Dict, List
from dataclasses import dataclass

import gc
import torch
import numpy as np
import mido

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for unified inference."""
    # Demucs
    demucs_model: str = "htdemucs"
    
    # Drums
    drums_checkpoint: str = "checkpoints/drums_v6.2/best.pt"
    drums_threshold: float = 0.6
    
    # Guitar
    guitar_checkpoint: str = "checkpoints/guitar_onset/best.pt"
    guitar_threshold: float = 0.6  # Higher threshold = fewer notes
    guitar_quantize_strength: float = 0.8  # 0=no quantization, 1=hard snap
    guitar_quantize_grid: str = '1/16'  # Beat subdivision
    
    # Bass (same model as guitar)
    bass_checkpoint: str = "checkpoints/guitar_onset/best.pt"  
    bass_threshold: float = 0.5  # Balanced threshold for bass
    
    # Vocals
    whisper_model: str = "medium"  # tiny, base, small, medium, large-v3
    vocals_timing_offset: float = -0.05  # Base offset (reduced, dynamic alignment handles rest)
    vocals_dynamic_alignment: bool = True  # Enable onset-based dynamic alignment
    vocals_alignment_tolerance: float = 0.15  # Max time to shift a word to reach onset
    
    # Output
    tempo_bpm: float = None  # None = auto-detect from audio
    ticks_per_beat: int = 480
    start_buffer_sec: float = 2.0  # Silence before first note for player readiness
    
    # Metadata & Video
    fetch_metadata: bool = True  # Lookup metadata from MusicBrainz/Spotify
    download_video: bool = False  # Download YouTube video as background
    video_search_query: str = None  # Custom YouTube search (default: "{artist} {title} official video")


class UnifiedInference:
    """
    Unified inference pipeline for full chart generation.
    
    Chains: Audio → Demucs → Instrument Models → Combined MIDI
    """
    
    def __init__(
        self,
        config: InferenceConfig = None,
        device: str = None,
    ):
        self.config = config or InferenceConfig()
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        logger.info(f"Unified Inference initialized on {self.device}")
        
        # Lazy load models
        self._drums_engine = None
        self._guitar_charter = None
        self._bass_charter = None
        self._vocals_charter = None
        self._keys_charter = None
    
    def _cleanup_gpu(self, unload: List[str] = None):
        """Free GPU memory between instrument steps.
        
        Args:
            unload: List of model names to explicitly unload.
                    Supported: 'drums', 'guitar', 'bass', 'vocals', 'keys'
        """
        if unload:
            for name in unload:
                attr = f"_{name}_engine" if name == 'drums' else f"_{name}_charter"
                model = getattr(self, attr, None)
                if model is not None:
                    # Try to move model to CPU or delete it
                    if hasattr(model, 'model'):
                        del model.model
                    setattr(self, attr, None)
                    logger.debug(f"Unloaded {name} model")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    @property
    def drums_engine(self):
        """Lazy-load drums inference engine."""
        if self._drums_engine is None:
            try:
                from src.inference.drums_cli import DrumsInferenceEngine
                self._drums_engine = DrumsInferenceEngine(
                    checkpoint_path=self.config.drums_checkpoint,
                    device=str(self.device)
                )
            except Exception as e:
                logger.warning(f"Could not load drums model: {e}")
        return self._drums_engine
    
    @property
    def guitar_charter(self):
        """Lazy-load guitar hybrid charter."""
        if self._guitar_charter is None:
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
                from guitar_hybrid import HybridGuitarCharter
                self._guitar_charter = HybridGuitarCharter(
                    onset_checkpoint=Path(self.config.guitar_checkpoint),
                    device=self.device,
                    onset_threshold=self.config.guitar_threshold,
                    tempo_bpm=self.config.tempo_bpm,
                    quantize_strength=self.config.guitar_quantize_strength,
                    quantize_grid=self.config.guitar_quantize_grid,
                )
            except Exception as e:
                logger.warning(f"Could not load guitar model: {e}")
        return self._guitar_charter
    
    @property
    def bass_charter(self):
        """Lazy-load bass hybrid charter (same model, different threshold)."""
        if self._bass_charter is None:
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
                from guitar_hybrid import HybridGuitarCharter
                self._bass_charter = HybridGuitarCharter(
                    onset_checkpoint=Path(self.config.bass_checkpoint),
                    device=self.device,
                    onset_threshold=self.config.bass_threshold,
                    tempo_bpm=self.config.tempo_bpm,
                    quantize_strength=self.config.guitar_quantize_strength,
                    quantize_grid=self.config.guitar_quantize_grid,
                )
            except Exception as e:
                logger.warning(f"Could not load bass model: {e}")
        return self._bass_charter
    
    @property
    def vocals_charter(self):
        """Lazy-load vocals charter with WhisperX and Basic Pitch."""
        if self._vocals_charter is None:
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
                from vocals_charter import VocalsCharter
                self._vocals_charter = VocalsCharter(
                    whisper_model=self.config.whisper_model,
                    device=str(self.device),
                    timing_offset=self.config.vocals_timing_offset,
                    dynamic_alignment=self.config.vocals_dynamic_alignment,
                    alignment_tolerance=self.config.vocals_alignment_tolerance,
                )
            except Exception as e:
                logger.warning(f"Could not load vocals model: {e}")
                import traceback
                traceback.print_exc()
        return self._vocals_charter
    
    @property
    def keys_charter(self):
        """Lazy-load keys charter for keyboard detection and transcription."""
        if self._keys_charter is None:
            try:
                sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'scripts'))
                from keys_charter import KeysCharter
                self._keys_charter = KeysCharter()
            except Exception as e:
                logger.warning(f"Could not load keys charter: {e}")
                import traceback
                traceback.print_exc()
        return self._keys_charter
    
    def reduce_difficulty(
        self,
        midi_file: mido.MidiFile,
        track_name_prefix: str = "PART GUITAR",
    ) -> mido.MidiFile:
        """
        Generate Hard/Medium/Easy difficulties from Expert track.
        
        Reduction strategy:
        - Hard: Keep 80% of notes, simplify some chords
        - Medium: Keep 50% of notes, no chords
        - Easy: Keep 30% of notes, no chords, fewer lanes
        
        Args:
            midi_file: MIDI file with Expert track
            track_name_prefix: Track name prefix (PART GUITAR or PART BASS)
            
        Returns:
            MIDI file with all difficulty tracks added
        """
        if midi_file is None:
            return None
        
        import random
        random.seed(42)  # Deterministic reduction
        
        # Note ranges per difficulty in Clone Hero format
        # Expert: 96-100, Hard: 84-88, Medium: 72-76, Easy: 60-64
        DIFF_OFFSETS = {
            'X': 0,    # Expert (source)
            'H': -12,  # Hard
            'M': -24,  # Medium  
            'E': -36,  # Easy
        }
        
        KEEP_RATIOS = {'H': 0.8, 'M': 0.5, 'E': 0.3}
        
        # Find Expert track
        expert_track = None
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'track_name' and track_name_prefix in msg.name:
                    expert_track = track
                    break
            if expert_track:
                break
        
        if not expert_track:
            logger.warning(f"No {track_name_prefix} track found for difficulty reduction")
            return midi_file
        
        # Extract notes from Expert track (notes 96-100)
        notes = []  # (abs_tick, type, note, velocity)
        abs_tick = 0
        for msg in expert_track:
            abs_tick += msg.time
            if msg.type in ('note_on', 'note_off') and 96 <= msg.note <= 100:
                notes.append((abs_tick, msg.type, msg.note, msg.velocity if msg.type == 'note_on' else 0))
        
        # Group notes by time for chord detection
        from collections import defaultdict
        notes_by_time = defaultdict(list)
        for tick, mtype, note, vel in notes:
            if mtype == 'note_on' and vel > 0:
                notes_by_time[tick].append(note)
        
        # Get sorted unique times
        note_times = sorted(notes_by_time.keys())
        
        # Generate reduced difficulty tracks
        for diff in ['H', 'M', 'E']:
            offset = DIFF_OFFSETS[diff]
            keep_ratio = KEEP_RATIOS[diff]
            
            # Select which times to keep
            kept_times = set()
            for i, t in enumerate(note_times):
                # Always keep first/last
                if i == 0 or i == len(note_times) - 1:
                    kept_times.add(t)
                # Otherwise use keep ratio with some variance
                elif random.random() < keep_ratio:
                    kept_times.add(t)
            
            # Build new track
            new_track = mido.MidiTrack()
            new_track.append(mido.MetaMessage(
                'track_name', 
                name=f"{track_name_prefix}",  # Same name, different notes
                time=0
            ))
            
            # Process notes
            events = []
            abs_tick = 0
            for msg in expert_track:
                abs_tick += msg.time
                
                if msg.type in ('note_on', 'note_off') and 96 <= msg.note <= 100:
                    if abs_tick in kept_times or msg.type == 'note_off':
                        new_note = msg.note + offset
                        
                        # For Easy/Medium, simplify chords to single note
                        if diff in ['E', 'M'] and msg.type == 'note_on' and msg.velocity > 0:
                            chord_notes = notes_by_time.get(abs_tick, [])
                            if len(chord_notes) > 1:
                                # Keep only lowest note
                                if msg.note != min(chord_notes):
                                    continue
                        
                        # For Easy, reduce to 3 lanes (0, 1, 2)
                        if diff == 'E':
                            lane = msg.note - 96
                            if lane > 2:
                                new_note = 60 + (lane % 3)
                            else:
                                new_note = 60 + lane
                        
                        events.append((abs_tick, msg.type, new_note, msg.velocity))
                        
                elif msg.type not in ('note_on', 'note_off', 'end_of_track'):
                    # Keep non-note messages
                    events.append((abs_tick, 'meta', msg, 0))
            
            # Convert to delta times
            events.sort(key=lambda e: e[0])
            prev_tick = 0
            for tick, etype, data, vel in events:
                delta = tick - prev_tick
                if etype in ('note_on', 'note_off'):
                    new_track.append(mido.Message(etype, note=data, velocity=vel, time=delta))
                elif etype == 'meta':
                    data.time = delta
                    new_track.append(data.copy())
                prev_tick = tick
            
            new_track.append(mido.MetaMessage('end_of_track', time=0))
            midi_file.tracks.append(new_track)
        
        logger.info(f"Added Hard/Medium/Easy difficulties for {track_name_prefix}")
        return midi_file
    
    def detect_tempo(self, audio_path: Path) -> float:
        """
        Auto-detect tempo from audio using librosa beat tracking.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Detected tempo in BPM (rounded to nearest integer)
        """
        import librosa
        
        logger.info(f"Auto-detecting tempo from {audio_path.name}...")
        
        # Load audio (use first 60 seconds for speed)
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=60)
        
        # Detect tempo using beat tracking
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Handle array return in newer librosa versions
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0])
        else:
            tempo = float(tempo)
        
        # Round to nearest integer for cleaner charts
        tempo = round(tempo)
        
        # Sanity check - most songs are 60-200 BPM
        if tempo < 60:
            tempo *= 2  # Likely detected half-time
        elif tempo > 200:
            tempo //= 2  # Likely detected double-time
        
        logger.info(f"Detected tempo: {tempo} BPM")
        return tempo
    
    def fetch_song_metadata(self, artist: str, title: str) -> dict:
        """
        Fetch song metadata from MusicBrainz and other sources.
        
        Args:
            artist: Artist name
            title: Song title
            
        Returns:
            Dict with metadata: album, year, genre, cover_url, etc.
        """
        import urllib.request
        import urllib.parse
        import json
        
        metadata = {
            'artist': artist,
            'title': title,
            'album': '',
            'year': '',
            'genre': '',
            'cover_url': None,
        }
        
        try:
            # Search MusicBrainz
            query = urllib.parse.quote(f'artist:"{artist}" AND recording:"{title}"')
            url = f"https://musicbrainz.org/ws/2/recording?query={query}&fmt=json&limit=1"
            
            req = urllib.request.Request(url, headers={'User-Agent': 'STRUM/1.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())
                
            if data.get('recordings'):
                rec = data['recordings'][0]
                
                # Get release info
                if rec.get('releases'):
                    release = rec['releases'][0]
                    metadata['album'] = release.get('title', '')
                    if release.get('date'):
                        metadata['year'] = release['date'][:4]
                    
                    # Try to get cover art
                    if release.get('id'):
                        metadata['cover_url'] = f"https://coverartarchive.org/release/{release['id']}/front-250"
                
                logger.info(f"Found metadata: {metadata['album']} ({metadata['year']})")
                
        except Exception as e:
            logger.warning(f"Could not fetch metadata: {e}")
        
        return metadata
    
    def download_youtube_video(
        self,
        artist: str,
        title: str,
        output_path: Path,
        search_query: str = None,
    ) -> Optional[Path]:
        """
        Download YouTube video as background for the chart.
        
        Uses yt-dlp to search and download the official video.
        
        Args:
            artist: Artist name
            title: Song title  
            output_path: Where to save video.mp4
            search_query: Custom search query (default: "{artist} {title} official video")
            
        Returns:
            Path to downloaded video or None if failed
        """
        try:
            import subprocess
            
            if search_query is None:
                search_query = f"{artist} {title} official video"
            
            logger.info(f"Searching YouTube for: {search_query}")
            
            # Use yt-dlp to search and download
            cmd = [
                'yt-dlp',
                f'ytsearch1:{search_query}',  # Search and get first result
                '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best',
                '-o', str(output_path),
                '--no-playlist',
                '--quiet',
                '--no-warnings',
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Downloaded video: {output_path}")
                return output_path
            else:
                logger.warning(f"yt-dlp failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            logger.warning("yt-dlp not found. Install with: pip install yt-dlp")
            return None
        except Exception as e:
            logger.warning(f"Video download failed: {e}")
            return None

    def separate_stems(
        self,
        audio_path: Path,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Separate audio into stems using Demucs.
        
        Returns:
            Dict mapping stem names to paths
        """
        from src.preprocessing.separation import separate_stems
        
        logger.info(f"Separating {audio_path} with Demucs...")
        return separate_stems(
            audio_path=audio_path,
            output_dir=output_dir,
            model_name=self.config.demucs_model,
            device=str(self.device),
        )
    
    def transcribe_drums(self, drums_wav: Path) -> Optional[mido.MidiFile]:
        """Transcribe drums from drums.wav stem."""
        if self.drums_engine is None:
            logger.warning("Drums model not available, skipping")
            return None
        
        logger.info("Transcribing drums...")
        try:
            chart = self.drums_engine.transcribe(str(drums_wav), tempo_bpm=self.config.tempo_bpm)
            
            # Convert to MIDI
            from src.export.midi import export_drums_midi, MidiExportConfig
            from io import BytesIO
            
            midi_bytes = BytesIO()
            config = MidiExportConfig(default_tempo_bpm=self.config.tempo_bpm)
            # Save to temp file and load as MidiFile
            temp_path = Path(tempfile.gettempdir()) / "drums_temp.mid"
            export_drums_midi(chart, temp_path, config)
            midi = mido.MidiFile(temp_path)
            temp_path.unlink()
            return midi
        except Exception as e:
            logger.error(f"Drums transcription failed: {e}")
            return None
    
    def transcribe_guitar(self, other_wav: Path, bass_wav: Path) -> Optional[mido.MidiFile]:
        """Transcribe guitar from other.wav + bass.wav stems."""
        if self.guitar_charter is None:
            logger.warning("Guitar model not available, skipping")
            return None
        
        logger.info("Transcribing guitar...")
        try:
            notes = self.guitar_charter.chart_from_stems(other_wav, bass_wav)
            
            # Convert to MIDI
            midi = mido.MidiFile(ticks_per_beat=self.config.ticks_per_beat)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            
            # Track name must come first (no tempo in instrument tracks)
            track.append(mido.MetaMessage('track_name', name='PART GUITAR', time=0))
            
            # Difficulty note offsets
            difficulty_offsets = {
                'expert': 0,    # 96-100
                'hard': -12,    # 84-88
                'medium': -24,  # 72-76
                'easy': -36,    # 60-64
            }
            
            # Add notes for ALL difficulties
            events = []
            for note in notes:
                tick = int(note.time_sec * self.config.tempo_bpm / 60 * self.config.ticks_per_beat)
                
                # KEY FIX: Only sustain notes get real duration
                # Non-sustain notes get 1 tick (no visual sustain tail)
                if hasattr(note, 'is_sustain') and note.is_sustain:
                    duration_ticks = max(1, int(note.duration_sec * self.config.tempo_bpm / 60 * self.config.ticks_per_beat))
                else:
                    duration_ticks = 1  # Minimal - just a tap
                
                # Collect all frets for this note (primary + chord frets)
                all_frets = [note.fret]
                if hasattr(note, 'chord_frets') and note.chord_frets:
                    all_frets.extend(note.chord_frets)
                
                # Add note for each difficulty
                for diff_name, offset in difficulty_offsets.items():
                    for fret in all_frets:
                        midi_note = 96 + fret + offset
                        events.append((tick, 'note_on', midi_note, 100))
                        events.append((tick + duration_ticks, 'note_off', midi_note, 0))
            
            # Sort by time and convert to delta times
            events.sort(key=lambda x: (x[0], x[1] == 'note_off'))
            prev_tick = 0
            for tick, msg_type, note, vel in events:
                delta = tick - prev_tick
                track.append(mido.Message(msg_type, note=note, velocity=vel, time=delta))
                prev_tick = tick
            
            n_chords = sum(1 for n in notes if hasattr(n, 'chord_frets') and n.chord_frets)
            logger.info(f"Guitar: {len(notes)} notes ({n_chords} chords) transcribed (x4 difficulties)")
            return midi
        except Exception as e:
            logger.error(f"Guitar transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def transcribe_bass(self, bass_wav: Path, other_wav: Path) -> Optional[mido.MidiFile]:
        """
        Transcribe bass from bass.wav stem.
        Uses same onset model as guitar but with lower threshold for bass.
        """
        if self.bass_charter is None:
            logger.warning("Bass model not available, skipping")
            return None
        
        logger.info("Transcribing bass...")
        try:
            # For bass, use bass_charter with lower threshold
            notes = self.bass_charter.chart_from_stems(bass_wav, other_wav)
            
            # Convert to MIDI  
            midi = mido.MidiFile(ticks_per_beat=self.config.ticks_per_beat)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            
            # Track name must come first (no tempo in instrument tracks)
            track.append(mido.MetaMessage('track_name', name='PART BASS', time=0))
            
            # Difficulty note offsets
            difficulty_offsets = {
                'expert': 0,    # 96-100
                'hard': -12,    # 84-88
                'medium': -24,  # 72-76
                'easy': -36,    # 60-64
            }
            
            # Add notes for ALL difficulties
            events = []
            for note in notes:
                tick = int(note.time_sec * self.config.tempo_bpm / 60 * self.config.ticks_per_beat)
                
                # KEY FIX: Only sustain notes get real duration
                # Non-sustain notes get 1 tick (no visual sustain tail)
                if hasattr(note, 'is_sustain') and note.is_sustain:
                    duration_ticks = max(1, int(note.duration_sec * self.config.tempo_bpm / 60 * self.config.ticks_per_beat))
                else:
                    duration_ticks = 1  # Minimal - just a tap
                
                # Collect all frets for this note (primary + chord frets)
                # Note: bass typically doesn't have chords, but support it anyway
                all_frets = [note.fret]
                if hasattr(note, 'chord_frets') and note.chord_frets:
                    all_frets.extend(note.chord_frets)
                
                # Add note for each difficulty
                for diff_name, offset in difficulty_offsets.items():
                    for fret in all_frets:
                        midi_note = 96 + fret + offset
                        events.append((tick, 'note_on', midi_note, 100))
                        events.append((tick + duration_ticks, 'note_off', midi_note, 0))
            
            # Sort by time and convert to delta times
            events.sort(key=lambda x: (x[0], x[1] == 'note_off'))
            prev_tick = 0
            for tick, msg_type, note, vel in events:
                delta = tick - prev_tick
                track.append(mido.Message(msg_type, note=note, velocity=vel, time=delta))
                prev_tick = tick
            
            logger.info(f"Bass: {len(notes)} notes transcribed (x4 difficulties)")
            return midi
        except Exception as e:
            logger.error(f"Bass transcription failed: {e}")
            return None
    
    def transcribe_vocals(self, vocals_wav: Path) -> Optional[mido.MidiFile]:
        """
        Transcribe vocals from vocals.wav stem.
        
        Creates PART VOCALS and HARM1 tracks with:
        - Phrase markers (note 105 spanning full phrase)
        - Pitch notes (36-84)
        - Lyric text events
        """
        if self.vocals_charter is None:
            logger.warning("Vocals model not available, skipping")
            return None
        
        logger.info("Transcribing vocals...")
        try:
            artist = getattr(self, '_artist', None)
            title = getattr(self, '_title', None)
            lead_phrases, harmony_phrases = self.vocals_charter.transcribe(
                str(vocals_wav),
                artist=artist,
                title=title
            )
            
            midi = mido.MidiFile(ticks_per_beat=self.config.ticks_per_beat)
            ticks_per_sec = self.config.ticks_per_beat * self.config.tempo_bpm / 60
            
            def export_vocal_track(phrases, track_name):
                """Export a vocal track with proper phrase boundaries."""
                track = mido.MidiTrack()
                track.append(mido.MetaMessage('track_name', name=track_name, time=0))
                
                PHRASE_LEAD_TICKS = 180  # Phrase marker before first note
                MIN_NOTE_TICKS = 50  # Minimum note duration
                
                # Filter empty phrases
                valid_phrases = [p for p in phrases if p.notes]
                if not valid_phrases:
                    return track
                
                # FIRST: Calculate phrase boundaries and prevent overlap
                phrase_bounds = []
                for phrase in valid_phrases:
                    first_note_start = int(phrase.notes[0].start_time * ticks_per_sec)
                    last_note_end = int(phrase.notes[-1].end_time * ticks_per_sec)
                    
                    phrase_start = max(0, first_note_start - PHRASE_LEAD_TICKS)
                    phrase_end = last_note_end + 50
                    phrase_bounds.append([phrase_start, phrase_end])
                
                # Prevent phrase overlap: phrase N end must be < phrase N+1 start
                for i in range(len(phrase_bounds) - 1):
                    current_end = phrase_bounds[i][1]
                    next_start = phrase_bounds[i + 1][0]
                    if current_end >= next_start:
                        # Shrink current phrase to end before next starts
                        phrase_bounds[i][1] = next_start - 10
                
                # Build all events as (absolute_tick, priority, event_data)
                all_events = []
                
                for idx, phrase in enumerate(valid_phrases):
                    phrase_start_tick, phrase_end_tick = phrase_bounds[idx]
                    
                    # Phrase start (note 105 on)
                    all_events.append((phrase_start_tick, 0, ('phrase_on', 105)))
                    # Phrase end (note 105 off)
                    all_events.append((phrase_end_tick, 5, ('phrase_off', 105)))
                    
                    # Add notes within phrase
                    sorted_notes = sorted(phrase.notes, key=lambda n: n.start_time)
                    for i, note in enumerate(sorted_notes):
                        start_tick = int(note.start_time * ticks_per_sec)
                        end_tick = int(note.end_time * ticks_per_sec)
                        
                        # Prevent overlap with next note
                        if i + 1 < len(sorted_notes):
                            next_start = int(sorted_notes[i + 1].start_time * ticks_per_sec)
                            end_tick = min(end_tick, next_start - 1)
                        
                        # Enforce minimum duration
                        duration = max(end_tick - start_tick, MIN_NOTE_TICKS)
                        end_tick = start_tick + duration
                        
                        # Clamp note to phrase boundaries!
                        start_tick = max(start_tick, phrase_start_tick + 1)
                        end_tick = min(end_tick, phrase_end_tick - 1)
                        
                        # Lyric event
                        lyric = note.lyric if note.lyric else '+'
                        all_events.append((start_tick, 1, ('lyric', lyric)))
                        
                        # Note on/off (clamp pitch to 36-84)
                        midi_pitch = max(36, min(84, note.midi_pitch))
                        all_events.append((start_tick, 2, ('note_on', midi_pitch, 100)))
                        all_events.append((end_tick, 4, ('note_off', midi_pitch, 0)))
                
                # Sort by (tick, priority)
                all_events.sort(key=lambda x: (x[0], x[1]))
                
                # Convert to delta times
                current_tick = 0
                for tick, _, event_data in all_events:
                    delta = max(0, tick - current_tick)
                    event_type = event_data[0]
                    
                    if event_type == 'phrase_on':
                        track.append(mido.Message('note_on', note=event_data[1], velocity=100, time=delta))
                    elif event_type == 'phrase_off':
                        track.append(mido.Message('note_off', note=event_data[1], velocity=0, time=delta))
                    elif event_type == 'lyric':
                        track.append(mido.MetaMessage('lyrics', text=event_data[1], time=delta))
                    elif event_type == 'note_on':
                        track.append(mido.Message('note_on', note=event_data[1], velocity=event_data[2], time=delta))
                    elif event_type == 'note_off':
                        track.append(mido.Message('note_off', note=event_data[1], velocity=event_data[2], time=delta))
                    
                    current_tick = tick
                
                return track
            
            # Export lead vocals
            vocals_track = export_vocal_track(lead_phrases, 'PART VOCALS')
            midi.tracks.append(vocals_track)
            
            total_notes = sum(len(p.notes) for p in lead_phrases)
            logger.info(f"Vocals: {total_notes} notes in {len(lead_phrases)} phrases")
            
            # Export harmony
            if harmony_phrases:
                harm_track = export_vocal_track(harmony_phrases, 'HARM1')
                midi.tracks.append(harm_track)
                harm_notes = sum(len(p.notes) for p in harmony_phrases)
                logger.info(f"Harmony: {harm_notes} notes")
            
            return midi
        except Exception as e:
            logger.error(f"Vocals transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def transcribe_keys(self, other_wav: Path) -> Optional[mido.MidiFile]:
        """
        Transcribe keys from other.wav stem.
        
        Only generates tracks if actual keyboard instruments are detected.
        """
        if self.keys_charter is None:
            logger.warning("Keys charter not available, skipping")
            return None
        
        logger.info("Checking for keyboard instruments...")
        try:
            notes, details = self.keys_charter.transcribe(str(other_wav), force=False)
            
            if notes is None:
                logger.info("No keyboard instruments detected, skipping keys track")
                return None
            
            logger.info(f"Keys detected! Transcribing {len(notes)} notes...")
            
            # Generate MIDI with both 5-lane and Pro Keys
            midi = self.keys_charter.export_midi(
                notes,
                output_path=None,  # Don't save, just return
                tempo_bpm=self.config.tempo_bpm,
                ticks_per_beat=self.config.ticks_per_beat,
            )
            
            logger.info(f"Keys: {len(notes)} notes (5-lane + Pro Keys tracks)")
            return midi
        except Exception as e:
            logger.error(f"Keys transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def merge_midi(
        self,
        drums_midi: Optional[mido.MidiFile],
        guitar_midi: Optional[mido.MidiFile],
        bass_midi: Optional[mido.MidiFile],
        vocals_midi: Optional[mido.MidiFile] = None,
        keys_midi: Optional[mido.MidiFile] = None,
    ) -> mido.MidiFile:
        """Merge instrument MIDIs into a single file with start buffer."""
        merged = mido.MidiFile(type=1, ticks_per_beat=self.config.ticks_per_beat)
        
        # Calculate buffer offset in ticks
        buffer_ticks = int(self.config.start_buffer_sec * self.config.tempo_bpm / 60 * self.config.ticks_per_beat)
        
        # Add tempo/sync track (first track)
        tempo_track = mido.MidiTrack()
        merged.tracks.append(tempo_track)
        tempo_track.append(mido.MetaMessage('track_name', name='TEMPO TRACK', time=0))
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(self.config.tempo_bpm), time=0))
        tempo_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        tempo_track.append(mido.MetaMessage('end_of_track', time=0))
        
        def add_track_with_buffer(midi_file, merged_file):
            """Add tracks from midi_file to merged_file with buffer offset."""
            if midi_file is None:
                return
            for track in midi_file.tracks:
                has_notes = any(m.type in ('note_on', 'note_off', 'lyrics') for m in track)
                if has_notes:
                    new_track = mido.MidiTrack()
                    first_note = True
                    for msg in track:
                        msg_copy = msg.copy()
                        # Add buffer to first NOTE event's time, not meta messages
                        if first_note and buffer_ticks > 0 and msg.type in ('note_on', 'note_off', 'lyrics'):
                            msg_copy.time += buffer_ticks
                            first_note = False
                        new_track.append(msg_copy)
                    new_track.append(mido.MetaMessage('end_of_track', time=0))
                    merged_file.tracks.append(new_track)
        
        # Add each instrument track with buffer
        add_track_with_buffer(drums_midi, merged)
        add_track_with_buffer(guitar_midi, merged)
        add_track_with_buffer(bass_midi, merged)
        add_track_with_buffer(vocals_midi, merged)
        add_track_with_buffer(keys_midi, merged)
        
        # Add EVENTS track (required by YARG for vocals)
        # Calculate song length from all tracks
        max_ticks = 0
        for track in merged.tracks:
            track_ticks = sum(msg.time for msg in track)
            max_ticks = max(max_ticks, track_ticks)
        
        events_track = mido.MidiTrack()
        events_track.append(mido.MetaMessage('track_name', name='EVENTS', time=0))
        # Add buffer to music_start marker
        events_track.append(mido.MetaMessage('text', text='[music_start]', time=buffer_ticks))
        events_track.append(mido.MetaMessage('text', text='[end]', time=max_ticks - buffer_ticks))
        events_track.append(mido.MetaMessage('end_of_track', time=0))
        merged.tracks.append(events_track)
        
        return merged
    
    def infer(
        self,
        audio_path: Path,
        output_path: Path,
        stems_dir: Optional[Path] = None,
        instruments: List[str] = None,
        artist: Optional[str] = None,
        title: Optional[str] = None,
    ):
        """
        Full inference pipeline.
        
        Args:
            audio_path: Input audio file
            output_path: Output MIDI path
            stems_dir: Pre-separated stems directory (skip Demucs if provided)
            instruments: List of instruments to transcribe ['drums', 'guitar', 'bass']
            artist: Artist name for lyrics lookup (tries to extract from path if not provided)
            title: Song title for lyrics lookup (tries to extract from path if not provided)
        """
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        
        # Store artist/title for vocals transcription
        self._artist = artist
        self._title = title
        # Try to extract from input path if not provided
        if not self._artist or not self._title:
            from src.lyrics.fetcher import extract_artist_title_from_path
            path_artist, path_title = extract_artist_title_from_path(str(audio_path))
            self._artist = self._artist or path_artist
            self._title = self._title or path_title
        
        if instruments is None:
            instruments = ['drums', 'guitar', 'bass', 'vocals', 'keys']
        
        logger.info("=" * 60)
        logger.info("STRUM - Unified Inference")
        logger.info("=" * 60)
        logger.info(f"Input: {audio_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Instruments: {instruments}")
        if self._artist and self._title:
            logger.info(f"Song: {self._artist} - {self._title}")
        
        # Auto-detect tempo if not specified
        if self.config.tempo_bpm is None:
            detected_tempo = self.detect_tempo(audio_path)
            self.config.tempo_bpm = detected_tempo
            logger.info(f"Auto-detected tempo: {detected_tempo} BPM")
        else:
            logger.info(f"Using specified tempo: {self.config.tempo_bpm} BPM")
        
        # Step 1: Get stems
        cleanup_stems = False
        if stems_dir and Path(stems_dir).exists():
            logger.info(f"Using pre-separated stems from {stems_dir}")
            stem_paths = {
                'drums': Path(stems_dir) / 'drums.wav',
                'bass': Path(stems_dir) / 'bass.wav',
                'other': Path(stems_dir) / 'other.wav',
                'vocals': Path(stems_dir) / 'vocals.wav',
            }
        else:
            # Run Demucs separation
            stems_dir = Path(tempfile.mkdtemp(prefix="strum_"))
            cleanup_stems = True
            stem_paths = self.separate_stems(audio_path, stems_dir)
        
        # Verify stems exist
        for stem, path in stem_paths.items():
            if not Path(path).exists():
                logger.warning(f"Stem {stem} not found at {path}")
        
        # Step 2: Transcribe each instrument
        drums_midi = None
        guitar_midi = None
        bass_midi = None
        vocals_midi = None
        keys_midi = None
        
        if 'drums' in instruments and stem_paths.get('drums'):
            drums_midi = self.transcribe_drums(stem_paths['drums'])
            # Free drums model GPU memory before loading other models
            self._cleanup_gpu(unload=['drums'])
        
        if 'guitar' in instruments and stem_paths.get('other') and stem_paths.get('bass'):
            try:
                guitar_midi = self.transcribe_guitar(stem_paths['other'], stem_paths['bass'])
            except Exception as e:
                logger.warning(f"Guitar transcription failed: {e}")
                guitar_midi = None
        
        if 'bass' in instruments and stem_paths.get('bass') and stem_paths.get('other'):
            try:
                bass_midi = self.transcribe_bass(stem_paths['bass'], stem_paths['other'])
            except Exception as e:
                logger.warning(f"Bass transcription failed: {e}")
                bass_midi = None
        
        # Free guitar/bass models before loading Whisper for vocals
        if 'vocals' in instruments or 'keys' in instruments:
            self._cleanup_gpu(unload=['guitar', 'bass'])
        
        if 'vocals' in instruments and stem_paths.get('vocals'):
            try:
                vocals_midi = self.transcribe_vocals(stem_paths['vocals'])
            except Exception as e:
                logger.warning(f"Vocals transcription failed (chart will have no vocals): {e}")
                vocals_midi = None
            # Free Whisper model after vocals
            self._cleanup_gpu(unload=['vocals'])
        
        if 'keys' in instruments and stem_paths.get('other'):
            keys_midi = self.transcribe_keys(stem_paths['other'])
        
        # Step 3: Merge and export
        logger.info("Merging tracks...")
        merged = self.merge_midi(drums_midi, guitar_midi, bass_midi, vocals_midi, keys_midi)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.save(output_path)
        
        logger.info(f"Chart saved to {output_path}")
        
        # Cleanup temp stems
        if cleanup_stems and stems_dir:
            shutil.rmtree(stems_dir, ignore_errors=True)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        total_tracks = len(merged.tracks)
        logger.info(f"Total tracks: {total_tracks}")
        
        return merged
    
    def chart(self, audio_path: Path, output_dir: Path, instruments: List[str] = None):
        """
        Generate a full Clone Hero / YARG compatible chart package.
        
        Creates:
        - song.ogg: Converted audio (with optional start buffer)
        - song.ini: Song metadata (from MusicBrainz if available)
        - notes.mid: Chart with all instruments
        - video.mp4: YouTube video background (if enabled)
        
        Args:
            audio_path: Input audio file (MP3, WAV, etc.)
            output_dir: Output directory (will be created as "Artist - Title" folder)
            instruments: List of instruments to chart
        """
        import subprocess
        from datetime import datetime
        
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        
        if instruments is None:
            instruments = ['drums', 'guitar', 'bass', 'vocals']
        
        logger.info("=" * 60)
        logger.info("STRUM - Full Chart Generation")
        logger.info("=" * 60)
        logger.info(f"Input: {audio_path}")
        logger.info(f"Output folder: {output_dir}")
        logger.info(f"Instruments: {instruments}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract artist/title from filename
        name = audio_path.stem
        if " - " in name:
            parts = name.split(" - ", 1)
            artist = parts[0].strip()
            title = parts[1].strip()
            for suffix in ["(Lyrics)", "(Official Video)", "(Audio)", "(Official Audio)", "[Official Video]", "(Official Music Video)"]:
                title = title.replace(suffix, "").strip()
        else:
            artist = "Unknown Artist"
            title = name
        
        # Step 2: Auto-detect tempo if not specified
        if self.config.tempo_bpm is None:
            self.config.tempo_bpm = self.detect_tempo(audio_path)
        else:
            logger.info(f"Using specified tempo: {self.config.tempo_bpm} BPM")
        
        # Step 3: Fetch metadata from MusicBrainz
        metadata = {'album': '', 'year': '', 'genre': ''}
        if self.config.fetch_metadata and artist != "Unknown Artist":
            metadata = self.fetch_song_metadata(artist, title)
            # Use fetched artist/title if better
            if metadata.get('artist'):
                artist = metadata.get('artist', artist)
        
        # Step 4: Convert audio to song.ogg with start buffer
        song_ogg = output_dir / "song.ogg"
        buffer_sec = self.config.start_buffer_sec
        
        logger.info(f"Converting audio to {song_ogg} (adding {buffer_sec}s start buffer)...")
        try:
            # Add silence at the beginning using ffmpeg's adelay filter
            delay_ms = int(buffer_sec * 1000)
            result = subprocess.run(
                [
                    'ffmpeg', '-y',
                    '-i', str(audio_path),
                    '-af', f'adelay={delay_ms}|{delay_ms}',  # Delay both channels
                    '-vn', '-c:a', 'libvorbis', '-q:a', '6',
                    str(song_ogg)
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error(f"FFmpeg failed: {result.stderr}")
                raise RuntimeError("Audio conversion failed")
            logger.info(f"Created {song_ogg}")
        except FileNotFoundError:
            logger.warning("FFmpeg not found, copying original audio without buffer")
            shutil.copy(audio_path, output_dir / f"song{audio_path.suffix}")
            buffer_sec = 0  # No buffer applied
        
        # Step 5: Download YouTube video if enabled
        video_path = None
        if self.config.download_video:
            video_path = self.download_youtube_video(
                artist, title,
                output_dir / "video.mp4",
                self.config.video_search_query,
            )
        
        # Step 6: Create song.ini with metadata
        song_ini = output_dir / "song.ini"
        
        diff_drums = "3" if 'drums' in instruments else "-1"
        diff_guitar = "3" if 'guitar' in instruments else "-1"
        diff_bass = "3" if 'bass' in instruments else "-1"
        diff_vocals = "3" if 'vocals' in instruments else "-1"
        diff_vocals_harm = "3" if 'vocals' in instruments else "-1"
        diff_keys = "3" if 'keys' in instruments else "-1"
        
        ini_content = f"""[song]
name = {title}
artist = {artist}
charter = STRUM
album = {metadata.get('album', '')}
year = {metadata.get('year', '')}
genre = {metadata.get('genre', '')}
difficulty = 0
preview_start_time = {int(buffer_sec + 30) * 1000}
icon = 
loading_phrase = Generated by STRUM
delay = 0
diff_drums = {diff_drums}
diff_drums_real = {diff_drums}
diff_guitar = {diff_guitar}
diff_bass = {diff_bass}
diff_vocals = {diff_vocals}
diff_vocals_harm = {diff_vocals_harm}
diff_keys = {diff_keys}
"""
        if video_path and video_path.exists():
            ini_content += f"video = video.mp4\n"
        song_ini.write_text(ini_content, encoding='utf-8')
        logger.info(f"Created {song_ini}")
        logger.info(f"  Artist: {artist}")
        logger.info(f"  Title: {title}")
        
        # Step 4: Generate chart with MIDI output
        notes_mid = output_dir / "notes.mid"
        self.infer(
            audio_path=audio_path,
            output_path=notes_mid,
            instruments=instruments,
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("CHART COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output folder: {output_dir}")
        logger.info(f"  song.ogg:  Audio")
        logger.info(f"  song.ini:  Metadata")
        logger.info(f"  notes.mid: Chart")
        
        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='STRUM - Generate game charts from audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (Demucs + all instruments)
    python -m src.inference.unified song.mp3 -o chart.mid
    
    # With pre-separated stems
    python -m src.inference.unified song.mp3 -o chart.mid --stems-dir ./stems/
    
    # Guitar only
    python -m src.inference.unified song.mp3 -o chart.mid --instruments guitar
    
    # Drums + bass
    python -m src.inference.unified song.mp3 -o chart.mid --instruments drums bass
        """
    )
    
    parser.add_argument('input', type=str, help='Input audio file (WAV, MP3, etc.)')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output MIDI file')
    parser.add_argument('--stems-dir', type=str, help='Directory with pre-separated stems')
    parser.add_argument('--instruments', nargs='+', default=['drums', 'guitar', 'bass', 'vocals', 'keys'],
                        choices=['drums', 'guitar', 'bass', 'vocals', 'keys'],
                        help='Instruments to transcribe')
    parser.add_argument('--tempo', type=float, default=None, help='Chart tempo BPM (auto-detect if not specified)')
    parser.add_argument('--start-buffer', type=float, default=2.0,
                        help='Seconds of silence before chart starts (default: 2.0)')
    parser.add_argument('--no-metadata', action='store_true',
                        help='Disable MusicBrainz metadata fetch')
    parser.add_argument('--download-video', action='store_true',
                        help='Download YouTube video as background')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    # Checkpoint paths
    parser.add_argument('--drums-checkpoint', type=str, 
                        default='checkpoints/drums_v6.2/best.pt',
                        help='Drums model checkpoint')
    parser.add_argument('--guitar-checkpoint', type=str,
                        default='checkpoints/guitar_onset/best.pt', 
                        help='Guitar onset model checkpoint')
    
    # Thresholds
    parser.add_argument('--drums-threshold', type=float, default=0.6)
    parser.add_argument('--guitar-threshold', type=float, default=0.35)
    
    # Guitar/Bass quantization
    parser.add_argument('--quantize-strength', type=float, default=0.8,
                        help='Beat quantization strength: 0=none, 1=hard snap (default: 0.8)')
    parser.add_argument('--quantize-grid', type=str, default='1/16',
                        choices=['1/4', '1/8', '1/16', '1/32'],
                        help='Beat subdivision for quantization (default: 1/16)')
    
    # Vocals timing
    parser.add_argument('--vocals-offset', type=float, default=-0.05,
                        help='Vocals base timing offset in seconds, negative=earlier (default: -0.05)')
    parser.add_argument('--no-dynamic-alignment', action='store_true',
                        help='Disable dynamic onset-based vocal alignment')
    parser.add_argument('--alignment-tolerance', type=float, default=0.15,
                        help='Max time (seconds) to shift words to match vocal onsets (default: 0.15)')
    
    # Lyrics lookup
    parser.add_argument('--artist', type=str, help='Artist name for lyrics lookup')
    parser.add_argument('--title', type=str, help='Song title for lyrics lookup')
    
    args = parser.parse_args()
    
    # Build config
    config = InferenceConfig(
        drums_checkpoint=args.drums_checkpoint,
        guitar_checkpoint=args.guitar_checkpoint,
        bass_checkpoint=args.guitar_checkpoint,  # Reuse guitar model
        drums_threshold=args.drums_threshold,
        guitar_threshold=args.guitar_threshold,
        bass_threshold=args.guitar_threshold,
        guitar_quantize_strength=args.quantize_strength,
        guitar_quantize_grid=args.quantize_grid,
        vocals_timing_offset=args.vocals_offset,
        vocals_dynamic_alignment=not args.no_dynamic_alignment,
        vocals_alignment_tolerance=args.alignment_tolerance,
        tempo_bpm=args.tempo,
        start_buffer_sec=args.start_buffer,
        fetch_metadata=not args.no_metadata,
        download_video=args.download_video,
    )
    
    # Run inference
    engine = UnifiedInference(config=config, device=args.device)
    engine.infer(
        audio_path=Path(args.input),
        output_path=Path(args.output),
        stems_dir=Path(args.stems_dir) if args.stems_dir else None,
        instruments=args.instruments,
        artist=args.artist,
        title=args.title,
    )


if __name__ == '__main__':
    main()
