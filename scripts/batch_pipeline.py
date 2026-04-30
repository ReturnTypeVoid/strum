"""
Batch Pipeline - Process multiple songs through the full STRUM pipeline.

Takes a folder of audio files and creates complete Clone Hero/YARG chart folders
with all instruments: Drums, Guitar, Bass, Vocals, Keys (including Pro Keys).

Usage:
    python scripts/batch_pipeline.py input/ output/
    python scripts/batch_pipeline.py input/ output/ --no-vocals --no-keys
    python scripts/batch_pipeline.py input/ output/ --continue  # Resume interrupted batch
"""

import argparse
import sys
from pathlib import Path

# Add project root and scripts dir to path for imports
_project_root = Path(__file__).parent.parent
_scripts_dir = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

import shutil
import subprocess
import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import librosa
import mido

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class SongResult:
    """Result of processing a single song."""
    input_path: str
    output_path: str
    success: bool
    error: Optional[str] = None
    drums_hits: int = 0
    guitar_notes: int = 0
    bass_notes: int = 0
    vocals_phrases: int = 0
    keys_notes: int = 0


class BatchPipeline:
    """
    Complete batch pipeline for STRUM.
    
    Processes audio files through:
    1. Demucs stem separation
    2. Drums transcription (V11 model with tom/cymbal correction)
    3. Guitar transcription (Basic Pitch + rules)
    4. Bass transcription (Basic Pitch + rules)
    5. Vocals transcription (Whisper + pitch)
    6. Keys transcription (Basic Pitch)
    7. Chart enhancement (Star Power, difficulty, sections)
    8. Song.ini generation
    """
    
    def __init__(
        self,
        output_dir: Path,
        drums_checkpoint: str = "checkpoints/drums_v11/best.pt",
        demucs_model: str = "htdemucs_6s",  # 6-stem model: drums, bass, vocals, guitar, piano, other
        include_drums: bool = True,
        include_guitar: bool = True,
        include_bass: bool = True,
        include_vocals: bool = True,
        include_keys: bool = False,
        include_video: bool = False,
        device: str = None,
        use_v11: bool = True,  # Use V11 with tom/cymbal correction
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.drums_checkpoint = drums_checkpoint
        self.demucs_model = demucs_model
        self.include_drums = include_drums
        self.include_guitar = include_guitar
        self.include_bass = include_bass
        self.include_vocals = include_vocals
        self.include_keys = include_keys
        self.include_video = include_video
        self.device = device
        self.use_v11 = use_v11
        
        # Lazy-load engines
        self._drums_models = None
        self._vocals_charter = None
        self._keys_charter = None
        self.tomcym = None
    
    @property
    def drums_models(self):
        """Lazy load drums V14 onset detector + ensemble + phase3 + tom_refinement."""
        if self._drums_models is None and self.include_drums:
            try:
                from batch_infer_hybrid import (
                    load_v14_onset_detector, load_ensemble, load_tomcym_classifier,
                    load_phase3_model, load_tom_refinement,
                )
                import torch
                device = torch.device(self.device) if self.device else (
                    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                )
                logger.info("Loading drums V14 onset detector + ensemble + phase3 + tom_refinement...")
                v14_model = load_v14_onset_detector(device)
                ensemble = load_ensemble(device)
                tomcym = load_tomcym_classifier(device)
                phase3_model = load_phase3_model(device)
                tom_refinement = load_tom_refinement(device)
                self.tomcym = tomcym
                self._drums_models = {
                    "v14": v14_model,
                    "ensemble": ensemble,
                    "phase3": phase3_model,
                    "tom_refinement": tom_refinement,
                    "device": device,
                }
            except Exception as e:
                logger.warning(f"  ⚠ Could not load drums models: {e}")
                self._drums_models = {}
        return self._drums_models
    
    @property
    def vocals_charter(self):
        """Lazy load vocals charter."""
        if self._vocals_charter is None and self.include_vocals:
            from vocals_charter import VocalsCharter
            logger.info("Loading vocals charter (Whisper)...")
            self._vocals_charter = VocalsCharter(whisper_model="medium")
        return self._vocals_charter
    
    @property
    def keys_charter(self):
        """Lazy load keys charter."""
        if self._keys_charter is None and self.include_keys:
            from keys_charter import KeysCharter
            logger.info("Loading keys charter...")
            # Tightened threshold: 0.3 -> 0.7. Default 0.3 fires on any tonal
            # content in the 'other' stem (guitar+vocals bleed), producing ~4x
            # GT keys note count across our held-out set.
            self._keys_charter = KeysCharter(detection_threshold=0.7)
        return self._keys_charter
    
    def _quick_musicbrainz_check(self, artist: str, title: str) -> bool:
        """
        Quick MusicBrainz lookup to verify artist/title combo exists.
        Returns True if a recording is found.
        """
        import urllib.request
        import urllib.parse
        import json
        
        try:
            query = f'recording:"{title}" AND artist:"{artist}"'
            url = f"https://musicbrainz.org/ws/2/recording?query={urllib.parse.quote(query)}&fmt=json&limit=1"
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'STRUM/1.0 (https://github.com/strum)'
            })
            
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            # Check if we got a result with matching artist
            if data.get('recordings'):
                recording = data['recordings'][0]
                # Verify the artist name matches somewhat
                if recording.get('artist-credit'):
                    found_artist = recording['artist-credit'][0].get('name', '').lower()
                    if artist.lower() in found_artist or found_artist in artist.lower():
                        return True
            return False
        except Exception:
            return False
    
    def parse_filename(self, path: Path) -> Tuple[str, str]:
        """Parse artist and title from filename, validating via MusicBrainz if possible."""
        import time
        
        name = path.stem
        
        # Try "X - Y" format
        if " - " in name:
            parts = name.split(" - ", 1)
            part1, part2 = parts[0].strip(), parts[1].strip()
            
            # Try both orders and validate with MusicBrainz
            # Order 1: "Title - Artist" (part1=title, part2=artist)
            order1_valid = self._quick_musicbrainz_check(part2, part1)
            
            if order1_valid:
                logger.debug(f"MusicBrainz confirmed: Artist='{part2}', Title='{part1}'")
                return part2, part1  # artist, title
            
            time.sleep(1)  # Rate limit
            
            # Order 2: "Artist - Title" (part1=artist, part2=title)
            order2_valid = self._quick_musicbrainz_check(part1, part2)
            
            if order2_valid:
                logger.debug(f"MusicBrainz confirmed: Artist='{part1}', Title='{part2}'")
                return part1, part2  # artist, title
            
            # Neither validated - default to "Title - Artist" format
            logger.debug(f"MusicBrainz couldn't validate, defaulting to Title-Artist format")
            return part2, part1  # artist, title
        
        # Try "(Artist)" at end
        match = re.search(r'^(.+?)\s*\(([^)]+)\)\s*$', name)
        if match:
            return match.group(2).strip(), match.group(1).strip()
        
        return "Unknown Artist", name
    
    def analyze_audio(self, audio_path: Path, artist: str = '', title: str = '') -> Dict:
        """Analyze audio for tempo, duration, and metadata."""
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        duration_sec = len(y) / sr
        
        # Use grid-alignment BPM refinement from drums pipeline if available
        try:
            from batch_infer_hybrid import analyze_audio as drums_analyze
            audio_info = drums_analyze(audio_path)
            tempo = audio_info["tempo_bpm"]
            logger.info(f"  BPM (grid-aligned): {tempo:.1f}")
        except Exception:
            # Fallback: librosa.feature.rhythm.tempo (not beat_track which gives subharmonics)
            tempo = librosa.feature.rhythm.tempo(y=y, sr=sr)
            if hasattr(tempo, '__len__'):
                tempo = float(tempo[0])
            logger.info(f"  BPM (librosa): {tempo:.1f}")
        
        # Find good preview point (skip intro)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        if len(onset_frames) > 10:
            preview_frame = onset_frames[10]
            preview_sec = librosa.frames_to_time(preview_frame, sr=sr)
        else:
            preview_sec = min(30, duration_sec * 0.2)
        
        # Extract metadata from file
        metadata = self._extract_metadata(audio_path)
        
        # Fallback to MusicBrainz for missing metadata
        if artist and title and (not metadata['album'] or not metadata['year'] or not metadata['genre']):
            mb_metadata = self._fetch_musicbrainz_metadata(artist, title)
            if not metadata['album'] and mb_metadata['album']:
                metadata['album'] = mb_metadata['album']
            if not metadata['year'] and mb_metadata['year']:
                metadata['year'] = mb_metadata['year']
            if not metadata['genre'] and mb_metadata['genre']:
                metadata['genre'] = mb_metadata['genre']
        
        return {
            'tempo_bpm': round(tempo, 2),  # Keep 0.01 BPM precision (rounding to int caused 500ms+ drift)
            'duration_ms': int(duration_sec * 1000),
            'duration_sec': duration_sec,
            'preview_start_ms': int(preview_sec * 1000),
            'album': metadata.get('album', ''),
            'year': metadata.get('year', ''),
            'genre': metadata.get('genre', ''),
        }
    
    def _extract_metadata(self, audio_path: Path) -> Dict[str, str]:
        """Extract album, year, and genre from audio file metadata."""
        metadata = {'album': '', 'year': '', 'genre': ''}
        
        try:
            from mutagen import File
            from mutagen.id3 import ID3
            from mutagen.mp3 import MP3
            from mutagen.flac import FLAC
            from mutagen.mp4 import MP4
            
            audio = File(str(audio_path))
            if audio is None:
                return metadata
            
            # MP3 with ID3 tags
            if isinstance(audio, MP3) or hasattr(audio, 'tags'):
                tags = audio.tags if hasattr(audio, 'tags') else audio
                if tags:
                    # Album
                    for key in ['TALB', 'album', '\xa9alb']:
                        if key in tags:
                            val = tags[key]
                            metadata['album'] = str(val[0] if hasattr(val, '__getitem__') else val).strip()
                            break
                    
                    # Year/Date
                    for key in ['TDRC', 'TYER', 'date', '\xa9day']:
                        if key in tags:
                            val = tags[key]
                            year_str = str(val[0] if hasattr(val, '__getitem__') else val).strip()
                            # Extract just the year (first 4 digits)
                            import re
                            year_match = re.search(r'(\d{4})', year_str)
                            if year_match:
                                metadata['year'] = year_match.group(1)
                            break
                    
                    # Genre
                    for key in ['TCON', 'genre', '\xa9gen']:
                        if key in tags:
                            val = tags[key]
                            metadata['genre'] = str(val[0] if hasattr(val, '__getitem__') else val).strip()
                            break
            
            # FLAC
            elif isinstance(audio, FLAC):
                metadata['album'] = audio.get('album', [''])[0]
                metadata['year'] = audio.get('date', [''])[0][:4] if audio.get('date') else ''
                metadata['genre'] = audio.get('genre', [''])[0]
            
            # MP4/M4A
            elif isinstance(audio, MP4):
                metadata['album'] = audio.get('\xa9alb', [''])[0]
                metadata['year'] = audio.get('\xa9day', [''])[0][:4] if audio.get('\xa9day') else ''
                metadata['genre'] = audio.get('\xa9gen', [''])[0]
                
        except Exception as e:
            logger.debug(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def _fetch_musicbrainz_metadata(self, artist: str, title: str) -> Dict[str, str]:
        """
        Fetch album, year, and genre from MusicBrainz API.
        Free API, no auth required, but rate limited (1 req/sec).
        """
        import urllib.request
        import urllib.parse
        import json
        import time
        
        metadata = {'album': '', 'year': '', 'genre': ''}
        
        try:
            # Search for recording by artist and title
            query = f'recording:"{title}" AND artist:"{artist}"'
            url = f"https://musicbrainz.org/ws/2/recording?query={urllib.parse.quote(query)}&fmt=json&limit=1"
            
            req = urllib.request.Request(url, headers={
                'User-Agent': 'STRUM/1.0 (https://github.com/strum)'
            })
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
            
            if data.get('recordings'):
                recording = data['recordings'][0]
                
                # Get release (album) info
                if recording.get('releases'):
                    release = recording['releases'][0]
                    metadata['album'] = release.get('title', '')
                    
                    # Get release date
                    date = release.get('date', '')
                    if date and len(date) >= 4:
                        metadata['year'] = date[:4]
                
                # Get genre/tags from artist
                if recording.get('artist-credit'):
                    artist_id = recording['artist-credit'][0].get('artist', {}).get('id')
                    if artist_id:
                        time.sleep(1)  # Rate limit
                        tag_url = f"https://musicbrainz.org/ws/2/artist/{artist_id}?inc=tags&fmt=json"
                        tag_req = urllib.request.Request(tag_url, headers={
                            'User-Agent': 'STRUM/1.0 (https://github.com/strum)'
                        })
                        with urllib.request.urlopen(tag_req, timeout=10) as tag_response:
                            tag_data = json.loads(tag_response.read().decode('utf-8'))
                        
                        # Get most popular tag as genre
                        tags = tag_data.get('tags', [])
                        if tags:
                            # Sort by count, get top tag
                            tags.sort(key=lambda t: t.get('count', 0), reverse=True)
                            metadata['genre'] = tags[0].get('name', '').title()
                            
        except Exception as e:
            logger.debug(f"MusicBrainz lookup failed: {e}")
        
        return metadata

    def separate_stems(self, audio_path: Path, work_dir: Path) -> Dict[str, Path]:
        """Separate stems using Demucs Python API (avoids torchaudio save bug).

        Runs the configured `demucs_model` (default htdemucs_6s) for guitar/bass/
        vocals/keys/other. If `STRUM_DRUMS_DEMUCS` is set (default 'htdemucs_ft'),
        runs that model in addition and uses ITS drum stem — matches what the
        production drums pipeline (`batch_infer_hybrid`) and tom_refinement_demucs
        checkpoint were tuned on.
        """
        import os
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torch

        logger.info("  Separating stems with Demucs...")

        stem_dir = work_dir / "stems"
        stem_dir.mkdir(parents=True, exist_ok=True)

        drums_demucs = os.environ.get("STRUM_DRUMS_DEMUCS", "")
        use_separate_drums = bool(drums_demucs) and drums_demucs != self.demucs_model

        # Check if stems already exist
        expected_stems = ["drums", "bass", "other", "vocals"]
        if all((stem_dir / f"{s}.wav").exists() for s in expected_stems):
            logger.info("    Stems already exist, skipping separation")
            stems = {s: stem_dir / f"{s}.wav" for s in expected_stems}
            for extra in ["guitar", "piano"]:
                p = stem_dir / f"{extra}.wav"
                if p.exists():
                    stems[extra] = p
            return stems

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def _run_model(model_name: str):
            model = get_model(model_name)
            model.eval()
            model.to(device)
            y, sr_orig = librosa.load(str(audio_path), sr=None, mono=False)
            if y.ndim == 1:
                y = np.stack([y, y])
            target_sr = model.samplerate
            if sr_orig != target_sr:
                y = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)
            waveform = torch.from_numpy(y).float()
            ref = waveform.mean(0)
            waveform_norm = (waveform - ref.mean()) / ref.std()
            sources = apply_model(model, waveform_norm[None].to(device), device=device)[0]
            sources = sources * ref.std() + ref.mean()
            return model.sources, sources, target_sr

        # Primary model (6s for guitar/piano/etc)
        names, sources, target_sr = _run_model(self.demucs_model)
        stems = {}
        for i, name in enumerate(names):
            stem = sources[i].cpu().numpy()
            path = stem_dir / f"{name}.wav"
            sf.write(str(path), stem.T, target_sr)
            stems[name] = path
        logger.info(f"    Separated stems ({self.demucs_model}): {list(stems.keys())}")

        # Drums-specialist pass (htdemucs_ft) — overwrites drums.wav with the
        # higher-quality drum stem that production drums pipeline expects.
        if use_separate_drums:
            try:
                logger.info(f"    Re-separating drums with {drums_demucs}...")
                names_d, sources_d, target_sr_d = _run_model(drums_demucs)
                if "drums" in names_d:
                    idx = list(names_d).index("drums")
                    stem = sources_d[idx].cpu().numpy()
                    path = stem_dir / "drums.wav"
                    sf.write(str(path), stem.T, target_sr_d)
                    stems["drums"] = path
                    logger.info(f"    Drums stem replaced from {drums_demucs}")
            except Exception as e:
                logger.warning(f"    ⚠ {drums_demucs} drums pass failed, keeping {self.demucs_model} drums: {e}")

        return stems
    
    def transcribe_drums(self, drums_stem: Path, tempo_bpm: float):
        """Transcribe drums using the production batch_infer_hybrid pipeline.

        Mirrors `batch_infer_hybrid.process_song` post-separation flow so that
        batch_pipeline drums quality matches the standalone production pipeline.
        """
        if not self.include_drums:
            return None

        logger.info("  Transcribing drums (V14 hybrid + phase3 + tom_refinement)...")

        try:
            models = self.drums_models
            if not models or "v14" not in models:
                logger.warning("  ⚠ Drums models not loaded, skipping")
                return None

            from batch_infer_hybrid import (
                detect_onsets_v14, extract_onset_windows, classify_onsets_ensemble,
                build_context_vectors, build_chart, postprocess_chart,
                _compute_spectral_centroid_features, _compute_onset_rms,
                run_phase3_inference, phase3_reclassify, phase3_onset_rescue,
                phase3_cymbal_cooccurrence_rescue, spectral_reclassify,
                apply_tom_refinement_filter,
                DEFAULT_CLASS_THRESHOLDS,
                PHASE3_RECLASS_ENABLED, PHASE3_RESCUE_ENABLED,
                PHASE3_COOCCURRENCE_ENABLED, SPECTRAL_RECLASS_ENABLED,
            )
            from src.preprocessing.parsers.midi_parser import (
                DrumChart, TimeSignature, TempoEvent,
            )
            import numpy as np

            v14_model = models["v14"]
            ensemble = models["ensemble"]
            phase3_model = models.get("phase3")
            tom_refinement = models.get("tom_refinement")
            device = models["device"]

            # Stage 1: V14 onset detection (+ class probs + MC onset probs)
            onset_times_ms, v14_class_probs, _mc_onset_probs = detect_onsets_v14(
                v14_model, drums_stem, device, onset_threshold=0.4
            )
            logger.info(f"    Stage 1: {len(onset_times_ms)} onsets")

            # Use pipeline-level BPM (from full mix) for tempo events
            tempo_events = [TempoEvent(tick=0, tempo_bpm=tempo_bpm, time_ms=0.0)]

            if not onset_times_ms:
                chart = DrumChart(
                    hits=[],
                    tempo_events=tempo_events,
                    time_signatures=[TimeSignature(tick=0, numerator=4, denominator=4, time_ms=0.0)],
                    ticks_per_beat=480,
                )
            else:
                # Stage 2: extract windows + ensemble classify (two-pass context)
                any_needs_cqt = any(e["needs_cqt"] for e in ensemble)
                windows = extract_onset_windows(
                    drums_stem, onset_times_ms, needs_cqt=any_needs_cqt
                )

                context = build_context_vectors(onset_times_ms)
                logits_pass1 = classify_onsets_ensemble(ensemble, windows, context, device)
                probs_pass1 = 1.0 / (1.0 + np.exp(-logits_pass1))
                context = build_context_vectors(onset_times_ms, probs_pass1)
                logits = classify_onsets_ensemble(ensemble, windows, context, device)
                probs = 1.0 / (1.0 + np.exp(-logits))

                # Spectral + RMS features
                spectral_centroids, spectral_high_pcts = _compute_spectral_centroid_features(
                    drums_stem, onset_times_ms
                )
                onset_rms = _compute_onset_rms(drums_stem, onset_times_ms)

                # Build chart with production thresholds
                chart = build_chart(
                    onset_times_ms, probs, windows["valid_mask"],
                    tempo_events=tempo_events,
                    thresholds=DEFAULT_CLASS_THRESHOLDS,
                    spectral_centroids=spectral_centroids,
                    spectral_high_pcts=spectral_high_pcts,
                    onset_rms=onset_rms,
                    probs_pass1=probs_pass1,
                    v14_class_probs=v14_class_probs,
                )

            # Post-processing
            if chart.hits:
                chart = postprocess_chart(chart)

            # Phase 3 reclassification + onset rescue + co-occurrence rescue
            phase3_probs = None
            if phase3_model is not None and (PHASE3_RECLASS_ENABLED or PHASE3_RESCUE_ENABLED):
                phase3_probs = run_phase3_inference(phase3_model, drums_stem, device)
                if PHASE3_RECLASS_ENABLED:
                    chart = phase3_reclassify(chart, phase3_probs)
            if phase3_probs is not None and PHASE3_RESCUE_ENABLED:
                chart = phase3_onset_rescue(chart, phase3_probs)
            if phase3_probs is not None and PHASE3_COOCCURRENCE_ENABLED:
                chart = phase3_cymbal_cooccurrence_rescue(chart, phase3_probs)

            # Spectral reclassification (HiHat↔Snare, Crash↔Ride)
            if SPECTRAL_RECLASS_ENABLED and chart.hits:
                chart = spectral_reclassify(chart, drums_stem)

            # Tom refinement filter (verifies all tom hits, flips false-positives)
            if tom_refinement is not None:
                chart = apply_tom_refinement_filter(chart, drums_stem, tom_refinement, device)

            # FINAL hand-cap re-pass: phase3_onset_rescue,
            # phase3_cymbal_cooccurrence_rescue, complete_cymbal_patterns and
            # spectral_reclassify can all add or move hits AFTER the
            # postprocess_chart hand-cap step ran, leaving 3+ hand notes at
            # the same tick. Rerun resolve_playability to cap to 2 hands.
            try:
                from scripts.chart_postprocess import resolve_playability, protect_tom_fills
                if chart.hits:
                    chart = protect_tom_fills(chart)
                    chart = resolve_playability(chart)
            except Exception as _e:
                logger.warning(f"  Final hand-cap pass failed: {_e}")

            logger.info(f"    Drums: {len(chart.hits)} Expert hits")
            return chart

        except Exception as e:
            logger.warning(f"  ⚠ Drums transcription failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def transcribe_guitar(self, other_stem: Path, tempo_bpm: float, full_mix: Path | None = None):
        """Transcribe guitar.

        Backend selection (env STRUM_GUITAR_BACKEND, default 'hybrid'):
          * 'hybrid' (default): V2 onset CRNN (F1 0.81) + basic-pitch
            polyphonic pitch transcription + rule-based pitch→fret. Sidesteps
            the weak V2 fret head (Event F1 0.17). Runs on the htdemucs_6s
            guitar stem so basic-pitch transcribes only guitar pitches and
            the onset CRNN doesn't fire on vocals/drums.
          * 'neural': V2 onset CRNN + V2 fret classifier (full mix).
          * 'rule': legacy librosa+pYIN+rule on the Demucs other stem.
        """
        if not self.include_guitar:
            return None

        import os as _os_g
        backend = _os_g.environ.get("STRUM_GUITAR_BACKEND", "hybrid").lower()
        # Legacy compat
        if _os_g.environ.get("STRUM_GUITAR_RULE", "0") == "1":
            backend = "rule"

        # Source selection per backend:
        #   hybrid → guitar stem (clean signal for basic-pitch + onset model)
        #   neural → full mix (model trained on full mix)
        #   rule   → other stem (legacy)
        if backend == "neural" and full_mix is not None:
            src_path = full_mix
        else:
            src_path = other_stem  # hybrid uses the passed-in guitar stem
        logger.info(f"  Transcribing guitar ({backend}, src={src_path.name})...")

        try:
            if backend == "rule":
                from src.inference.guitar_bass import transcribe_guitar
                chart = transcribe_guitar(src_path, tempo_bpm=tempo_bpm, confidence_threshold=0.3)
            elif backend == "neural":
                from src.inference.guitar_neural import transcribe_guitar_neural
                chart = transcribe_guitar_neural(src_path, tempo_bpm=tempo_bpm)
            else:  # hybrid
                from src.inference.guitar_hybrid_v2 import transcribe_guitar_hybrid
                chart = transcribe_guitar_hybrid(src_path, tempo_bpm=tempo_bpm)
            logger.info(f"    Guitar: {len(chart.notes)} notes + {len(chart.chords)} chords")
            return chart
        except Exception as e:
            logger.warning(f"  ⚠ Guitar transcription failed: {e}")
            return None
    
    def transcribe_bass(self, bass_stem: Path, tempo_bpm: float, full_mix: Path | None = None):
        """Transcribe bass using the rule-based pipeline on the bass stem.

        The neural V2 model was trained on guitar — running it on the full mix
        for bass yields the *same* output as guitar (same model, same input).
        Until a bass-specific model exists, bass stays on the rule-based
        librosa+pYIN pipeline operating on Demucs's bass.wav stem.
        STRUM_BASS_NEURAL=1 forces the (broken) neural path for A/B testing.
        """
        if not self.include_bass:
            return None

        import os as _os_b
        force_neural = _os_b.environ.get("STRUM_BASS_NEURAL", "0") == "1"
        if force_neural and full_mix is not None:
            backend = "neural(forced, =guitar)"
            src_path = full_mix
        else:
            backend = "rule"
            src_path = bass_stem
        logger.info(f"  Transcribing bass ({backend}, src={src_path.name}, no chords per C3 rules)...")

        try:
            if force_neural and full_mix is not None:
                from src.inference.guitar_neural import transcribe_guitar_neural
                chart = transcribe_guitar_neural(src_path, tempo_bpm=tempo_bpm, is_bass=True)
                chart.chords = []
            else:
                from src.inference.guitar_bass import transcribe_bass
                chart = transcribe_bass(src_path, tempo_bpm=tempo_bpm, confidence_threshold=0.4)
            logger.info(f"    Bass: {len(chart.notes)} notes (no chords per C3 rules)")
            return chart
        except Exception as e:
            logger.warning(f"  ⚠ Bass transcription failed: {e}")
            return None
    
    def transcribe_vocals(self, vocals_stem: Path, artist: str, title: str):
        """Transcribe vocals using Whisper + pitch detection."""
        if not self.include_vocals:
            return None, None
        
        logger.info("  Transcribing vocals...")
        
        try:
            lead_phrases, harmony_phrases = self.vocals_charter.transcribe(
                str(vocals_stem),
                artist=artist,
                title=title
            )
            return lead_phrases, harmony_phrases
        except Exception as e:
            logger.warning(f"  ⚠ Vocals transcription failed: {e}")
            return None, None
    
    def transcribe_keys(self, other_stem: Path):
        """Transcribe keys/keyboards from other stem.

        Uses the keyboard-presence detector (force=False) so we don't hallucinate
        keys onto songs that don't have a keyboard part. Most songs in our test
        set lack PART KEYS — gating here avoids massive precision loss.
        """
        if not self.include_keys:
            return None

        logger.info("  Transcribing keys...")

        try:
            notes, details = self.keys_charter.transcribe(str(other_stem), force=False)
            if notes is None:
                logger.info("    Keys: no keyboard detected, skipping")
            return notes
        except Exception as e:
            logger.warning(f"  ⚠ Keys transcription failed: {e}")
            return None
    
    def create_combined_midi(
        self,
        output_path: Path,
        tempo_bpm: float,
        drums_chart=None,
        guitar_chart=None,
        bass_chart=None,
        lead_phrases=None,
        harmony_phrases=None,
        keys_notes=None,
        ticks_per_beat: int = 480,
    ):
        """Create combined MIDI with all instruments."""
        from src.inference.guitar_bass import reduce_to_difficulty as reduce_guitar_difficulty, FRET_NOTE_OFFSETS
        from src.export.midi import (
            DIFFICULTY_NOTE_OFFSETS,
            _compute_hit_ticks,
            _get_midi_note,
            _get_tom_marker,
            _reduce_to_hard,
            _reduce_to_medium,
            _reduce_to_easy,
        )
        
        mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
        ticks_per_sec = ticks_per_beat * tempo_bpm / 60
        
        # Tempo track
        tempo_track = mido.MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.name = "TEMPO TRACK"
        tempo_us = int(60_000_000 / tempo_bpm)
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
        tempo_track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        tempo_track.append(mido.MetaMessage("end_of_track", time=0))
        
        # Beat track (for practice mode)
        beat_track = mido.MidiTrack()
        mid.tracks.append(beat_track)
        beat_track.name = "BEAT"
        beat_track.append(mido.MetaMessage("end_of_track", time=0))
        
        # Events track (for sections - will be populated by enhancer)
        events_track = mido.MidiTrack()
        mid.tracks.append(events_track)
        events_track.name = "EVENTS"
        events_track.append(mido.MetaMessage("end_of_track", time=0))
        
        # PART DRUMS
        if drums_chart and drums_chart.hits:
            drums_track = self._create_drums_track(drums_chart, ticks_per_beat)
            mid.tracks.append(drums_track)
            logger.info(f"  Added PART DRUMS ({len(drums_chart.hits)} expert hits)")
        
        # PART GUITAR
        if guitar_chart and guitar_chart.notes:
            guitar_track = self._create_guitar_track(guitar_chart, "PART GUITAR", ticks_per_beat)
            mid.tracks.append(guitar_track)
            logger.info(f"  Added PART GUITAR ({len(guitar_chart.notes)} expert notes)")
        
        # PART BASS
        if bass_chart and bass_chart.notes:
            bass_track = self._create_guitar_track(bass_chart, "PART BASS", ticks_per_beat)
            mid.tracks.append(bass_track)
            logger.info(f"  Added PART BASS ({len(bass_chart.notes)} expert notes)")
        
        # PART VOCALS
        if lead_phrases:
            vocals_track = self._create_vocals_track(lead_phrases, "PART VOCALS", tempo_bpm, ticks_per_beat)
            mid.tracks.append(vocals_track)
            logger.info(f"  Added PART VOCALS ({len(lead_phrases)} phrases)")
            
            # HARM1 (harmonies)
            if harmony_phrases:
                harm_track = self._create_vocals_track(harmony_phrases, "HARM1", tempo_bpm, ticks_per_beat)
                mid.tracks.append(harm_track)
                logger.info(f"  Added HARM1 ({len(harmony_phrases)} phrases)")
        
        # PART KEYS (5-lane)
        if keys_notes:
            keys_track = self._create_keys_track(keys_notes, "PART KEYS", tempo_bpm, ticks_per_beat)
            mid.tracks.append(keys_track)
            logger.info(f"  Added PART KEYS ({len(keys_notes)} notes)")
            
            # Pro Keys tracks
            for diff, suffix in [("E", "Easy"), ("M", "Medium"), ("H", "Hard"), ("X", "Expert")]:
                prokeys_track = self._create_prokeys_track(
                    keys_notes, f"PART REAL_KEYS_{diff}", tempo_bpm, ticks_per_beat, suffix.lower()
                )
                mid.tracks.append(prokeys_track)
        
        mid.save(output_path)
    
    def _create_drums_track(self, chart, ticks_per_beat: int) -> mido.MidiTrack:
        """Create drums track with all difficulties."""
        from src.export.midi import (
            DIFFICULTY_NOTE_OFFSETS,
            _compute_hit_ticks,
            _get_midi_note,
            _get_tom_marker,
            _reduce_to_hard,
            _reduce_to_medium,
            _reduce_to_easy,
        )
        
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name='PART DRUMS', time=0))
        
        hard_chart = _reduce_to_hard(chart)
        medium_chart = _reduce_to_medium(hard_chart)
        easy_chart = _reduce_to_easy(medium_chart)
        
        events = []
        for difficulty, diff_chart in [("expert", chart), ("hard", hard_chart), 
                                        ("medium", medium_chart), ("easy", easy_chart)]:
            note_offset = DIFFICULTY_NOTE_OFFSETS[difficulty]
            hits_with_ticks = _compute_hit_ticks(diff_chart, ticks_per_beat)
            
            for hit in hits_with_ticks:
                base_note = _get_midi_note(hit) + note_offset
                note_duration = ticks_per_beat // 8
                
                events.append(("on", hit.tick, base_note, hit.velocity))
                events.append(("off", hit.tick + note_duration, base_note, 0))
                
                tom_marker = _get_tom_marker(hit)
                if tom_marker is not None:
                    events.append(("on", hit.tick, tom_marker + note_offset, hit.velocity))
                    events.append(("off", hit.tick + note_duration, tom_marker + note_offset, 0))
        
        events.sort(key=lambda e: (e[1], e[0] == "off"))
        
        prev_tick = 0
        for etype, tick, note, vel in events:
            delta = tick - prev_tick
            track.append(mido.Message('note_on' if etype == 'on' else 'note_off', 
                                      note=note, velocity=vel, time=delta))
            prev_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def _create_guitar_track(self, chart, name: str, ticks_per_beat: int) -> mido.MidiTrack:
        """Create guitar/bass track with all difficulties."""
        from src.inference.guitar_bass import reduce_to_difficulty, FRET_NOTE_OFFSETS
        
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        
        tempo_bpm = chart.tempo_bpm
        ms_per_tick = (60_000 / tempo_bpm) / ticks_per_beat
        
        events = []
        for difficulty in ["expert", "hard", "medium", "easy"]:
            diff_chart = reduce_to_difficulty(chart, difficulty)
            note_offset = FRET_NOTE_OFFSETS[difficulty]
            
            for note in diff_chart.notes:
                start_tick = int(note.time_ms / ms_per_tick)
                duration_ticks = max(int(note.duration_ms / ms_per_tick), ticks_per_beat // 8)
                midi_note = note_offset + note.fret
                
                events.append(("on", start_tick, midi_note, note.velocity))
                events.append(("off", start_tick + duration_ticks, midi_note, 0))
        
        events.sort(key=lambda e: (e[1], e[0] == "off"))
        
        prev_tick = 0
        for etype, tick, note, vel in events:
            delta = tick - prev_tick
            track.append(mido.Message('note_on' if etype == 'on' else 'note_off',
                                      note=note, velocity=vel, time=delta))
            prev_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def _create_vocals_track(self, phrases, name: str, tempo_bpm: float, ticks_per_beat: int) -> mido.MidiTrack:
        """Create vocals track with phrase markers and pitch notes."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        
        ticks_per_sec = ticks_per_beat * tempo_bpm / 60
        events = []
        
        for phrase in phrases:
            # Phrase marker (note 105)
            start_tick = int(phrase.start_time * ticks_per_sec)
            end_tick = int(phrase.end_time * ticks_per_sec)
            
            events.append(("on", start_tick, 105, 100))
            events.append(("off", end_tick, 105, 0))
            
            # Individual notes with pitch
            for note in phrase.notes:
                note_start = int(note.start_time * ticks_per_sec)
                note_end = int(note.end_time * ticks_per_sec)
                midi_pitch = note.midi_pitch  # VocalNote uses midi_pitch attribute
                
                events.append(("on", note_start, midi_pitch, 100))
                events.append(("off", note_end, midi_pitch, 0))
                
                # Add lyric text
                if note.lyric:
                    events.append(("lyric", note_start, note.lyric, 0))
        
        events.sort(key=lambda e: (e[1], 0 if e[0] == "lyric" else (1 if e[0] == "on" else 2)))
        
        prev_tick = 0
        for event in events:
            etype, tick, data, vel = event
            delta = tick - prev_tick
            
            if etype == "lyric":
                track.append(mido.MetaMessage('lyrics', text=data, time=delta))
            elif etype == "on":
                track.append(mido.Message('note_on', note=data, velocity=vel, time=delta))
            else:
                track.append(mido.Message('note_off', note=data, velocity=0, time=delta))
            
            prev_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def _create_keys_track(self, notes, name: str, tempo_bpm: float, ticks_per_beat: int) -> mido.MidiTrack:
        """Create 5-lane keys track."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        
        ticks_per_sec = ticks_per_beat * tempo_bpm / 60

        # Map pitches to 5 lanes
        pitches = [n.midi_pitch for n in notes]
        if pitches:
            min_pitch = min(pitches)
            max_pitch = max(pitches)
            pitch_range = max(max_pitch - min_pitch, 1)
        
        events = []
        for difficulty, offset in [("expert", 96), ("hard", 84), ("medium", 72), ("easy", 60)]:
            for note in notes:
                # Map pitch to 0-4 lanes
                lane = int((note.midi_pitch - min_pitch) / pitch_range * 4.99)
                lane = max(0, min(4, lane))
                
                # Reduce notes for lower difficulties
                if difficulty == "hard" and hash(str(note.start_time)) % 10 < 2:
                    continue
                if difficulty == "medium" and hash(str(note.start_time)) % 10 < 4:
                    continue
                if difficulty == "easy" and hash(str(note.start_time)) % 10 < 6:
                    continue
                
                start_tick = int(note.start_time * ticks_per_sec)
                end_tick = int(note.end_time * ticks_per_sec)
                midi_note = offset + lane
                
                events.append(("on", start_tick, midi_note, 100))
                events.append(("off", end_tick, midi_note, 0))
        
        events.sort(key=lambda e: (e[1], e[0] == "off"))
        
        prev_tick = 0
        for etype, tick, note, vel in events:
            delta = tick - prev_tick
            track.append(mido.Message('note_on' if etype == 'on' else 'note_off',
                                      note=note, velocity=vel, time=delta))
            prev_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def _create_prokeys_track(self, notes, name: str, tempo_bpm: float, ticks_per_beat: int, difficulty: str) -> mido.MidiTrack:
        """Create Pro Keys track with actual pitches and range shifts."""
        track = mido.MidiTrack()
        track.append(mido.MetaMessage('track_name', name=name, time=0))
        
        ticks_per_sec = ticks_per_beat * tempo_bpm / 60
        
        # Reduce notes for lower difficulties
        keep_ratio = {"expert": 1.0, "hard": 0.75, "medium": 0.5, "easy": 0.35}[difficulty]
        
        # Filter notes
        filtered_notes = []
        for i, note in enumerate(notes):
            if hash(str(note.start_time)) % 100 < keep_ratio * 100:
                filtered_notes.append(note)
        
        if not filtered_notes:
            track.append(mido.MetaMessage('end_of_track', time=0))
            return track
        
        events = []
        
        # Add range shift markers
        prev_range_pos = -1
        for note in filtered_notes:
            # Calculate range position (0-5) based on pitch
            # Pro Keys visible range is 17 keys, we shift to keep notes visible
            range_pos = max(0, min(5, (note.midi_pitch - 48) // 5))
            
            if range_pos != prev_range_pos:
                start_tick = int(note.start_time * ticks_per_sec)
                events.append(("range", start_tick, range_pos))
                prev_range_pos = range_pos
        
        # Add notes
        for note in filtered_notes:
            start_tick = int(note.start_time * ticks_per_sec)
            end_tick = int(note.end_time * ticks_per_sec)
            
            # Clamp to Pro Keys range (48-72 = C3-C5)
            midi_pitch = max(48, min(72, note.midi_pitch))
        events.sort(key=lambda e: (e[1], 0 if e[0] == "range" else (1 if e[0] == "on" else 2)))
        
        prev_tick = 0
        for event in events:
            etype = event[0]
            tick = event[1]
            delta = tick - prev_tick
            
            if etype == "range":
                # Note 9 = range shift, velocity = position (0-5)
                track.append(mido.Message('note_on', note=9, velocity=event[2], time=delta))
                track.append(mido.Message('note_off', note=9, velocity=0, time=0))
            elif etype == "on":
                track.append(mido.Message('note_on', note=event[2], velocity=event[3], time=delta))
            else:
                track.append(mido.Message('note_off', note=event[2], velocity=0, time=delta))
            
            prev_tick = tick
        
        track.append(mido.MetaMessage('end_of_track', time=0))
        return track
    
    def run_chart_enhancer(self, midi_path: Path, audio_path: Path):
        """Run the chart enhancer for SP, difficulty reduction, and sections."""
        logger.info("  Enhancing chart (SP, difficulty, sections)...")
        
        from chart_enhancer import ChartEnhancer
        enhancer = ChartEnhancer()
        enhancer.enhance_chart(str(midi_path), str(audio_path), str(midi_path))
    
    def convert_to_ogg(self, input_path: Path, output_path: Path):
        """Convert audio to OGG format."""
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vn",  # Strip any video/image streams (album art)
            "-c:a", "libvorbis", "-q:a", "6",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Try with powershell/system ffmpeg
            logger.warning("ffmpeg failed, audio will be copied instead")
            shutil.copy(input_path, output_path.with_suffix(input_path.suffix))
    
    def create_song_ini(
        self,
        song_folder: Path,
        artist: str,
        title: str,
        tempo_bpm: float,
        duration_ms: int,
        preview_start_ms: int,
        album: str = '',
        year: str = '',
        genre: str = '',
    ):
        """Create song.ini file."""
        ini_content = f"""[song]
name = {title}
artist = {artist}
charter = STRUM
album = {album}
year = {year}
genre = {genre}
diff_drums = -1
diff_guitar = -1
diff_bass = -1
diff_vocals = -1
diff_keys = -1
diff_keys_real = -1
preview_start_time = {preview_start_ms}
song_length = {duration_ms}
"""
        
        ini_path = song_folder / "song.ini"
        ini_path.write_text(ini_content, encoding='utf-8')
    
    def fetch_album_art(self, audio_path: Path, song_folder: Path, artist: str, title: str) -> bool:
        """
        Fetch album art from multiple sources and save to song folder.
        
        Priority:
        1. Embedded ID3 tag in audio file
        2. iTunes Search API (free, no auth needed)
        3. Music file in same folder named album.png/jpg
        
        Returns True if album art was found and saved.
        """
        album_path = song_folder / "album.png"
        
        # 1. Try extracting from audio file embedded art
        if self._extract_embedded_art(audio_path, album_path):
            logger.info("  Album art: extracted from audio file")
            return True
        
        # 2. Try iTunes Search API
        if self._fetch_itunes_art(artist, title, album_path):
            logger.info("  Album art: fetched from iTunes")
            return True
        
        # 3. Check for existing album art in source folder
        source_folder = audio_path.parent
        for art_name in ['album.png', 'album.jpg', 'cover.png', 'cover.jpg', 'folder.jpg', 'folder.png']:
            art_source = source_folder / art_name
            if art_source.exists():
                shutil.copy(art_source, album_path.with_suffix(art_source.suffix))
                logger.info(f"  Album art: copied from source folder ({art_name})")
                return True
        
        logger.warning("  ⚠ No album art found")
        return False
    
    def _extract_embedded_art(self, audio_path: Path, output_path: Path) -> bool:
        """Extract album art embedded in audio file."""
        try:
            from mutagen import File
            from mutagen.id3 import ID3
            from mutagen.mp3 import MP3
            from mutagen.flac import FLAC
            from mutagen.mp4 import MP4
            
            audio = File(str(audio_path))
            if audio is None:
                return False
            
            art_data = None
            
            # MP3 with ID3 tags
            if str(audio_path).lower().endswith('.mp3'):
                try:
                    tags = ID3(str(audio_path))
                    for key in tags.keys():
                        if key.startswith('APIC'):
                            art_data = tags[key].data
                            break
                except Exception:
                    pass
            
            # FLAC
            elif str(audio_path).lower().endswith('.flac'):
                try:
                    flac = FLAC(str(audio_path))
                    if flac.pictures:
                        art_data = flac.pictures[0].data
                except Exception:
                    pass
            
            # M4A/MP4
            elif str(audio_path).lower().endswith(('.m4a', '.mp4')):
                try:
                    mp4 = MP4(str(audio_path))
                    if 'covr' in mp4.tags:
                        art_data = bytes(mp4.tags['covr'][0])
                except Exception:
                    pass
            
            if art_data:
                # Determine format from magic bytes
                if art_data[:8] == b'\x89PNG\r\n\x1a\n':
                    output_path = output_path.with_suffix('.png')
                else:
                    output_path = output_path.with_suffix('.jpg')
                
                output_path.write_bytes(art_data)
                return True
            
            return False
            
        except ImportError:
            logger.debug("mutagen not installed, skipping embedded art extraction")
            return False
        except Exception as e:
            logger.debug(f"Failed to extract embedded art: {e}")
            return False
    
    def _fetch_itunes_art(self, artist: str, title: str, output_path: Path) -> bool:
        """Fetch album art from iTunes Search API."""
        try:
            import urllib.request
            import urllib.parse
            import json as json_lib
            
            # Build search query
            query = f"{artist} {title}"
            encoded_query = urllib.parse.quote(query)
            url = f"https://itunes.apple.com/search?term={encoded_query}&media=music&entity=song&limit=5"
            
            # Make request
            req = urllib.request.Request(url, headers={'User-Agent': 'STRUM/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json_lib.loads(response.read().decode('utf-8'))
            
            if data['resultCount'] == 0:
                return False
            
            # Find best match (look for artist/title match)
            art_url = None
            for result in data['results']:
                result_artist = result.get('artistName', '').lower()
                result_track = result.get('trackName', '').lower()
                
                if artist.lower() in result_artist or result_artist in artist.lower():
                    if title.lower() in result_track or result_track in title.lower():
                        art_url = result.get('artworkUrl100', '')
                        break
            
            # Fallback to first result
            if not art_url and data['results']:
                art_url = data['results'][0].get('artworkUrl100', '')
            
            if not art_url:
                return False
            
            # Get higher resolution (600x600)
            art_url = art_url.replace('100x100', '600x600')
            
            # Download image
            req = urllib.request.Request(art_url, headers={'User-Agent': 'STRUM/1.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                image_data = response.read()
            
            # Save as jpg (iTunes returns jpg)
            output_path = output_path.with_suffix('.jpg')
            output_path.write_bytes(image_data)
            return True
            
        except Exception as e:
            logger.debug(f"Failed to fetch from iTunes: {e}")
            return False
    
    def fetch_music_video(self, song_folder: Path, artist: str, title: str) -> bool:
        """
        Optionally fetch music video from YouTube (if yt-dlp is available).
        
        Returns True if video was found and downloaded.
        """
        video_path = song_folder / "video.mp4"
        
        try:
            # Check if yt-dlp is available
            result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True)
            if result.returncode != 0:
                return False
            
            # Search and download video
            query = f"ytsearch1:{artist} - {title} official music video"
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
                "-o", str(video_path),
                "--no-playlist",
                "--max-filesize", "200M",
                query
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if video_path.exists():
                logger.info(f"  Video: downloaded music video")
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Video download failed: {e}")
            return False

    def process_song(self, audio_path: Path) -> SongResult:
        """Process a single song through the full pipeline."""
        artist, title = self.parse_filename(audio_path)
        safe_name = re.sub(r'[<>:"/\\|?*]', '', f"{artist} - {title}")
        song_folder = self.output_dir / safe_name
        
        result = SongResult(
            input_path=str(audio_path),
            output_path=str(song_folder),
            success=False
        )
        
        stems = {}
        tempo_bpm = 120  # Default fallback
        audio_info = None
        
        try:
            song_folder.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Processing: {artist} - {title}")
            
            # Analyze audio
            audio_info = self.analyze_audio(audio_path, artist, title)
            tempo_bpm = audio_info['tempo_bpm']
            logger.info(f"  Tempo: {tempo_bpm} BPM, Duration: {audio_info['duration_sec']:.1f}s")
            
            # Separate stems
            stems = self.separate_stems(audio_path, song_folder)
            
        except Exception as e:
            result.error = f"Preprocessing failed: {e}"
            logger.error(f"  ✗ Preprocessing failed: {e}")
            return result
        
        # Transcribe each instrument - isolated errors per instrument
        drums_chart = None
        guitar_chart = None
        bass_chart = None
        lead_phrases = None
        harmony_phrases = None
        keys_notes = None
        errors = []
        
        # Drums
        if "drums" in stems:
            try:
                drums_chart = self.transcribe_drums(stems["drums"], tempo_bpm)
                if drums_chart:
                    result.drums_hits = len(drums_chart.hits)
                    logger.info(f"    Drums: {result.drums_hits} hits")
            except Exception as e:
                errors.append(f"drums: {e}")
                logger.warning(f"  ⚠ Drums failed: {e}")
        
        # Guitar - prefer dedicated guitar stem (htdemucs_6s), fallback to other
        guitar_stem = stems.get("guitar") or stems.get("other")
        if guitar_stem:
            try:
                stem_type = "guitar" if "guitar" in stems else "other"
                logger.info(f"    Using {stem_type} stem for guitar transcription")
                guitar_chart = self.transcribe_guitar(guitar_stem, tempo_bpm, full_mix=audio_path)
                if guitar_chart:
                    result.guitar_notes = len(guitar_chart.notes)
                    logger.info(f"    Guitar: {result.guitar_notes} notes")
            except Exception as e:
                errors.append(f"guitar: {e}")
                logger.warning(f"  ⚠ Guitar failed: {e}")
        
        # Keys - prefer dedicated piano stem (htdemucs_6s), fallback to other
        keys_stem = stems.get("piano") or stems.get("other")
        if keys_stem:
            try:
                stem_type = "piano" if "piano" in stems else "other"
                logger.info(f"    Using {stem_type} stem for keys transcription")
                keys_notes = self.transcribe_keys(keys_stem)
                if keys_notes:
                    result.keys_notes = len(keys_notes)
                    logger.info(f"    Keys: {result.keys_notes} notes")
            except Exception as e:
                errors.append(f"keys: {e}")
                logger.warning(f"  ⚠ Keys failed: {e}")
        
        # Bass
        if "bass" in stems:
            try:
                bass_chart = self.transcribe_bass(stems["bass"], tempo_bpm, full_mix=audio_path)
                if bass_chart:
                    result.bass_notes = len(bass_chart.notes)
                    logger.info(f"    Bass: {result.bass_notes} notes")
            except Exception as e:
                errors.append(f"bass: {e}")
                logger.warning(f"  ⚠ Bass failed: {e}")
        
        # Vocals
        if "vocals" in stems:
            try:
                lead_phrases, harmony_phrases = self.transcribe_vocals(
                    stems["vocals"], artist, title
                )
                if lead_phrases:
                    result.vocals_phrases = len(lead_phrases)
                    logger.info(f"    Vocals: {result.vocals_phrases} phrases")
            except Exception as e:
                errors.append(f"vocals: {e}")
                logger.warning(f"  ⚠ Vocals failed: {e}")
        
        # Check if we have ANY transcribed content
        has_content = any([
            drums_chart and drums_chart.hits,
            guitar_chart and guitar_chart.notes,
            bass_chart and bass_chart.notes,
            lead_phrases,
            keys_notes,
        ])
        
        if not has_content:
            result.error = "No instruments transcribed successfully"
            logger.error(f"  ✗ No instruments transcribed")
            # Still cleanup stems
            self._cleanup_stems(song_folder)
            return result
        
        # Create combined MIDI (even if some instruments failed)
        try:
            logger.info("  Creating combined chart...")
            notes_path = song_folder / "notes.mid"
            self.create_combined_midi(
                notes_path,
                tempo_bpm,
                drums_chart=drums_chart,
                guitar_chart=guitar_chart,
                bass_chart=bass_chart,
                lead_phrases=lead_phrases,
                harmony_phrases=harmony_phrases,
                keys_notes=keys_notes,
            )
            
            # Convert audio
            song_ogg = song_folder / "song.ogg"
            self.convert_to_ogg(audio_path, song_ogg)
            
            # Create song.ini
            self.create_song_ini(
                song_folder,
                artist=artist,
                title=title,
                tempo_bpm=tempo_bpm,
                duration_ms=audio_info['duration_ms'],
                preview_start_ms=audio_info['preview_start_ms'],
                album=audio_info.get('album', ''),
                year=audio_info.get('year', ''),
                genre=audio_info.get('genre', ''),
            )
            
            # Run chart enhancer
            try:
                audio_for_enhancer = song_ogg if song_ogg.exists() else audio_path
                self.run_chart_enhancer(notes_path, audio_for_enhancer)
            except Exception as e:
                logger.warning(f"  ⚠ Chart enhancement failed: {e}")
            
            # Fetch album art (required)
            try:
                self.fetch_album_art(audio_path, song_folder, artist, title)
            except Exception as e:
                logger.warning(f"  ⚠ Album art fetch failed: {e}")
            
            # Optionally fetch music video
            if self.include_video:
                try:
                    self.fetch_music_video(song_folder, artist, title)
                except Exception as e:
                    logger.debug(f"  Video fetch skipped: {e}")
            
            result.success = True
            if errors:
                result.error = f"Partial: {'; '.join(errors)}"
                logger.info(f"  ✓ Complete (with warnings): {song_folder.name}")
            else:
                logger.info(f"  ✓ Complete: {song_folder.name}")
            
        except Exception as e:
            result.error = f"Chart creation failed: {e}"
            logger.error(f"  ✗ Chart creation failed: {e}")
        
        # Always cleanup stems at the end
        self._cleanup_stems(song_folder)
        
        return result
    
    def _cleanup_stems(self, song_folder: Path):
        """Remove temporary stem files."""
        demucs_temp = song_folder / "demucs_temp"
        if demucs_temp.exists():
            try:
                shutil.rmtree(demucs_temp)
                logger.info("  Cleaned up stem files")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to cleanup stems: {e}")
    
    def process_batch(self, input_dir: Path, continue_from: bool = False) -> List[SongResult]:
        """Process all audio files in a directory."""
        input_dir = Path(input_dir)
        
        # Find audio files
        extensions = ["*.mp3", "*.wav", "*.ogg", "*.flac", "*.m4a"]
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_dir.glob(ext))
        
        audio_files = sorted(set(audio_files))
        
        if not audio_files:
            logger.error(f"No audio files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(audio_files)} audio files")
        logger.info("=" * 60)
        
        # Check for already processed if continuing
        if continue_from:
            existing = set(p.name for p in self.output_dir.iterdir() if p.is_dir())
            original_count = len(audio_files)
            audio_files = [
                f for f in audio_files
                if re.sub(r'[<>:"/\\|?*]', '', f"{self.parse_filename(f)[0]} - {self.parse_filename(f)[1]}") not in existing
            ]
            if len(audio_files) < original_count:
                logger.info(f"Skipping {original_count - len(audio_files)} already processed songs")
        
        # Process each file
        results = []
        for i, audio_path in enumerate(audio_files, 1):
            logger.info(f"\n[{i}/{len(audio_files)}] {audio_path.name}")
            result = self.process_song(audio_path)
            results.append(result)
        
        # Summary
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        logger.info("\n" + "=" * 60)
        logger.info("BATCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed: {len(results)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        
        if failed:
            logger.info("\nFailed songs:")
            for r in failed:
                logger.info(f"  - {Path(r.input_path).name}: {r.error}")
        
        # Save results
        results_path = self.output_dir / "batch_results.json"
        with open(results_path, 'w') as f:
            json.dump([vars(r) for r in results], f, indent=2)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="STRUM Batch Pipeline - Process multiple songs"
    )
    parser.add_argument("input_dir", help="Input directory with audio files")
    parser.add_argument("output_dir", help="Output directory for chart folders")
    parser.add_argument("--no-drums", action="store_true", help="Skip drums")
    parser.add_argument("--no-guitar", action="store_true", help="Skip guitar")
    parser.add_argument("--no-bass", action="store_true", help="Skip bass")
    parser.add_argument("--no-vocals", action="store_true", help="Skip vocals")
    parser.add_argument("--no-keys", action="store_true", help="(deprecated; keys are off by default)")
    parser.add_argument("--keys", action="store_true",
                        help="Enable keys transcription (off by default; detector is unreliable)")
    parser.add_argument("--include-video", action="store_true",
                        help="Download music videos from YouTube (requires yt-dlp)")
    parser.add_argument("--continue", dest="continue_from", action="store_true",
                        help="Skip already processed songs")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    pipeline = BatchPipeline(
        output_dir=Path(args.output_dir),
        include_drums=not args.no_drums,
        include_guitar=not args.no_guitar,
        include_bass=not args.no_bass,
        include_vocals=not args.no_vocals,
        include_keys=args.keys and not args.no_keys,
        include_video=args.include_video,
        device=args.device,
    )
    
    pipeline.process_batch(Path(args.input_dir), continue_from=args.continue_from)


if __name__ == "__main__":
    main()
