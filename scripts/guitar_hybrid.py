"""
Guitar Charting - Hybrid Pipeline

Two-stage approach:
1. ONSET DETECTION: Neural model predicts when notes occur
2. FRET ASSIGNMENT: Rule-based using pitch + playability patterns

This combines the best of both worlds:
- Neural: Learns complex onset patterns from audio (transients, attacks)
- Rules: Handles subjective fret mapping with musical heuristics
"""

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import onset model
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_guitar_onset import OnsetCRNN, OnsetConfig, load_model


@dataclass  
class ChartNote:
    """A single note or chord in the chart."""
    time_sec: float
    fret: int  # 0-4 (primary fret)
    duration_sec: float = 0.0
    is_hopo: bool = False
    is_sustain: bool = False
    midi_pitch: Optional[int] = None
    chord_frets: Optional[List[int]] = None  # Additional frets for chords


class HybridGuitarCharter:
    """
    Hybrid guitar charting: neural onset detection + rule-based frets.
    """
    
    def __init__(
        self,
        onset_checkpoint: Path,
        device: torch.device = None,
        onset_threshold: float = 0.6,  # Higher threshold = fewer notes
        hopo_threshold_ms: float = 170.0,
        sustain_min_ms: float = 1000.0,  # 1 second minimum for sustains
        tempo_bpm: float = None,  # Auto-detect if None
        quantize_strength: float = 0.8,  # 0=no quantization, 1=hard snap to grid
        quantize_grid: str = '1/16',  # Beat subdivision: '1/4', '1/8', '1/16', '1/32'
        min_start_time_sec: float = 0.0,  # Skip phantom notes before this time (0 = disabled)
    ):
        """
        Args:
            onset_checkpoint: Path to trained onset model
            device: Torch device (auto-detect if None)
            onset_threshold: Threshold for onset detection
            hopo_threshold_ms: Max time between notes for HOPO
            sustain_min_ms: Min duration for sustain notes
            tempo_bpm: Song tempo (auto-detected if None)
            quantize_strength: How strongly to snap to grid (0-1)
            quantize_grid: Beat subdivision for grid ('1/4', '1/8', '1/16', '1/32')
            min_start_time_sec: Skip onset detections before this time (Demucs artifacts)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.onset_threshold = onset_threshold
        self.hopo_threshold_sec = hopo_threshold_ms / 1000.0
        self.sustain_min_sec = sustain_min_ms / 1000.0
        self.tempo_bpm = tempo_bpm
        self.quantize_strength = quantize_strength
        self.quantize_grid = quantize_grid
        self.min_start_time_sec = min_start_time_sec
        
        # Load onset model
        logger.info(f"Loading onset model from {onset_checkpoint}")
        self.onset_model, self.config = load_model(onset_checkpoint, self.device)
        self.onset_model.eval()
        
        # Mel transform
        import torchaudio
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
        )
    
    def chart_from_stems(
        self,
        other_path: Path,
        bass_path: Path,
    ) -> List[ChartNote]:
        """
        Generate chart from separated stems.
        
        Args:
            other_path: Path to other.wav (guitar/keys)
            bass_path: Path to bass.wav
            
        Returns:
            List of ChartNote objects
        """
        # Load audio
        y_other, _ = librosa.load(other_path, sr=self.config.sample_rate, mono=True)
        y_bass, _ = librosa.load(bass_path, sr=self.config.sample_rate, mono=True)
        
        # Align lengths
        min_len = min(len(y_other), len(y_bass))
        y_other = y_other[:min_len]
        y_bass = y_bass[:min_len]
        
        # Stage 0: Detect or use provided tempo
        if self.tempo_bpm is None:
            detected_tempo = self._detect_tempo(y_other + 0.5 * y_bass)
            self.tempo_bpm = detected_tempo
            logger.info(f"Auto-detected tempo: {self.tempo_bpm:.1f} BPM")
        else:
            logger.info(f"Using provided tempo: {self.tempo_bpm:.1f} BPM")
        
        # Stage 1: Detect onsets
        onset_frames = self._detect_onsets(y_other, y_bass)
        onset_times = onset_frames * self.config.hop_length / self.config.sample_rate
        logger.info(f"Detected {len(onset_times)} raw note onsets")
        
        # Stage 1.5: Quantize to beat grid
        if self.quantize_strength > 0:
            onset_times = self._quantize_to_grid(onset_times)
            logger.info(f"Quantized to {self.quantize_grid} grid ({len(onset_times)} notes after dedup)")
        
        # Stage 1.6: Filter phantom notes at audio start (Demucs artifacts)
        if self.min_start_time_sec > 0:
            pre_filter = len(onset_times)
            onset_times = onset_times[onset_times >= self.min_start_time_sec]
            filtered_count = pre_filter - len(onset_times)
            if filtered_count > 0:
                logger.info(f"Filtered {filtered_count} phantom notes before {self.min_start_time_sec:.1f}s")
        
        # Stage 2: Get pitches at onset times
        # Mix stems for better pitch detection (isolated stems have poor pitch tracking)
        y_mixed = y_other + 0.5 * y_bass
        pitches = self._get_pitches_at_onsets(y_mixed, onset_times)
        
        # Log pitch detection quality
        valid_pitches = pitches > 0
        logger.info(f"Valid pitches: {valid_pitches.sum()}/{len(pitches)} ({100*valid_pitches.mean():.1f}%)")
        
        # Stage 3: Assign frets using pitch-based mapping
        frets = self._assign_frets(pitches, onset_times)
        
        # Stage 4: Detect chord candidates based on energy
        chord_mask, onset_energies = self._detect_chords(y_other, onset_times)
        
        # Stage 5: Detect actual note durations from audio energy
        durations = self._detect_note_durations(y_other, onset_times)
        
        # Stage 6: Build chart notes with HOPOs, sustains, and chords
        notes = self._build_notes(onset_times, frets, pitches, chord_mask, durations, onset_energies)
        
        return notes
    
    def _detect_onsets(self, y_other: np.ndarray, y_bass: np.ndarray) -> np.ndarray:
        """Run neural onset detection with adaptive thresholding."""
        # Create spectrograms
        mel_other = self._compute_mel(y_other)
        mel_bass = self._compute_mel(y_bass)
        
        # Align
        min_frames = min(mel_other.shape[-1], mel_bass.shape[-1])
        mel_other = mel_other[:, :min_frames]
        mel_bass = mel_bass[:, :min_frames]
        
        # Stack: (1, 2, n_mels, T)
        spec = torch.stack([mel_other, mel_bass], dim=0).unsqueeze(0)
        
        # Predict in segments
        frames_per_seg = self.config.frames_per_segment
        total_frames = spec.size(-1)
        all_probs = torch.zeros(total_frames)
        counts = torch.zeros(total_frames)
        
        with torch.no_grad():
            for start in range(0, total_frames, frames_per_seg // 2):
                end = min(start + frames_per_seg, total_frames)
                segment = spec[:, :, :, start:end].to(self.device)
                
                # Pad if needed
                if segment.size(-1) < frames_per_seg:
                    pad = frames_per_seg - segment.size(-1)
                    segment = F.pad(segment, (0, pad))
                
                logits = self.onset_model(segment)
                probs = torch.sigmoid(logits).cpu()
                
                seg_len = min(end - start, probs.size(-1))
                all_probs[start:start+seg_len] += probs[0, :seg_len]
                counts[start:start+seg_len] += 1
        
        # Average overlapping predictions
        all_probs /= counts.clamp(min=1)
        
        # Compute adaptive threshold based on local audio energy
        adaptive_threshold = self._compute_adaptive_threshold(y_other, total_frames)
        
        # Peak picking with adaptive threshold
        onset_frames = self._peak_pick_adaptive(all_probs.numpy(), adaptive_threshold)
        
        return onset_frames
    
    def _compute_adaptive_threshold(self, y: np.ndarray, n_frames: int) -> np.ndarray:
        """
        Compute frame-wise adaptive threshold based on local audio energy.
        
        During high-energy sections, we lower the threshold to catch more notes.
        During quiet sections, we raise it to avoid false positives.
        
        Uses look-ahead to catch notes at the START of loud sections,
        and look-behind to catch notes at the END of loud sections.
        """
        hop = self.config.hop_length
        
        # Compute frame-level RMS energy
        frame_energy = np.zeros(n_frames)
        for i in range(n_frames):
            start = i * hop
            end = start + hop * 2  # 2-frame window
            if end > len(y):
                end = len(y)
            if start < len(y):
                frame_energy[i] = np.sqrt(np.mean(y[start:end] ** 2) + 1e-10)
        
        from scipy.ndimage import uniform_filter1d, maximum_filter1d
        
        # Smooth energy with smaller window for faster response (~0.5 sec)
        smooth_energy = uniform_filter1d(frame_energy, size=22)
        
        # LOOK-AHEAD: Use maximum filter shifted forward
        # This catches notes BEFORE a loud section starts
        look_ahead_frames = 15  # ~350ms look-ahead
        look_ahead_energy = np.roll(smooth_energy, -look_ahead_frames)
        look_ahead_energy[-look_ahead_frames:] = smooth_energy[-look_ahead_frames:]
        
        # LOOK-BEHIND: Use maximum filter to extend high energy period
        # This catches notes AFTER a loud section ends
        look_behind_frames = 20  # ~460ms sustain
        sustained_energy = maximum_filter1d(smooth_energy, size=look_behind_frames)
        
        # Combine: take maximum of look-ahead and sustained energy
        combined_energy = np.maximum(look_ahead_energy, sustained_energy)
        
        # Normalize energy to 0-1 range
        e_min, e_max = combined_energy.min(), combined_energy.max()
        if e_max > e_min:
            norm_energy = (combined_energy - e_min) / (e_max - e_min)
        else:
            norm_energy = np.ones_like(combined_energy) * 0.5
        
        # Adaptive threshold: lower during high energy, higher during quiet
        # Base threshold varies from base*0.6 (loud) to base*1.2 (quiet)
        base_threshold = self.onset_threshold
        threshold_range = 0.30  # ±30% of base threshold (wider range)
        adaptive = base_threshold * (1.0 + threshold_range - 2 * threshold_range * norm_energy)
        
        # Clamp to reasonable range
        adaptive = np.clip(adaptive, base_threshold * 0.5, base_threshold * 1.3)
        
        return adaptive
    
    def _peak_pick_adaptive(self, probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """
        Peak picking with frame-wise adaptive thresholds.
        """
        from scipy.ndimage import maximum_filter1d
        
        # Use maximum filter for efficient local maxima detection
        window_size = 7  # 3 before + 3 after + 1
        local_max = maximum_filter1d(probs, size=window_size, mode='constant')
        
        # Peaks are where value equals local max AND above adaptive threshold
        is_peak = np.isclose(probs, local_max, atol=1e-6) & (probs > thresholds)
        peaks = np.where(is_peak)[0]
        
        if len(peaks) == 0:
            return np.array([])
        
        # Enforce minimum distance between peaks (2 frames = ~46ms)
        selected = [peaks[0]]
        for p in peaks[1:]:
            if p - selected[-1] >= 2:
                selected.append(p)
        
        return np.array(selected)
    
    def _compute_mel(self, y: np.ndarray) -> torch.Tensor:
        """Compute log-mel spectrogram."""
        y_tensor = torch.from_numpy(y).float().unsqueeze(0)
        mel = self.mel_transform(y_tensor)
        mel = torch.log(mel + 1e-8)
        return mel.squeeze(0)
    
    def _detect_tempo(self, y: np.ndarray) -> float:
        """
        Detect tempo from audio using librosa.
        
        Returns estimated tempo in BPM, clamped to reasonable range for rock/metal.
        """
        # Use librosa's beat tracker which works well for rock music
        tempo, _ = librosa.beat.beat_track(y=y, sr=self.config.sample_rate)
        
        # Handle array return (newer librosa versions)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        
        # Clamp to reasonable range (60-200 BPM for most music)
        # If detected tempo is very low, it might be half-time - double it
        if tempo < 60:
            tempo *= 2
        # If very high, might be double-time - halve it
        elif tempo > 200:
            tempo /= 2
        
        return float(np.clip(tempo, 60, 200))
    
    def _quantize_to_grid(self, onset_times: np.ndarray) -> np.ndarray:
        """
        Quantize onset times to a beat grid.
        
        This ensures notes fall on musically meaningful positions (beats, half-beats, etc)
        which makes patterns feel correct when playing.
        
        Uses "soft" quantization - notes are pulled toward grid points based on
        quantize_strength, not hard-snapped. This preserves intentional "feel"
        while correcting timing errors.
        """
        if len(onset_times) == 0:
            return onset_times
        
        # Calculate grid interval based on subdivision
        beat_duration = 60.0 / self.tempo_bpm  # Duration of one beat in seconds
        grid_map = {
            '1/4': 1.0,    # Quarter note = 1 beat
            '1/8': 0.5,    # Eighth note = half beat
            '1/16': 0.25,  # Sixteenth = quarter beat
            '1/32': 0.125, # Thirty-second = eighth beat
        }
        grid_factor = grid_map.get(self.quantize_grid, 0.25)
        grid_interval = beat_duration * grid_factor
        
        logger.info(f"Beat duration: {beat_duration:.3f}s, grid interval: {grid_interval:.3f}s")
        
        # Quantize each onset time
        quantized = []
        for t in onset_times:
            # Find nearest grid point
            grid_position = round(t / grid_interval)
            nearest_grid = grid_position * grid_interval
            
            # Soft quantization: blend between original and grid based on strength
            # Higher strength = closer to grid
            quantized_time = t + self.quantize_strength * (nearest_grid - t)
            quantized.append(quantized_time)
        
        quantized = np.array(quantized)
        
        # Remove duplicates (notes that quantized to same grid point)
        # Keep the first note at each grid position
        min_gap = grid_interval * 0.5  # Notes must be at least half a grid apart
        deduped = [quantized[0]]
        for t in quantized[1:]:
            if t - deduped[-1] >= min_gap:
                deduped.append(t)
        
        return np.array(deduped)
    
    def _peak_pick(self, probs: np.ndarray, pre_max: int = 3, post_max: int = 3) -> np.ndarray:
        """
        Peak picking for onset detection.
        Find local maxima that are above threshold.
        """
        from scipy.ndimage import maximum_filter1d
        
        # Use maximum filter for efficient local maxima detection
        window_size = pre_max + post_max + 1
        local_max = maximum_filter1d(probs, size=window_size, mode='constant')
        
        # Peaks are where value equals local max AND above threshold
        # Use small tolerance for float comparison
        is_peak = np.isclose(probs, local_max, atol=1e-6) & (probs > self.onset_threshold)
        peaks = np.where(is_peak)[0]
        
        if len(peaks) == 0:
            return np.array([])
        
        # Enforce minimum distance between peaks (2 frames = ~46ms)
        selected = [peaks[0]]
        for p in peaks[1:]:
            if p - selected[-1] >= 2:
                selected.append(p)
        
        return np.array(selected)
    
    def _get_pitches_at_onsets(self, y: np.ndarray, onset_times: np.ndarray) -> np.ndarray:
        """Get pitch at each onset using pYIN."""
        # Extract melody with pYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=self.config.sample_rate,
            hop_length=self.config.hop_length,
        )
        
        times = librosa.frames_to_time(
            np.arange(len(f0)), 
            sr=self.config.sample_rate, 
            hop_length=self.config.hop_length
        )
        
        # Convert F0 to MIDI
        pitches = np.zeros_like(f0)
        valid = ~np.isnan(f0) & (f0 > 0)
        pitches[valid] = librosa.hz_to_midi(f0[valid])
        
        # Get pitch at each onset (nearest frame)
        onset_pitches = []
        for t in onset_times:
            idx = np.argmin(np.abs(times - t))
            onset_pitches.append(pitches[idx] if valid[idx] else 0)
        
        return np.array(onset_pitches)
    
    def _assign_frets(self, pitches: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Pitch-based fret assignment - maps actual pitches to frets.
        
        When valid pitches are detected, maps them directly to frets based on
        their position in the pitch range. Falls back to simple patterns only
        when pitch detection fails.
        """
        if len(pitches) == 0:
            return np.array([], dtype=int)
        
        frets = np.zeros(len(pitches), dtype=int)
        valid_mask = pitches > 0
        valid_ratio = valid_mask.mean()
        
        if valid_ratio >= 0.3:
            # Good pitch data - use direct pitch-to-fret mapping
            valid_pitches = pitches[valid_mask]
            
            # Get pitch range (use percentiles to ignore outliers)
            p_min = np.percentile(valid_pitches, 5)
            p_max = np.percentile(valid_pitches, 95)
            p_range = max(p_max - p_min, 1)  # Avoid division by zero
            
            logger.info(f"Pitch range: {p_min:.1f} - {p_max:.1f} MIDI ({p_range:.1f} semitones)")
            
            # Map each pitch to fret 0-4
            for i in range(len(pitches)):
                if pitches[i] > 0:
                    # Normalize pitch to 0-1 range, then scale to 0-4
                    normalized = (pitches[i] - p_min) / p_range
                    normalized = np.clip(normalized, 0.0, 1.0)
                    frets[i] = int(round(normalized * 4))
                else:
                    # Invalid pitch - use previous fret or middle
                    frets[i] = frets[i-1] if i > 0 else 2
        else:
            # Poor pitch data - use simple cycling (not alternating)
            logger.warning(f"Low pitch validity ({valid_ratio*100:.1f}%), using fallback")
            for i in range(len(pitches)):
                # Cycle through frets based on time gaps
                if i == 0:
                    frets[i] = 0
                else:
                    gap = times[i] - times[i-1]
                    if gap > 0.3:  # New phrase - reset
                        frets[i] = 0
                    else:
                        # Move up one fret, wrap at 4
                        frets[i] = (frets[i-1] + 1) % 5
        
        return frets
    
    def _detect_chords(self, y: np.ndarray, onset_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect chord candidates based on spectral complexity.
        
        Returns:
            chord_mask: Boolean mask, True = chord, False = single note.
            onset_energies: Normalized energy at each onset (for 3-note chord decisions).
        Uses spectral flatness to detect if multiple frequencies are present.
        """
        if len(onset_times) == 0:
            return np.array([], dtype=bool), np.array([])
        
        # Compute spectral flatness over time
        hop_length = self.config.hop_length
        sr = self.config.sample_rate
        
        # Use STFT for spectral analysis
        S = np.abs(librosa.stft(y, hop_length=hop_length))
        
        # Compute spectral flatness per frame
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        times_flat = librosa.frames_to_time(np.arange(len(flatness)), sr=sr, hop_length=hop_length)
        
        # Also compute RMS energy
        rms = librosa.feature.rms(S=S)[0]
        rms_norm = rms / (rms.max() + 1e-8)
        
        # Get flatness and energy at each onset
        chord_mask = np.zeros(len(onset_times), dtype=bool)
        
        # Compute thresholds (top 25% of energy is chord candidate)
        onset_energies = []
        for t in onset_times:
            idx = np.argmin(np.abs(times_flat - t))
            onset_energies.append(rms_norm[idx] if idx < len(rms_norm) else 0)
        onset_energies = np.array(onset_energies)
        
        energy_threshold = np.percentile(onset_energies, 75)
        
        # Mark high-energy notes as chords (but not too many)
        n_chords = 0
        max_chords = len(onset_times) // 4  # Max 25% chords
        
        for i, t in enumerate(onset_times):
            if onset_energies[i] >= energy_threshold and n_chords < max_chords:
                chord_mask[i] = True
                n_chords += 1
        
        logger.info(f"Detected {n_chords} chord candidates ({100*n_chords/len(onset_times):.1f}%)")
        return chord_mask, onset_energies
    
    def _detect_note_durations(self, y: np.ndarray, onset_times: np.ndarray) -> np.ndarray:
        """
        Detect actual note durations by tracking energy decay after each onset.
        
        Returns duration in seconds for each note based on when energy drops
        below a threshold of the onset energy.
        """
        if len(onset_times) == 0:
            return np.array([])
        
        sr = self.config.sample_rate
        hop_length = self.config.hop_length
        
        # Compute RMS energy envelope
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        durations = np.zeros(len(onset_times))
        decay_threshold = 0.25  # Note ends when energy drops to 25% of onset energy
        min_duration = 0.1  # Minimum 100ms
        max_duration = 4.0  # Maximum 4 seconds
        
        # Pre-compute note density for each onset (notes within 1 second window)
        note_density = np.zeros(len(onset_times))
        for i, t in enumerate(onset_times):
            nearby = np.sum((onset_times >= t - 0.5) & (onset_times <= t + 0.5))
            note_density[i] = nearby
        
        for i, onset_t in enumerate(onset_times):
            # Find onset frame
            onset_idx = np.argmin(np.abs(times_rms - onset_t))
            onset_energy = rms[onset_idx] if onset_idx < len(rms) else 0
            
            if onset_energy < 1e-6:
                # No energy at onset - use minimum duration
                durations[i] = min_duration
                continue
            
            # Find max energy in small window after onset (attack peak)
            peak_window = min(10, len(rms) - onset_idx)  # ~100ms window
            if peak_window > 0:
                peak_idx = onset_idx + np.argmax(rms[onset_idx:onset_idx + peak_window])
                peak_energy = rms[peak_idx]
            else:
                peak_energy = onset_energy
            
            # Threshold is 25% of peak energy
            threshold_energy = peak_energy * decay_threshold
            
            # KEY FIX: If there's another note nearby, cap the duration
            # This prevents sustains in note clusters
            if i < len(onset_times) - 1:
                gap_to_next = onset_times[i + 1] - onset_t
                
                # In dense sections (3+ notes in 1 sec window), don't sustain
                if note_density[i] >= 3:
                    # Cap at 80% of gap to next note (no sustain in clusters)
                    max_search_time = onset_t + min(gap_to_next * 0.8, 0.3)
                elif gap_to_next < 0.5:
                    # Next note is within 500ms - cap duration at 80% of gap
                    max_search_time = onset_t + gap_to_next * 0.8
                else:
                    # Isolated note - can sustain up to next note or max
                    max_search_time = min(onset_times[i + 1] - 0.05, onset_t + max_duration)
            else:
                max_search_time = onset_t + max_duration
            
            max_search_idx = np.argmin(np.abs(times_rms - max_search_time))
            
            # Search for decay point
            duration = min_duration
            for j in range(peak_idx, min(max_search_idx, len(rms))):
                if rms[j] < threshold_energy:
                    duration = times_rms[j] - onset_t
                    break
            else:
                # Didn't find decay - use time to max search time
                duration = max_search_time - onset_t
            
            durations[i] = np.clip(duration, min_duration, max_duration)
        
        # Additional filter: only allow sustain if note is truly isolated
        # (next note is >1.5 seconds away)
        for i in range(len(durations)):
            if i < len(onset_times) - 1:
                gap_to_next = onset_times[i + 1] - onset_times[i]
                # Only sustain if gap is large AND duration would be long
                if gap_to_next < 1.5 and durations[i] >= self.sustain_min_sec:
                    # Don't sustain - cap just below sustain threshold
                    durations[i] = min(durations[i], self.sustain_min_sec * 0.8)
        
        # Log stats
        n_sustains = (durations >= self.sustain_min_sec).sum()
        logger.info(f"Note durations: min={durations.min():.2f}s, max={durations.max():.2f}s, "
                   f"mean={durations.mean():.2f}s, sustains={n_sustains} ({100*n_sustains/len(durations):.1f}%)")
        
        return durations
    
    def _classify_phrase_pattern(self, pitches: np.ndarray, times: np.ndarray) -> str:
        """
        Classify phrase into pattern type based on pitch contour.
        
        Returns one of:
        - 'ascending': generally moving up
        - 'descending': generally moving down  
        - 'oscillate_wide': alternating up/down with big intervals
        - 'oscillate_narrow': alternating with small intervals
        - 'repeat': mostly same pitch
        - 'arch': up then down (or inverse)
        """
        valid = pitches > 0
        if valid.sum() < 2:
            return 'repeat'
        
        valid_pitches = pitches[valid]
        
        # Calculate deltas
        deltas = np.diff(valid_pitches)
        
        if len(deltas) == 0:
            return 'repeat'
        
        # Statistics
        ups = (deltas > 0.5).sum()
        downs = (deltas < -0.5).sum()
        same = len(deltas) - ups - downs
        
        total = len(deltas)
        
        # Classify
        if same > total * 0.6:
            return 'repeat'
        
        # Check for oscillation
        sign_changes = np.sum(np.diff(np.sign(deltas)) != 0)
        oscillation_ratio = sign_changes / max(1, len(deltas) - 1)
        
        if oscillation_ratio > 0.6:
            avg_jump = np.abs(deltas).mean()
            return 'oscillate_wide' if avg_jump > 3 else 'oscillate_narrow'
        
        # Check for arch
        if len(deltas) > 4:
            first_half = deltas[:len(deltas)//2]
            second_half = deltas[len(deltas)//2:]
            
            if (first_half.mean() > 0.5 and second_half.mean() < -0.5) or \
               (first_half.mean() < -0.5 and second_half.mean() > 0.5):
                return 'arch'
        
        # Simple direction
        if ups > downs * 1.5:
            return 'ascending'
        elif downs > ups * 1.5:
            return 'descending'
        
        return 'oscillate_narrow'
    
    def _apply_pattern(self, pitches: np.ndarray, pattern: str) -> np.ndarray:
        """Apply fret assignment based on phrase pattern."""
        n = len(pitches)
        frets = np.zeros(n, dtype=int)
        
        # Check if we have valid pitch data
        valid_ratio = (pitches > 0).mean() if len(pitches) > 0 else 0
        
        if valid_ratio < 0.2:
            # Most pitches invalid - use rhythmic variety
            # Cycle through simple patterns to avoid constant same fret
            for i in range(n):
                # 4-beat cycling pattern: Green, Red, Yellow, Blue, repeat
                frets[i] = i % 4
            return frets
        
        if pattern == 'repeat':
            # Slight variation even for repeat - alternate 1 and 3
            for i in range(n):
                frets[i] = 1 if (i % 2 == 0) else 3
        
        elif pattern == 'ascending':
            # Start low, end high
            frets = np.linspace(0, 4, n).round().astype(int)
        
        elif pattern == 'descending':
            # Start high, end low
            frets = np.linspace(4, 0, n).round().astype(int)
        
        elif pattern == 'oscillate_wide':
            # Alternate between low and high (0-4, 4-0, 0-4...)
            for i in range(n):
                frets[i] = 0 if (i % 2 == 0) else 4
        
        elif pattern == 'oscillate_narrow':
            # Alternate between adjacent frets
            center = 2
            for i in range(n):
                offset = (i % 2) * 2 - 1  # -1 or +1
                frets[i] = np.clip(center + offset, 0, 4)
        
        elif pattern == 'arch':
            # Up to peak, then down
            mid = n // 2
            frets[:mid] = np.linspace(0, 4, mid).round().astype(int)
            frets[mid:] = np.linspace(4, 0, n - mid).round().astype(int)
        
        return frets
    
    def _build_notes(
        self, 
        times: np.ndarray, 
        frets: np.ndarray, 
        pitches: np.ndarray,
        chord_mask: np.ndarray = None,
        durations: np.ndarray = None,
        onset_energies: np.ndarray = None
    ) -> List[ChartNote]:
        """Build chart notes with HOPOs, sustains, and chords (2-note and 3-note)."""
        notes = []
        
        if chord_mask is None:
            chord_mask = np.zeros(len(times), dtype=bool)
        
        if durations is None:
            durations = np.full(len(times), 0.1)  # Default short duration
        
        if onset_energies is None:
            onset_energies = np.ones(len(times))  # Default equal energy
        
        for i in range(len(times)):
            t = times[i]
            fret = frets[i]
            pitch = pitches[i] if i < len(pitches) else 0
            duration = durations[i] if i < len(durations) else 0.1
            
            # Gap to next note (for HOPO detection)
            if i < len(times) - 1:
                gap_to_next = times[i + 1] - t
            else:
                gap_to_next = 1.0
            
            # HOPO: different fret and quick succession (not for chords)
            is_hopo = False
            if not chord_mask[i] and i > 0 and gap_to_next < self.hopo_threshold_sec:
                if frets[i] != frets[i - 1]:
                    is_hopo = True
            
            # Sustain: based on actual note duration from audio analysis
            is_sustain = duration >= self.sustain_min_sec
            
            # Chord: add secondary fret(s) - support 2-note and 3-note chords
            chord_frets = None
            if chord_mask[i]:
                # Determine chord size based on energy - top 10% get 3-note chords
                is_power_chord = onset_energies[i] >= np.percentile(onset_energies, 90) if i < len(onset_energies) else False
                
                if is_power_chord:
                    # 3-note power chord - classic rock voicing
                    # Pick frets that are playable (not too spread out)
                    if fret <= 1:
                        # Low root: add 3rd and 5th (e.g., G-Y-B)
                        chord_frets = [fret + 2, fret + 3] if fret == 0 else [fret + 1, fret + 3]
                    elif fret >= 3:
                        # High root: add lower frets (e.g., G-Y-O or G-R-O)
                        chord_frets = [fret - 2, fret - 3] if fret == 4 else [fret - 1, fret - 3]
                    else:
                        # Middle root (fret 2 = yellow): G-Y-B (0, 2, 3)
                        chord_frets = [fret - 2, fret + 1]
                else:
                    # 2-note chord
                    if fret <= 1:
                        chord_frets = [fret + 2]  # Add yellow or blue
                    elif fret >= 3:
                        chord_frets = [fret - 2]  # Add lower fret
                    else:
                        # Middle fret - add one above
                        chord_frets = [fret + 1] if fret < 4 else [fret - 1]
                
                # Clamp all chord frets to valid range [0-4]
                chord_frets = [max(0, min(4, f)) for f in chord_frets]
                # Remove duplicates and the root fret
                chord_frets = [f for f in set(chord_frets) if f != fret]
            
            notes.append(ChartNote(
                time_sec=float(t),
                fret=int(fret),
                duration_sec=float(min(duration, 2.0)),  # Cap at 2s
                is_hopo=is_hopo,
                is_sustain=is_sustain,
                midi_pitch=int(pitch) if pitch > 0 else None,
                chord_frets=chord_frets,
            ))
        
        return notes


def export_to_midi(notes: List[ChartNote], output_path: Path, tempo: float = 120.0):
    """Export chart notes to MIDI file, including chords."""
    import mido
    
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Tempo
    tempo_us = int(60_000_000 / tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us))
    
    ticks_per_beat = mid.ticks_per_beat
    
    # Guitar fret to MIDI note mapping (Expert difficulty)
    FRET_TO_MIDI = {0: 96, 1: 97, 2: 98, 3: 99, 4: 100}
    
    # Sort by time
    notes = sorted(notes, key=lambda n: n.time_sec)
    
    # Build events list (time, is_on, note, velocity)
    events = []
    for note in notes:
        midi_note = FRET_TO_MIDI[note.fret]
        on_time = note.time_sec
        
        # KEY FIX: Only sustain notes get real duration
        # Non-sustain notes get very short duration (1 tick = ~10ms)
        # This prevents visual sustain tails in Clone Hero
        if note.is_sustain:
            # Real sustain - use actual duration (capped at 2s)
            off_time = on_time + min(note.duration_sec, 2.0)
        else:
            # Short tap - use minimal duration (will be 1 tick in MIDI)
            off_time = on_time + 0.01  # 10ms
        
        # Primary note
        events.append((on_time, True, midi_note, 100))
        events.append((off_time, False, midi_note, 0))
        
        # Chord notes
        if note.chord_frets:
            for chord_fret in note.chord_frets:
                chord_midi = FRET_TO_MIDI[chord_fret]
                events.append((on_time, True, chord_midi, 100))
                events.append((off_time, False, chord_midi, 0))
    
    # Sort by time, with note_off before note_on at same time
    events.sort(key=lambda e: (e[0], e[1]))  # False (off) < True (on)
    
    last_time = 0.0
    for time_sec, is_on, midi_note, velocity in events:
        delta_sec = time_sec - last_time
        delta_ticks = int(delta_sec * tempo / 60 * ticks_per_beat)
        delta_ticks = max(0, delta_ticks)
        
        msg_type = 'note_on' if is_on else 'note_off'
        track.append(mido.Message(msg_type, note=midi_note, velocity=velocity, time=delta_ticks))
        
        last_time = time_sec
    
    mid.save(output_path)
    
    # Count chords
    n_chords = sum(1 for n in notes if n.chord_frets)
    logger.info(f"Exported {len(notes)} notes ({n_chords} chords) to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid guitar charting")
    parser.add_argument('--checkpoint', type=Path, required=True,
                       help='Path to onset model checkpoint')
    parser.add_argument('--other', type=Path, required=True,
                       help='Path to other.wav (guitar stem)')
    parser.add_argument('--bass', type=Path, required=True,
                       help='Path to bass.wav')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output MIDI path')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Onset detection threshold')
    parser.add_argument('--tempo', type=float, default=120.0,
                       help='Tempo for MIDI export')
    
    args = parser.parse_args()
    
    charter = HybridGuitarCharter(
        onset_checkpoint=args.checkpoint,
        onset_threshold=args.threshold,
    )
    
    notes = charter.chart_from_stems(args.other, args.bass)
    
    logger.info(f"Generated {len(notes)} notes")
    logger.info(f"HOPOs: {sum(1 for n in notes if n.is_hopo)}")
    logger.info(f"Sustains: {sum(1 for n in notes if n.is_sustain)}")
    
    export_to_midi(notes, args.output, args.tempo)


if __name__ == '__main__':
    main()
