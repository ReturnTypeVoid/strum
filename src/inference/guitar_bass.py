"""
Guitar/Bass Transcription Pipeline

Librosa-based onset detection + pYIN pitch tracking + rule-based fret mapping.
Produces Clone Hero / YARG compatible charts with 4 difficulty levels.

No neural model required — uses signal processing for onset detection.
"""

import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Clone Hero MIDI note offsets per difficulty
FRET_NOTE_OFFSETS = {
    "easy": 60,
    "medium": 72,
    "hard": 84,
    "expert": 96,
}


@dataclass
class GuitarNote:
    """A single note in a guitar/bass chart."""
    time_ms: float
    fret: int  # 0-4
    duration_ms: float = 100.0
    velocity: int = 100
    is_hopo: bool = False
    is_chord: bool = False
    midi_pitch: Optional[float] = None


@dataclass
class GuitarChord:
    """A chord (multiple simultaneous frets)."""
    time_ms: float
    frets: List[int]  # List of fret indices (0-4)
    duration_ms: float = 100.0
    velocity: int = 100


@dataclass
class GuitarChart:
    """Complete guitar/bass chart with notes and chords."""
    notes: List[GuitarNote] = field(default_factory=list)
    chords: List[GuitarChord] = field(default_factory=list)
    tempo_bpm: float = 120.0
    instrument: str = "guitar"


def transcribe_guitar(
    audio_path: Path,
    tempo_bpm: float = 0.0,
    confidence_threshold: float = 0.3,
    is_bass: bool = False,
) -> GuitarChart:
    """
    Transcribe guitar or bass from an audio stem.

    Args:
        audio_path: Path to the audio stem (other.wav for guitar, bass.wav for bass).
        tempo_bpm: Song tempo in BPM. If 0, auto-detect from audio.
        confidence_threshold: Onset detection sensitivity (lower = more notes).
        is_bass: If True, use bass-specific parameters.

    Returns:
        GuitarChart with expert-level notes and chords.
    """
    sr = 22050
    hop_length = 512

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    duration_sec = len(y) / sr

    if len(y) < sr:
        logger.warning(f"Audio too short ({duration_sec:.1f}s), returning empty chart")
        return GuitarChart(tempo_bpm=tempo_bpm or 120.0, instrument="bass" if is_bass else "guitar")

    # Auto-detect tempo if not provided
    if tempo_bpm <= 0:
        tempo_arr = librosa.feature.rhythm.tempo(y=y, sr=sr)
        tempo_bpm = float(tempo_arr[0]) if hasattr(tempo_arr, '__len__') else float(tempo_arr)
        logger.info(f"Auto-detected tempo: {tempo_bpm:.1f} BPM")

    # --- Stage 1: Onset detection ---
    onset_times = _detect_onsets(y, sr, hop_length, tempo_bpm, confidence_threshold, is_bass)
    logger.info(f"Detected {len(onset_times)} raw onsets")

    if len(onset_times) == 0:
        return GuitarChart(tempo_bpm=tempo_bpm, instrument="bass" if is_bass else "guitar")

    # --- Stage 2: Quantize to beat grid (before all other processing) ---
    onset_times = _quantize_to_grid(onset_times, tempo_bpm)
    logger.info(f"After quantization: {len(onset_times)} onsets")

    if len(onset_times) == 0:
        return GuitarChart(tempo_bpm=tempo_bpm, instrument="bass" if is_bass else "guitar")

    # --- Stage 3: Pitch detection ---
    pitches, pitch_confidences = _detect_pitches(y, sr, hop_length, is_bass)

    # Get pitch at each onset (using quantized times)
    onset_pitches = _sample_pitches_at_onsets(pitches, pitch_confidences, onset_times, sr, hop_length)
    valid_ratio = (onset_pitches > 0).mean()
    logger.info(f"Valid pitches: {(onset_pitches > 0).sum()}/{len(onset_pitches)} ({valid_ratio*100:.1f}%)")

    # --- Stage 4: Fret assignment ---
    frets = _assign_frets(onset_pitches, onset_times, is_bass)

    # --- Stage 5: Duration detection ---
    durations_sec = _detect_durations(y, sr, hop_length, onset_times)

    # --- Stage 6: Chord detection (guitar only) ---
    chord_mask = np.zeros(len(onset_times), dtype=bool)
    if not is_bass:
        chord_mask = _detect_chords(y, sr, hop_length, onset_times, durations_sec)

    # --- Stage 7: HOPO detection (tempo-aware, on quantized times) ---
    hopo_mask = _detect_hopos(onset_times, frets, chord_mask, tempo_bpm)

    # --- Stage 8: Build chart ---
    chart = _build_chart(
        onset_times, frets, durations_sec, onset_pitches,
        chord_mask, hopo_mask, tempo_bpm, is_bass,
    )

    return chart


def transcribe_bass(
    audio_path: Path,
    tempo_bpm: float = 120.0,
    confidence_threshold: float = 0.4,
) -> GuitarChart:
    """Transcribe bass. Wrapper around transcribe_guitar with bass=True."""
    return transcribe_guitar(
        audio_path, tempo_bpm, confidence_threshold, is_bass=True,
    )


# ---------------------------------------------------------------------------
# Onset Detection
# ---------------------------------------------------------------------------

def _detect_onsets(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    tempo_bpm: float,
    threshold: float,
    is_bass: bool,
) -> np.ndarray:
    """
    Hybrid onset detection: spectral peaks + beat-grid rhythm.

    Pure onset detection misses many guitar strums because rhythm guitar
    plays at consistent energy (no clear transient peaks). This hybrid
    approach combines:
    1. Spectral onset peaks — catches leads, accents, fills
    2. Beat-grid placement — catches rhythmic strumming

    Both are energy-gated to avoid placing notes during silence.
    """
    from scipy.signal import find_peaks

    duration = len(y) / sr

    # --- Spectral onset peaks ---
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    oenv_norm = oenv / (oenv.max() + 1e-8)

    # Height threshold: 0.07 gives ~4-5 nps for rock guitar
    peak_height = 0.07 if not is_bass else 0.10
    peaks, _ = find_peaks(oenv_norm, height=peak_height, distance=2)

    # Backtrack peaks to the nearest preceding local minimum of onset
    # strength — the spectral flux peak occurs ~50ms AFTER the actual
    # transient attack, so we slide back to where the attack starts.
    peaks = librosa.onset.onset_backtrack(peaks, oenv)

    peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)

    # --- Beat-grid rhythm notes (8th note subdivisions) ---
    # Use beat_track only for beat POSITIONS (phase); use the pipeline's
    # BPM (grid-aligned) for subdivision spacing.  beat_track's tempo
    # estimate is often a subharmonic (half/3-quarter) of the real BPM.
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)

    beat_sec = 60.0 / tempo_bpm
    grid_times = []
    for bt in beat_times:
        grid_times.append(bt)
        grid_times.append(bt + beat_sec * 0.5)  # 8th note subdivision
    grid_times = np.array(sorted(grid_times))

    # Energy-gate grid notes: only place where guitar is audible
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    energy_floor = np.percentile(rms, 15)

    # Also compute local energy density for 16th note detection
    # High-energy sections with strong onset activity get 16th notes
    oenv_smooth = np.convolve(oenv, np.ones(20)/20, mode='same')
    oenv_high = np.percentile(oenv_smooth, 70)

    filtered_grid = []
    for t in grid_times:
        idx = np.argmin(np.abs(rms_times - t))
        if idx < len(rms) and rms[idx] > energy_floor:
            filtered_grid.append(t)

            # In high-energy sections, also add 16th note subdivisions
            oenv_idx = np.argmin(np.abs(rms_times - t))
            if oenv_idx < len(oenv_smooth) and oenv_smooth[oenv_idx] > oenv_high and not is_bass:
                # Add 16th note between this 8th and the next
                sixteenth = t + beat_sec * 0.25
                sixteenth_idx = np.argmin(np.abs(rms_times - sixteenth))
                if sixteenth_idx < len(rms) and rms[sixteenth_idx] > energy_floor:
                    filtered_grid.append(sixteenth)

    filtered_grid = np.array(sorted(filtered_grid)) if filtered_grid else np.array([])

    # --- Merge onset peaks + beat grid ---
    all_times = np.sort(np.concatenate([peak_times, filtered_grid]))

    # Filter time range
    all_times = all_times[(all_times >= 0.2) & (all_times <= duration - 0.5)]

    if len(all_times) == 0:
        return np.array([])

    # Deduplicate (within 30ms)
    deduped = [all_times[0]]
    for t in all_times[1:]:
        if t - deduped[-1] > 0.03:
            deduped.append(t)
    onset_times = np.array(deduped)

    # Cap note density
    max_nps = 6.0 if is_bass else 10.0
    onset_times = _cap_note_density(onset_times, max_nps)

    return onset_times


def _cap_note_density(onset_times: np.ndarray, max_nps: float) -> np.ndarray:
    """Cap note density by removing weakest onsets in dense windows."""
    if len(onset_times) <= 1:
        return onset_times

    min_gap = 1.0 / max_nps
    filtered = [onset_times[0]]
    for t in onset_times[1:]:
        if t - filtered[-1] >= min_gap:
            filtered.append(t)
    return np.array(filtered)


# ---------------------------------------------------------------------------
# Pitch Detection
# ---------------------------------------------------------------------------

def _detect_pitches(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    is_bass: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect pitches using pYIN.

    Returns:
        pitches: MIDI pitch per frame (0 where invalid)
        confidences: Voiced probability per frame
    """
    fmin = librosa.note_to_hz('E1' if is_bass else 'E2')  # Low E on bass/guitar
    fmax = librosa.note_to_hz('C5' if is_bass else 'E6')  # Reasonable upper range

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
        fill_na=0.0,
    )

    # Convert to MIDI
    pitches = np.zeros_like(f0)
    valid = ~np.isnan(f0) & (f0 > 0)
    pitches[valid] = librosa.hz_to_midi(f0[valid])

    confidences = np.where(np.isnan(voiced_probs), 0.0, voiced_probs)

    return pitches, confidences


def _sample_pitches_at_onsets(
    pitches: np.ndarray,
    confidences: np.ndarray,
    onset_times: np.ndarray,
    sr: int,
    hop_length: int,
) -> np.ndarray:
    """
    Get the pitch at each onset time.

    Uses a small window around the onset and picks the pitch with highest
    confidence, which is more robust than just taking the nearest frame.
    """
    frame_times = librosa.frames_to_time(np.arange(len(pitches)), sr=sr, hop_length=hop_length)
    onset_pitches = np.zeros(len(onset_times))

    # Window: ±3 frames (~70ms at hop=512, sr=22050)
    window = 3

    for i, t in enumerate(onset_times):
        center = np.argmin(np.abs(frame_times - t))
        lo = max(0, center - window)
        hi = min(len(pitches), center + window + 1)

        region_p = pitches[lo:hi]
        region_c = confidences[lo:hi]

        # Pick pitch of frame with highest confidence
        valid = region_p > 0
        if valid.any():
            best_idx = np.argmax(region_c[valid]) if valid.sum() > 0 else 0
            onset_pitches[i] = region_p[valid][best_idx]

    return onset_pitches


# ---------------------------------------------------------------------------
# Fret Assignment
# ---------------------------------------------------------------------------

def _assign_frets(
    pitches: np.ndarray,
    times: np.ndarray,
    is_bass: bool,
) -> np.ndarray:
    """
    Map pitches to frets 0-4.

    Uses windowed normalization: pitches are normalized relative to a local
    window (8 seconds) rather than the entire song. This adapts to key changes
    and sections with different registers.

    When pitch is unavailable, uses context-aware interpolation instead of
    nonsensical cycling patterns.
    """
    n = len(pitches)
    if n == 0:
        return np.array([], dtype=int)

    frets = np.full(n, 2, dtype=int)  # Default to middle fret
    valid_mask = pitches > 0
    valid_ratio = valid_mask.mean()

    if valid_ratio < 0.1:
        # Almost no pitch data — use energy-based simple assignment
        logger.warning(f"Very low pitch validity ({valid_ratio*100:.1f}%), using simple assignment")
        return _assign_frets_no_pitch(times, n)

    # Window-based normalization (8-second windows with 4s overlap)
    window_sec = 8.0
    step_sec = 4.0

    # First pass: compute frets for notes with valid pitch
    raw_frets = np.full(n, -1, dtype=int)

    for win_start in np.arange(0, times[-1] + step_sec, step_sec):
        win_end = win_start + window_sec
        in_window = (times >= win_start) & (times < win_end) & valid_mask

        if in_window.sum() < 2:
            continue

        win_pitches = pitches[in_window]

        # Use 10th/90th percentiles for robustness
        p_lo = np.percentile(win_pitches, 10)
        p_hi = np.percentile(win_pitches, 90)
        p_range = max(p_hi - p_lo, 2.0)  # At least 2 semitones range

        # Normalize to 0-4
        indices = np.where(in_window)[0]
        for idx in indices:
            if raw_frets[idx] >= 0:
                continue  # Already assigned by earlier window
            norm = (pitches[idx] - p_lo) / p_range
            norm = np.clip(norm, 0.0, 1.0)
            raw_frets[idx] = int(round(norm * 4))

    # Copy valid assignments
    assigned = raw_frets >= 0
    frets[assigned] = raw_frets[assigned]

    # Second pass: fill gaps using interpolation from neighbors
    for i in range(n):
        if assigned[i]:
            continue

        # Find nearest valid fret before and after
        prev_fret = _find_nearest_fret(frets, assigned, i, direction=-1)
        next_fret = _find_nearest_fret(frets, assigned, i, direction=+1)

        if prev_fret >= 0 and next_fret >= 0:
            # Interpolate (round toward previous for ties)
            frets[i] = int(round((prev_fret + next_fret) / 2 - 0.01))
        elif prev_fret >= 0:
            frets[i] = prev_fret
        elif next_fret >= 0:
            frets[i] = next_fret
        # else: stays at default (2)

    # Post-process: penalize large jumps (ergonomics)
    frets = _smooth_fret_jumps(frets, times, max_jump=3)

    return frets


def _find_nearest_fret(frets: np.ndarray, valid: np.ndarray, idx: int, direction: int) -> int:
    """Find nearest valid fret in given direction. Returns -1 if not found within 10 notes."""
    for offset in range(1, min(11, len(frets))):
        j = idx + direction * offset
        if 0 <= j < len(frets) and valid[j]:
            return frets[j]
    return -1


def _assign_frets_no_pitch(times: np.ndarray, n: int) -> np.ndarray:
    """
    Fallback fret assignment when pitch detection fails.

    Uses phrase-based patterns: detect phrases by time gaps,
    assign repeating patterns within each phrase.
    """
    frets = np.zeros(n, dtype=int)
    # Phrase patterns that feel musical in Clone Hero
    patterns = [
        [0, 1, 2, 1],      # Green-Red-Yellow-Red
        [0, 0, 1, 2],      # GG-RY
        [2, 1, 0, 1],      # Y-R-G-R (descending)
        [0, 1, 2, 3],      # Ascending scale
        [3, 2, 1, 0],      # Descending scale
    ]

    phrase_idx = 0
    note_in_phrase = 0

    for i in range(n):
        if i > 0:
            gap = times[i] - times[i - 1]
            if gap > 0.8:
                # New phrase
                phrase_idx = (phrase_idx + 1) % len(patterns)
                note_in_phrase = 0

        pat = patterns[phrase_idx]
        frets[i] = pat[note_in_phrase % len(pat)]
        note_in_phrase += 1

    return frets


def _smooth_fret_jumps(frets: np.ndarray, times: np.ndarray, max_jump: int = 3) -> np.ndarray:
    """
    Reduce large fret jumps for playability.

    If a note jumps more than max_jump frets from both its neighbors,
    pull it toward the midpoint.
    """
    result = frets.copy()

    for i in range(1, len(result) - 1):
        gap_before = times[i] - times[i - 1]
        gap_after = times[i + 1] - times[i]

        # Only smooth in fast passages (< 300ms gaps)
        if gap_before > 0.3 or gap_after > 0.3:
            continue

        jump_before = abs(result[i] - result[i - 1])
        jump_after = abs(result[i] - result[i + 1])

        if jump_before > max_jump and jump_after > max_jump:
            # Pull toward neighbors' midpoint
            mid = (result[i - 1] + result[i + 1]) / 2
            result[i] = int(np.clip(round(mid), 0, 4))

    return result


# ---------------------------------------------------------------------------
# Duration Detection
# ---------------------------------------------------------------------------

def _detect_durations(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    onset_times: np.ndarray,
) -> np.ndarray:
    """
    Detect note durations from audio energy decay.

    Sustains are only applied to isolated notes (long gaps to next note).
    Dense passages get short tap durations.
    """
    if len(onset_times) == 0:
        return np.array([])

    # Compute RMS energy envelope
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    durations = np.full(len(onset_times), 0.1)  # Default 100ms

    for i, onset_t in enumerate(onset_times):
        # Gap to next note — determines max possible duration
        if i < len(onset_times) - 1:
            gap_to_next = onset_times[i + 1] - onset_t
        else:
            gap_to_next = 4.0

        # In dense passages (< 300ms gap), use short duration
        if gap_to_next < 0.3:
            durations[i] = min(gap_to_next * 0.7, 0.15)
            continue

        # Find onset energy
        onset_frame = np.argmin(np.abs(rms_times - onset_t))
        if onset_frame >= len(rms):
            continue

        # Peak energy in 100ms after onset
        peak_end = min(onset_frame + int(0.1 * sr / hop_length), len(rms))
        peak_energy = rms[onset_frame:peak_end].max() if peak_end > onset_frame else rms[onset_frame]

        if peak_energy < 1e-6:
            continue

        # Search for decay to 20% of peak
        decay_threshold = peak_energy * 0.20
        max_search = min(onset_t + min(gap_to_next * 0.85, 4.0), rms_times[-1])
        max_frame = np.argmin(np.abs(rms_times - max_search))

        duration = 0.1
        for j in range(onset_frame, min(max_frame, len(rms))):
            if rms[j] < decay_threshold:
                duration = rms_times[j] - onset_t
                break
        else:
            duration = max_search - onset_t

        durations[i] = np.clip(duration, 0.05, 4.0)

    return durations


# ---------------------------------------------------------------------------
# Chord Detection
# ---------------------------------------------------------------------------

def _detect_chords(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    onset_times: np.ndarray,
    durations: np.ndarray,
) -> np.ndarray:
    """
    Detect chords using onset strength + spectral width.

    Strong onsets with broad spectral energy = strummed chords.
    Weak onsets with narrow spectrum = single notes.
    """
    if len(onset_times) == 0:
        return np.array([], dtype=bool)

    # Onset strength at each onset
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    oenv_times = librosa.frames_to_time(np.arange(len(oenv)), sr=sr, hop_length=hop_length)

    # Spectral bandwidth — wider = more notes sounding
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]

    chord_mask = np.zeros(len(onset_times), dtype=bool)
    onset_strengths = np.zeros(len(onset_times))
    onset_bandwidths = np.zeros(len(onset_times))

    for i, t in enumerate(onset_times):
        frame = np.argmin(np.abs(oenv_times - t))
        if frame < len(oenv):
            onset_strengths[i] = oenv[frame]
        if frame < len(bandwidth):
            onset_bandwidths[i] = bandwidth[frame]

    # Thresholds: top 30% of both strength and bandwidth
    if len(onset_strengths) > 4:
        strength_thresh = np.percentile(onset_strengths, 70)
        bw_thresh = np.percentile(onset_bandwidths, 70)

        n_chords = 0
        max_chords = len(onset_times) // 4  # Cap at 25%

        for i in range(len(onset_times)):
            if n_chords >= max_chords:
                break
            if onset_strengths[i] >= strength_thresh and onset_bandwidths[i] >= bw_thresh:
                chord_mask[i] = True
                n_chords += 1

    logger.info(f"Detected {chord_mask.sum()} chords ({100*chord_mask.mean():.1f}%)")
    return chord_mask


# ---------------------------------------------------------------------------
# HOPO Detection
# ---------------------------------------------------------------------------

def _detect_hopos(
    onset_times: np.ndarray,
    frets: np.ndarray,
    chord_mask: np.ndarray,
    tempo_bpm: float,
) -> np.ndarray:
    """
    Detect hammer-ons / pull-offs (tempo-aware).

    HOPO threshold is based on beat subdivision:
    - At 120 BPM: 1/8 note = 250ms -> threshold ~200ms
    - At 180 BPM: 1/8 note = 167ms -> threshold ~133ms

    C3 rule: notes within 1/12 of a beat get HOPOs if they differ in fret
    and are not chords.
    """
    if len(onset_times) == 0:
        return np.array([], dtype=bool)

    # HOPO threshold: standard Clone Hero ~170ms window
    # At high tempos this catches 8th notes; at lower tempos only 16ths get HOPOs
    beat_sec = 60.0 / tempo_bpm
    # Use 65/192 of a beat (Clone Hero standard) with a 170ms floor
    hopo_threshold = max(beat_sec * 65 / 192, 0.100)
    hopo_threshold = min(hopo_threshold, 0.200)  # Cap at 200ms

    hopo_mask = np.zeros(len(onset_times), dtype=bool)

    for i in range(1, len(onset_times)):
        if chord_mask[i] or chord_mask[i - 1]:
            continue  # No HOPOs on chords

        gap = onset_times[i] - onset_times[i - 1]
        if gap <= hopo_threshold and frets[i] != frets[i - 1]:
            hopo_mask[i] = True

    logger.info(f"HOPOs: {hopo_mask.sum()}/{len(onset_times)} ({100*hopo_mask.mean():.1f}%)")
    return hopo_mask


# ---------------------------------------------------------------------------
# Grid Quantization
# ---------------------------------------------------------------------------

def _quantize_to_grid(
    onset_times: np.ndarray,
    tempo_bpm: float,
    grid: str = '1/16',
    strength: float = 1.0,
) -> np.ndarray:
    """
    Soft quantization to beat grid.

    Notes are pulled toward nearest grid point with given strength.
    Deduplicates notes that land on the same grid position.
    """
    if len(onset_times) == 0:
        return onset_times

    beat_sec = 60.0 / tempo_bpm
    grid_factors = {'1/4': 1.0, '1/8': 0.5, '1/16': 0.25, '1/32': 0.125}
    grid_interval = beat_sec * grid_factors.get(grid, 0.25)

    quantized = np.zeros_like(onset_times)
    for i, t in enumerate(onset_times):
        nearest_grid = round(t / grid_interval) * grid_interval
        quantized[i] = t + strength * (nearest_grid - t)

    # Deduplicate (keep first note at each grid position)
    min_gap = grid_interval * 0.5
    deduped = [quantized[0]]
    for t in quantized[1:]:
        if t - deduped[-1] >= min_gap:
            deduped.append(t)

    return np.array(deduped)


# ---------------------------------------------------------------------------
# Chart Building
# ---------------------------------------------------------------------------

def _build_chart(
    onset_times: np.ndarray,
    frets: np.ndarray,
    durations_sec: np.ndarray,
    pitches: np.ndarray,
    chord_mask: np.ndarray,
    hopo_mask: np.ndarray,
    tempo_bpm: float,
    is_bass: bool,
) -> GuitarChart:
    """Assemble final GuitarChart from all computed features."""

    # Sustain threshold — 500ms for guitar, 700ms for bass
    sustain_min = 0.7 if is_bass else 0.5

    notes = []
    chords = []

    for i in range(len(onset_times)):
        t_ms = onset_times[i] * 1000.0
        fret = int(np.clip(frets[i], 0, 4)) if i < len(frets) else 2
        dur_ms = durations_sec[i] * 1000.0 if i < len(durations_sec) else 100.0
        pitch = pitches[i] if i < len(pitches) else 0.0
        is_chord = chord_mask[i] if i < len(chord_mask) else False
        is_hopo = hopo_mask[i] if i < len(hopo_mask) else False

        # Sustain: only if duration exceeds threshold AND next note is far
        is_sustain = (durations_sec[i] >= sustain_min) if i < len(durations_sec) else False

        if not is_sustain:
            dur_ms = min(dur_ms, 100.0)  # Short tap

        note = GuitarNote(
            time_ms=t_ms,
            fret=fret,
            duration_ms=dur_ms,
            velocity=100,
            is_hopo=is_hopo,
            is_chord=is_chord,
            midi_pitch=float(pitch) if pitch > 0 else None,
        )
        notes.append(note)

        # Build chord object
        if is_chord:
            chord_frets = _compute_chord_frets(fret, pitches[i] if i < len(pitches) else 0)
            if chord_frets:
                chords.append(GuitarChord(
                    time_ms=t_ms,
                    frets=[fret] + chord_frets,
                    duration_ms=dur_ms,
                    velocity=100,
                ))
                # Add chord fret notes to the chart at the same time
                for cf in chord_frets:
                    notes.append(GuitarNote(
                        time_ms=t_ms,
                        fret=cf,
                        duration_ms=dur_ms,
                        velocity=100,
                        is_hopo=False,
                        is_chord=True,
                    ))

    # Sort by time
    notes.sort(key=lambda n: (n.time_ms, n.fret))

    instrument = "bass" if is_bass else "guitar"
    logger.info(f"{instrument.title()} chart: {len(notes)} notes, {len(chords)} chords")

    return GuitarChart(
        notes=notes,
        chords=chords,
        tempo_bpm=tempo_bpm,
        instrument=instrument,
    )


def _compute_chord_frets(root_fret: int, pitch: float) -> List[int]:
    """
    Compute additional frets for a chord based on root fret.

    Uses standard 2-note power chord voicings.
    """
    if root_fret <= 1:
        return [root_fret + 2]
    elif root_fret >= 3:
        return [root_fret - 2]
    else:
        return [root_fret + 1]


# ---------------------------------------------------------------------------
# Difficulty Reduction
# ---------------------------------------------------------------------------

def reduce_to_difficulty(chart: GuitarChart, difficulty: str) -> GuitarChart:
    """
    Reduce expert chart to lower difficulty.

    C3 authoring rules:
    - Hard: Remove some HOPOs and chords, max 4 frets (0-3)
    - Medium: Remove all chords, max 4 frets, thin notes ~70%
    - Easy: No chords, no HOPOs, max 3 frets (0-2), thin notes ~50%
    """
    if difficulty == "expert":
        return chart

    notes = chart.notes[:]

    is_bass = chart.instrument == "bass"

    if difficulty == "hard":
        notes = _reduce_hard(notes, is_bass=is_bass)
    elif difficulty == "medium":
        notes = _reduce_medium(notes, is_bass=is_bass)
    elif difficulty == "easy":
        notes = _reduce_easy(notes, is_bass=is_bass)

    return GuitarChart(
        notes=notes,
        chords=chart.chords if difficulty == "hard" else [],
        tempo_bpm=chart.tempo_bpm,
        instrument=chart.instrument,
    )


def _reduce_hard(notes: List[GuitarNote], is_bass: bool = False) -> List[GuitarNote]:
    """Hard: cap frets at 0-3, thin to ~65% (guitar) or ~85% (bass)."""
    # Remove chord duplicates (keep root only for some chords)
    seen_times = set()
    unique_notes = []
    for note in notes:
        if note.time_ms in seen_times and note.is_chord:
            continue
        seen_times.add(note.time_ms)
        unique_notes.append(note)

    # Bass: only thin moderately (wider gaps); Guitar: thin < 350ms gaps
    gap_threshold = 500 if is_bass else 350
    skip_mod = 5 if is_bass else 3  # bass: skip every 5th, guitar: every 3rd

    result = []
    skip_counter = 0
    for i, note in enumerate(unique_notes):
        if i > 0:
            gap = note.time_ms - unique_notes[i - 1].time_ms
            if gap < gap_threshold:
                skip_counter += 1
                if skip_counter % skip_mod == 0:
                    continue
            else:
                skip_counter = 0

        result.append(GuitarNote(
            time_ms=note.time_ms,
            fret=min(note.fret, 3),
            duration_ms=note.duration_ms,
            velocity=note.velocity,
            is_hopo=note.is_hopo,
            is_chord=note.is_chord,
        ))
    return result


def _reduce_medium(notes: List[GuitarNote], is_bass: bool = False) -> List[GuitarNote]:
    """Medium: no chords, max 4 frets 0-3, thin to ~55% (guitar) or ~65% (bass)."""
    # Remove chord duplicates (keep only root notes)
    seen_times = set()
    unique_notes = []
    for note in notes:
        if note.time_ms in seen_times and note.is_chord:
            continue
        seen_times.add(note.time_ms)
        unique_notes.append(note)

    result = []
    skip_counter = 0
    for i, note in enumerate(unique_notes):
        if i > 0:
            gap = note.time_ms - unique_notes[i - 1].time_ms
            if is_bass:
                # Bass: lighter thinning — skip every 3rd in passages < 500ms
                if gap < 500:
                    skip_counter += 1
                    if skip_counter % 3 == 0:
                        continue
                else:
                    skip_counter = 0
            else:
                # Guitar: heavier thinning
                if gap < 200:
                    skip_counter += 1
                    if skip_counter % 2 == 0:
                        continue
                elif gap < 500:
                    skip_counter += 1
                    if skip_counter % 3 == 0:
                        continue
                else:
                    skip_counter = 0

        result.append(GuitarNote(
            time_ms=note.time_ms,
            fret=min(note.fret, 3),
            duration_ms=note.duration_ms,
            velocity=note.velocity,
            is_hopo=False,
            is_chord=False,
        ))
    return result


def _reduce_easy(notes: List[GuitarNote], is_bass: bool = False) -> List[GuitarNote]:
    """Easy: no chords, no HOPOs, max 3 frets 0-2, thin to ~20-25% (guitar) or ~35% (bass)."""
    # Remove chord duplicates
    seen_times = set()
    unique_notes = []
    for note in notes:
        if note.time_ms in seen_times and note.is_chord:
            continue
        seen_times.add(note.time_ms)
        unique_notes.append(note)

    # Bass: moderate thinning; Guitar: aggressive thinning
    result = []
    skip_counter = 0
    for i, note in enumerate(unique_notes):
        if i > 0:
            gap = note.time_ms - unique_notes[i - 1].time_ms
            if is_bass:
                # Bass: keep ~1 in 3 for fast, ~1 in 2 for medium
                if gap < 400:
                    skip_counter += 1
                    if skip_counter % 3 != 0:
                        continue
                elif gap < 700:
                    skip_counter += 1
                    if skip_counter % 2 != 0:
                        continue
                else:
                    skip_counter = 0
            else:
                # Guitar: keep ~1 in 4 for fast, ~1 in 2 for medium
                if gap < 400:
                    skip_counter += 1
                    if skip_counter % 4 != 0:
                        continue
                elif gap < 800:
                    skip_counter += 1
                    if skip_counter % 2 != 0:
                        continue
                else:
                    skip_counter = 0

        result.append(GuitarNote(
            time_ms=note.time_ms,
            fret=min(note.fret, 2),
            duration_ms=note.duration_ms,
            velocity=note.velocity,
            is_hopo=False,
            is_chord=False,
        ))
    return result
