"""
Audio-chart alignment using cross-correlation.

Detects and corrects timing offsets between audio and chart.
"""

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import torch
import torchaudio

from src.preprocessing.parsers.midi_parser import DrumChart, DrumHit

logger = logging.getLogger(__name__)


def compute_alignment_offset(
    audio_path: Path,
    chart: DrumChart,
    max_offset_ms: float = 100.0,
    search_resolution_ms: float = 1.0,
) -> float:
    """
    Compute the timing offset between audio and chart using cross-correlation.
    
    Creates a synthetic click track from chart onsets and cross-correlates
    with onset detection from the audio.
    
    Args:
        audio_path: Path to audio file (ideally drum stem)
        chart: Parsed drum chart
        max_offset_ms: Maximum offset to search (±ms)
        search_resolution_ms: Resolution of search in ms
        
    Returns:
        Offset in milliseconds (positive = chart is ahead of audio)
    """
    if not chart.hits:
        logger.warning("Empty chart, cannot compute alignment")
        return 0.0
    
    # Load audio
    audio, sample_rate = torchaudio.load(audio_path)
    audio = audio.mean(dim=0)  # Convert to mono
    
    # Convert to numpy
    audio_np = audio.numpy()
    
    # Compute onset strength from audio
    audio_onsets = _compute_onset_strength(audio_np, sample_rate)
    
    # Create synthetic onset track from chart
    duration_ms = chart.get_duration_ms() + 1000  # Add 1 second buffer
    chart_onsets = _create_onset_track(
        chart.hits,
        duration_ms,
        sample_rate,
        hop_length=512,
    )
    
    # Ensure same length
    min_len = min(len(audio_onsets), len(chart_onsets))
    audio_onsets = audio_onsets[:min_len]
    chart_onsets = chart_onsets[:min_len]
    
    # Cross-correlate
    offset_samples = _cross_correlate_offset(
        audio_onsets,
        chart_onsets,
        max_offset_ms,
        search_resolution_ms,
        sample_rate,
    )
    
    # Convert to milliseconds
    hop_length = 512
    offset_ms = (offset_samples * hop_length / sample_rate) * 1000
    
    logger.info(f"Computed alignment offset: {offset_ms:.2f} ms")
    
    return offset_ms


def apply_alignment_offset(
    chart: DrumChart,
    offset_ms: float,
) -> DrumChart:
    """
    Apply timing offset to all events in a chart.
    
    Args:
        chart: Original chart
        offset_ms: Offset to apply (positive shifts events later)
        
    Returns:
        New chart with adjusted timings
    """
    adjusted_hits = []
    for hit in chart.hits:
        adjusted_hits.append(DrumHit(
            time_ms=hit.time_ms + offset_ms,
            tick=hit.tick,
            lane=hit.lane,
            is_cymbal=hit.is_cymbal,
            velocity=hit.velocity,
        ))
    
    return DrumChart(
        hits=adjusted_hits,
        tempo_events=chart.tempo_events,
        time_signatures=chart.time_signatures,
        ticks_per_beat=chart.ticks_per_beat,
    )


def _compute_onset_strength(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Compute onset strength envelope from audio."""
    import librosa
    
    # Compute onset strength
    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sample_rate,
        hop_length=hop_length,
    )
    
    # Normalize
    onset_env = onset_env / (onset_env.max() + 1e-8)
    
    return onset_env


def _create_onset_track(
    hits: list[DrumHit],
    duration_ms: float,
    sample_rate: int,
    hop_length: int = 512,
) -> np.ndarray:
    """Create synthetic onset track from chart hits."""
    # Calculate number of frames
    duration_sec = duration_ms / 1000
    num_frames = int(duration_sec * sample_rate / hop_length)
    
    onset_track = np.zeros(num_frames)
    
    for hit in hits:
        # Convert time to frame index
        frame_idx = int((hit.time_ms / 1000) * sample_rate / hop_length)
        if 0 <= frame_idx < num_frames:
            onset_track[frame_idx] = 1.0
    
    # Apply slight smoothing (Gaussian kernel)
    from scipy.ndimage import gaussian_filter1d
    onset_track = gaussian_filter1d(onset_track, sigma=1.0)
    
    # Normalize
    onset_track = onset_track / (onset_track.max() + 1e-8)
    
    return onset_track


def _cross_correlate_offset(
    signal1: np.ndarray,
    signal2: np.ndarray,
    max_offset_ms: float,
    resolution_ms: float,
    sample_rate: int,
    hop_length: int = 512,
) -> int:
    """
    Find optimal offset using cross-correlation.
    
    Returns offset in frames (positive = signal2 should be shifted right).
    """
    # Calculate max lag in frames
    max_lag_frames = int((max_offset_ms / 1000) * sample_rate / hop_length)
    
    # Cross-correlate
    correlation = np.correlate(signal1, signal2, mode='full')
    
    # Find center (zero lag position)
    center = len(signal2) - 1
    
    # Search within max_lag range
    search_start = max(0, center - max_lag_frames)
    search_end = min(len(correlation), center + max_lag_frames + 1)
    
    # Find peak
    search_range = correlation[search_start:search_end]
    peak_idx = np.argmax(search_range)
    
    # Convert to offset
    offset = peak_idx - (center - search_start)
    
    return offset
