"""Hybrid V2 guitar transcription pipeline.

Combines best-in-class components for each subtask:

  1. ONSET TIMING        — V2 GuitarOnsetCRNN          (val F1 0.814)
  2. PITCH TRANSCRIPTION — Spotify basic-pitch (ONNX)  (polyphonic, MIT)
  3. PITCH → FRET MAP    — rule-based, music-aware     (this module)
  4. SUSTAIN DETECTION   — basic-pitch note durations
  5. (later) HOPO/post-process — chart_postprocess.py

Each subtask uses the best tool. The neural fret head trained in V2 is
abandoned because pitch transcription is a solved problem and rule-based
fret mapping over true pitches outperforms guessing fret-bits from mel.

Output: list[GuitarEvent] compatible with src/inference/guitar_neural.py,
plus exports the same MIDI format.
"""
from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Quiet basic-pitch's missing-coreml/tflite/tf warnings on import
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("root").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.models.guitar_v1 import GuitarOnsetCRNN, OnsetCRNNConfig  # noqa: E402
from src.inference.guitar_neural import GuitarEvent, FRET_TO_MIDI  # noqa: E402
import preprocess_guitar_windows as pgw  # noqa: E402

log = logging.getLogger("guitar_hybrid_v2")


# ─────────────────────────── basic-pitch wrapper ───────────────────────────
_BP_MODEL = None


def _get_basic_pitch_model():
    global _BP_MODEL
    if _BP_MODEL is None:
        from basic_pitch.inference import Model
        from basic_pitch import ICASSP_2022_MODEL_PATH
        _BP_MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _BP_MODEL


@dataclass
class PitchNote:
    t_start: float
    t_end: float
    midi: int
    amplitude: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


def basic_pitch_predict(audio_path: Path, min_amplitude: float = 0.3,
                        min_pitch: int = 36, max_pitch: int = 88) -> list[PitchNote]:
    """Run basic-pitch on an audio file → filtered guitar-range note events.

    Args:
        min_amplitude: drop notes with amplitude below this (suppresses noise)
        min_pitch: MIDI 36 = C2, below standard guitar low E (40)
        max_pitch: MIDI 88 = E6, above 24th-fret high E (88)
    """
    from basic_pitch.inference import predict
    model = _get_basic_pitch_model()
    _, _, raw_events = predict(str(audio_path), model)

    notes: list[PitchNote] = []
    for ev in raw_events:
        t_start, t_end, pitch, amp, _bend = ev
        pitch = int(pitch)
        amp = float(amp)
        if amp < min_amplitude:
            continue
        if pitch < min_pitch or pitch > max_pitch:
            continue
        notes.append(PitchNote(
            t_start=float(t_start), t_end=float(t_end),
            midi=pitch, amplitude=amp,
        ))
    notes.sort(key=lambda n: n.t_start)
    return notes


# ─────────────────────────── Pitch → fret mapper ────────────────────────────
class PitchToFretMapper:
    """Rule-based pitch-set → fret-set mapping using song-wide pitch range.

    Strategy:
      * Compute song's playing range from basic-pitch p5..p95 of starting pitches
      * Bin range into 5 equal-width buckets → fret indices 0..4
      * For each onset's pitch set: bin each pitch, dedupe → fret set
      * Empty set → snap to nearest single pitch (rare fallback)

    Power chord recognition: if pitches form (root, root+7) or (root, root+12),
    map root to its bin and second pitch to root+1 fret (adjacent). This makes
    power chords feel right rather than spread across the neck.
    """

    def __init__(
        self,
        all_pitches: list[int],
        anchor_strength: float = 0.0,
    ):
        if not all_pitches:
            # Default to E2..E5 if no pitches
            self.p_lo, self.p_hi = 40.0, 76.0
        else:
            arr = np.array(all_pitches, dtype=np.float32)
            self.p_lo = float(np.percentile(arr, 5))
            self.p_hi = float(np.percentile(arr, 95))
        self.p_range = max(self.p_hi - self.p_lo, 1.0)
        self.anchor_strength = anchor_strength
        self._last_frets: tuple[int, ...] = ()

    def _pitch_to_fret(self, pitch: int) -> int:
        """Linear bin into 5 frets."""
        norm = (pitch - self.p_lo) / self.p_range
        norm = max(0.0, min(1.0, norm))
        return int(round(norm * 4))

    def map(self, pitches: list[int]) -> tuple[int, ...]:
        if not pitches:
            return self._last_frets if self._last_frets else (0,)

        sorted_p = sorted(set(pitches))

        # ─── Power-chord shortcut ────────────────────────────────────────
        # 2 pitches separated by perfect 5th (7 st) or octave (12 st) →
        # adjacent buttons rooted at the lower pitch's natural bin.
        if len(sorted_p) == 2:
            interval = sorted_p[1] - sorted_p[0]
            if interval in (7, 12):
                root_fret = self._pitch_to_fret(sorted_p[0])
                # adjacent button (cap at orange)
                second = min(4, root_fret + 1)
                if second == root_fret:  # at edge, use prev
                    second = max(0, root_fret - 1)
                self._last_frets = tuple(sorted({root_fret, second}))
                return self._last_frets

        # ─── 3-pitch power chord with octave doubling ────────────────────
        # (root, root+7, root+12) → 2 adjacent buttons (treat as power chord)
        if len(sorted_p) >= 3:
            ints = [sorted_p[i] - sorted_p[0] for i in range(1, len(sorted_p))]
            if 7 in ints and 12 in ints:
                root_fret = self._pitch_to_fret(sorted_p[0])
                second = min(4, root_fret + 1)
                if second == root_fret:
                    second = max(0, root_fret - 1)
                self._last_frets = tuple(sorted({root_fret, second}))
                return self._last_frets

        # ─── General mapping: bin each pitch, dedupe ─────────────────────
        frets = sorted({self._pitch_to_fret(p) for p in sorted_p})

        # If chord collapsed to 1 fret but >1 pitches present, spread across
        # neighbors so it feels like a chord
        if len(frets) == 1 and len(sorted_p) >= 2:
            f0 = frets[0]
            extras = []
            if f0 < 4:
                extras.append(f0 + 1)
            if f0 > 0 and len(sorted_p) >= 3:
                extras.append(f0 - 1)
            frets = sorted(set(frets + extras[: min(len(sorted_p) - 1, 2)]))

        self._last_frets = tuple(frets)
        return self._last_frets


# ─────────────────────────── Snap & merge ───────────────────────────────────
def snap_pitches_to_onsets(
    onset_times: np.ndarray,
    notes: list[PitchNote],
    snap_window_s: float = 0.075,
) -> list[list[PitchNote]]:
    """For each onset, return the list of basic-pitch notes that *start* near it.

    A note matches an onset if |note.t_start - onset_time| <= snap_window_s.
    Notes are not reused — each goes to the nearest onset within window.
    """
    if len(onset_times) == 0:
        return []
    onset_arr = np.asarray(onset_times, dtype=np.float64)
    buckets: list[list[PitchNote]] = [[] for _ in onset_times]

    for n in notes:
        # Find nearest onset
        idx = int(np.searchsorted(onset_arr, n.t_start))
        candidates = []
        if idx < len(onset_arr):
            candidates.append((idx, abs(onset_arr[idx] - n.t_start)))
        if idx > 0:
            candidates.append((idx - 1, abs(onset_arr[idx - 1] - n.t_start)))
        candidates = [(i, d) for i, d in candidates if d <= snap_window_s]
        if not candidates:
            continue
        best_i = min(candidates, key=lambda x: x[1])[0]
        buckets[best_i].append(n)

    return buckets


# ─────────────────────────── Dominant-voice filter ──────────────────────────
def filter_dominant_voice(
    buckets: list[list[PitchNote]],
    all_notes: list[PitchNote],
    lead_percentile: float = 80.0,
    lead_min_fraction: float = 0.08,
    lead_separation_semitones: int = 7,
    lead_amp_ratio: float = 1.2,
    max_lead_notes: int = 2,
) -> list[list[PitchNote]]:
    """Pick the perceptually-dominant voice at each onset.

    Tuned conservatively: only collapses to lead when there's CLEAR evidence
    of a true lead overdub (high pitch, large pitch gap, comparable amplitude
    to rhythm). Otherwise the chord/rhythm bucket is kept intact so chord
    density isn't destroyed.

    Strategy:
      * lead_threshold = song-wide 80th percentile pitch.
      * If <8% of notes are in lead register → no lead voice; pass through.
      * For each onset bucket:
          - lead_notes = bucket notes ≥ lead_threshold
          - rhythm_notes = bucket notes < lead_threshold
          - Only pick lead IF: top_lead ≥ top_rhythm + 7 semitones AND
            top_lead amplitude ≥ lead_amp_ratio × max(rhythm amplitudes).
            This rules out simple chord voicings whose top note happens to
            sit in the lead register.
          - Else pass the full bucket through (preserve chords).
    """
    if not all_notes:
        return buckets
    arr = np.array([n.midi for n in all_notes], dtype=np.float32)
    lead_threshold = float(np.percentile(arr, lead_percentile))
    lead_fraction = float(np.mean(arr >= lead_threshold))
    if lead_fraction < lead_min_fraction:
        return buckets

    out: list[list[PitchNote]] = []
    for bucket in buckets:
        if not bucket:
            out.append(bucket)
            continue
        lead_notes = sorted(
            [n for n in bucket if n.midi >= lead_threshold],
            key=lambda n: -n.midi,
        )
        rhythm_notes = [n for n in bucket if n.midi < lead_threshold]
        if not lead_notes or not rhythm_notes:
            out.append(bucket)
            continue
        top_lead = lead_notes[0]
        top_rhythm_midi = max(n.midi for n in rhythm_notes)
        max_rhythm_amp = max(n.amplitude for n in rhythm_notes)
        # Both gates must pass
        pitch_gap_ok = (top_lead.midi - top_rhythm_midi) >= lead_separation_semitones
        amp_ok = top_lead.amplitude >= lead_amp_ratio * max_rhythm_amp
        if pitch_gap_ok and amp_ok:
            out.append(lead_notes[:max_lead_notes])
        else:
            out.append(bucket)
    return out


# ─────────────────────────── Main charter ───────────────────────────────────
class GuitarHybridV2Charter:
    """V2 onset CRNN + basic-pitch + rule-based pitch→fret mapping."""

    def __init__(
        self,
        onset_ckpt: Path,
        config_path: Path = ROOT / "configs" / "guitar_v2.yaml",
        device: Optional[str] = None,
    ):
        self.cfg = yaml.safe_load(open(config_path))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # V2 onset CRNN
        ocfg = OnsetCRNNConfig(**self.cfg["onset"]["model"])
        self.onset = GuitarOnsetCRNN(ocfg).to(self.device).eval()
        ck = torch.load(onset_ckpt, map_location=self.device, weights_only=False)
        self.onset.load_state_dict(ck["state_dict"])
        self.onset_meta = {"epoch": ck.get("epoch"), "val_f1": ck.get("val_f1")}

        inf = self.cfg["onset"]["inference"]
        self.default_onset_thr = float(inf["peak_threshold"])
        self.default_min_dist_frames = int(inf["peak_min_distance_frames"])

        # Warm up basic-pitch model
        _get_basic_pitch_model()

    # ─── Stage 1: V2 onset CRNN ────────────────────────────────────────────
    @torch.inference_mode()
    def detect_onsets(
        self,
        audio: np.ndarray,
        threshold: float | None = None,
        latency_offset_s: float = 0.025,
    ) -> np.ndarray:
        """Detect onsets and apply attack-latency correction.

        Mel-spectrogram CNNs detect at the energy peak which lags the actual
        attack transient by ~20-30ms (one mel frame at hop=512/22050). We
        subtract `latency_offset_s` so events line up with note attacks in
        playback (matches drum/vocal pipelines which use librosa onset_strength
        + per-attack refinement).
        """
        thr = threshold if threshold is not None else self.default_onset_thr
        log_mel = pgw.compute_log_mel(audio)                   # (M, T)
        x = log_mel.unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,M,T)
        logits = self.onset(x).squeeze(0)
        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        from src.inference.guitar_neural import GuitarNeuralCharter
        peaks = GuitarNeuralCharter.peak_pick(probs, thr, self.default_min_dist_frames)
        frame_s = pgw.HOP_LENGTH / pgw.SAMPLE_RATE
        times = peaks * frame_s - latency_offset_s
        return np.clip(times, 0.0, None)

    # ─── Full pipeline ─────────────────────────────────────────────────────
    def transcribe(
        self,
        audio: np.ndarray,
        audio_path: Path,
        onset_threshold: float | None = None,
        snap_window_s: float = 0.075,
        min_pitch_amplitude: float = 0.3,
        sustain_min_duration_s: float = 0.40,
        max_chord_size: int = 3,
    ) -> list[GuitarEvent]:
        """Full hybrid pipeline.

        Args:
            audio: mono float32 @ 22050 Hz (used for V2 onset model)
            audio_path: path to original audio file (used by basic-pitch)
        """
        # Stage 1: onsets
        onset_times = self.detect_onsets(audio, threshold=onset_threshold)

        # Stage 2: pitches
        notes = basic_pitch_predict(audio_path, min_amplitude=min_pitch_amplitude)

        # Stage 3: snap
        buckets = snap_pitches_to_onsets(onset_times, notes, snap_window_s)

        # Stage 3b: dominant-voice filter — at onsets where a lead overdub
        # sits clearly above the rhythm chord, chart only the lead. Songs
        # with no lead-register activity are passed through unchanged.
        import os as _os
        if _os.environ.get("STRUM_GUITAR_VOICE_FILTER", "1") == "1":
            buckets = filter_dominant_voice(buckets, notes)

        # Stage 4: build mapper from all transcribed pitches
        mapper = PitchToFretMapper([n.midi for n in notes])

        # Stage 5: assemble events
        events: list[GuitarEvent] = []
        for t, bucket in zip(onset_times, buckets):
            pitches = [n.midi for n in bucket]
            frets = mapper.map(pitches)
            if not frets:
                continue
            # Cap chord size: sections with multiple guitars cause basic-pitch
            # to detect 4-5 simultaneous pitches, leading to over-charted full
            # chords. Real CH/YARG charts rarely exceed 3-button chords.
            # Keep the frets corresponding to the LOUDEST pitches.
            if len(frets) > max_chord_size and bucket:
                # Sort bucket by amplitude desc and take fret bins for top-N
                sorted_bucket = sorted(bucket, key=lambda n: -n.amplitude)
                kept_pitches = [n.midi for n in sorted_bucket[:max_chord_size + 1]]
                trimmed = mapper.map(kept_pitches)
                if trimmed:
                    frets = trimmed[:max_chord_size]
                else:
                    frets = frets[:max_chord_size]
            # Confidence proxy: mean amplitude or 0.5 if empty
            amp = float(np.mean([n.amplitude for n in bucket])) if bucket else 0.5
            # Sustain check (longest constituent note)
            max_dur = max((n.duration for n in bucket), default=0.0)
            events.append(GuitarEvent(
                time_sec=float(t),
                frets=frets,
                onset_prob=amp,
                fret_probs=tuple([0.0] * 5),  # not used downstream
            ))
            # stash sustain duration on the event for export
            events[-1].__dict__["sustain_duration_s"] = (
                max_dur if max_dur >= sustain_min_duration_s else 0.0
            )
        return events


# ─────────────────────────── MIDI export with sustains ──────────────────────
def export_events_to_midi_with_sustain(
    events: list[GuitarEvent],
    output_path: Path,
    tempo_bpm: float = 120.0,
    short_note_sec: float = 0.05,
    track_name: str = "PART GUITAR",
) -> None:
    """Like export_events_to_midi but uses per-event sustain_duration_s if present."""
    import mido

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name))
    tempo_us = int(60_000_000 / tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us))

    tpb = mid.ticks_per_beat
    sec_to_tick = lambda s: int(round(s * tempo_bpm / 60.0 * tpb))

    msgs: list[tuple[int, bool, int]] = []
    for ev in events:
        sustain = float(getattr(ev, "sustain_duration_s", 0.0) or 0.0)
        dur = sustain if sustain > 0 else short_note_sec
        on_t = sec_to_tick(ev.time_sec)
        off_t = sec_to_tick(ev.time_sec + dur)
        for f in ev.frets:
            if f not in FRET_TO_MIDI:
                continue
            note = FRET_TO_MIDI[f]
            msgs.append((on_t, True, note))
            msgs.append((off_t, False, note))
    msgs.sort(key=lambda x: (x[0], x[1]))   # off before on at same tick

    last_t = 0
    for tick, is_on, note in msgs:
        delta = max(0, tick - last_t)
        track.append(mido.Message(
            "note_on" if is_on else "note_off",
            note=note, velocity=100 if is_on else 0, time=delta,
        ))
        last_t = tick

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(output_path))


# ─────────────────────────── batch_pipeline adapter ─────────────────────────
_HYBRID_CHARTER_CACHE: dict[str, "GuitarHybridV2Charter"] = {}
_DEFAULT_HYBRID_ONSET_CKPT = ROOT / "checkpoints" / "guitar_v2" / "guitar_v2_onset" / "best.pt"


def _get_hybrid_charter(device: str | None = None) -> "GuitarHybridV2Charter":
    key = device or "auto"
    ch = _HYBRID_CHARTER_CACHE.get(key)
    if ch is None:
        ch = GuitarHybridV2Charter(onset_ckpt=_DEFAULT_HYBRID_ONSET_CKPT, device=device)
        _HYBRID_CHARTER_CACHE[key] = ch
    return ch


def transcribe_guitar_hybrid(
    audio_path: "Path | str",
    tempo_bpm: float = 0.0,
    is_bass: bool = False,
    onset_threshold: float | None = None,
    device: str | None = None,
):
    """Hybrid V2 (V2 onset CRNN + basic-pitch + rule pitch→fret) → GuitarChart.

    Drop-in replacement for transcribe_guitar_neural; same return type.
    """
    import librosa as _lr
    from src.inference.guitar_bass import GuitarChart, GuitarNote, GuitarChord

    audio_path = Path(audio_path)
    audio, sr = _lr.load(str(audio_path), sr=22050, mono=True)
    if tempo_bpm <= 0:
        try:
            tempo_fn = getattr(_lr.beat, "tempo", None) or _lr.feature.rhythm.tempo
            tempo_bpm = float(np.atleast_1d(tempo_fn(y=audio, sr=sr))[0])
        except Exception:
            tempo_bpm = 120.0

    ch = _get_hybrid_charter(device=device)
    events = ch.transcribe(audio, audio_path, onset_threshold=onset_threshold)

    chart = GuitarChart(
        tempo_bpm=float(tempo_bpm),
        instrument="bass" if is_bass else "guitar",
    )
    for ev in events:
        t_ms = ev.time_sec * 1000.0
        sustain_s = float(getattr(ev, "__dict__", {}).get("sustain_duration_s", 0.0) or 0.0)
        dur_ms = sustain_s * 1000.0 if sustain_s > 0 else 100.0
        if len(ev.frets) >= 2:
            chart.chords.append(GuitarChord(
                time_ms=t_ms, frets=list(ev.frets), duration_ms=dur_ms,
            ))
        elif len(ev.frets) == 1:
            chart.notes.append(GuitarNote(
                time_ms=t_ms, fret=int(ev.frets[0]), duration_ms=dur_ms,
            ))
    return chart


__all__ = [
    "GuitarHybridV2Charter",
    "PitchToFretMapper",
    "PitchNote",
    "basic_pitch_predict",
    "snap_pitches_to_onsets",
    "export_events_to_midi_with_sustain",
    "transcribe_guitar_hybrid",
]
