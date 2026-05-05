# STRUM Architecture

This document describes the **as-built** STRUM pipeline. For high-level usage
see [README.md](../README.md); for status & next steps see
[ROADMAP.md](ROADMAP.md).

## 1. System Overview

STRUM converts a single audio file (`song.ogg/.wav/.mp3`) into a Clone Hero /
YARG chart package containing PART DRUMS, PART GUITAR, PART BASS, PART VOCALS
(with lyrics), and PART KEYS — all aligned to a common tempo grid.

```
                           audio (44.1 kHz)
                                 │
                       ┌─────────▼──────────┐
                       │  Demucs htdemucs   │
                       │  6-stem separation │
                       └─────────┬──────────┘
                                 │
       ┌────────────┬────────────┼────────────┬────────────┐
       │            │            │            │            │
   drums.wav    guitar.wav    bass.wav    vocals.wav    other.wav
       │            │            │            │            │
       ▼            ▼            ▼            ▼            ▼
   Two-stage     Hybrid       Hybrid       Whisper      Spectral
   CRNN +        onset +      onset +      + pYIN +     keyboard
   ensemble      Basic Pitch  Basic Pitch  alignment    detector
   classifier    + fret map   + fret map   + LRCLIB     + piptrack
       │            │            │            │            │
       └────────────┴────────┬───┴────────────┴────────────┘
                             │
                ┌────────────▼─────────────┐
                │  BPM grid alignment      │
                │  (±5 BPM @ 0.1 res +     │
                │  beat-zero phase snap)   │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │  Phase shift + 32nd-note │
                │  snap (per-lane roll     │
                │  detection)              │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │  Difficulty reduction    │
                │  (Expert/Hard/Med/Easy)  │
                └────────────┬─────────────┘
                             │
                ┌────────────▼─────────────┐
                │  notes.mid + song.ini    │
                │  + album art + audio     │
                └──────────────────────────┘
```

## 2. Drums Pipeline

The flagship pipeline — a two-stage neural system with audio-coupled rescue
passes and ensemble voting. Implemented in `scripts/batch_infer_hybrid.py` and
called from `batch_pipeline.py`.

### 2.1 Stage 1 — Onset Detection (`TwoStageDrumsCRNN`, V14)

Mel spectrograms (128 mel bins, 22050 Hz, hop 512) → CNN → BiLSTM → 1-D
onset probability per frame. Detected onsets become **lane-agnostic** drum hits
with timestamps. Trained on ~5k human pro-drum charts; verified F1 = 93.9%.

Architecture: `src/models/drums_v13.py::TwoStageDrumsCRNN` (V14 reuses the V13
class with a stronger checkpoint).

### 2.2 Stage 2 — Lane Classification (Ensemble)

Each detected onset is classified across 8 lanes (Kick, Snare, Hi-Hat, Crash,
Ride, High Tom, Mid Tom, Floor Tom) by an ensemble of independently-trained
`OnsetClassifier` models with `PER_CLASS_WEIGHTS`:

| Member | Config | Notes |
|--------|--------|-------|
| V2  | `onset_classifier.yaml`            | Original baseline, full mix |
| V6  | `onset_classifier_v6.yaml`         | Stronger snare/hi-hat head |
| V12c| `onset_classifier_v12_clean.yaml`  | Trained on clean Demucs stems |
| V15 | `onset_classifier_v15.yaml`        | Bg-mel subtraction |
| V16 | `onset_classifier_v16.yaml`        | Cymbal-aware loss |

Voting uses class-weighted soft-max averaging followed by argmax. Verified
ensemble F1 = 85.2%; best single member (V12c) = 83.8%.

### 2.3 Rescue & Refinement Passes

After classification, six audio-coupled passes index back into the mel-frame
probability tensor `mc_frame_probs` to fix systematic errors. **Critically, all
six must run on the original time base — before any phase shift is applied**
(see §6).

| Pass | Purpose |
|------|---------|
| `phase3_onset_rescue` | Recover missed quiet onsets via mel-frame peak picking |
| `phase3_cymbal_cooccurrence_rescue` | Add missing cymbals when kick+cymbal frame coincides |
| `apply_cymbal_to_tom_rescue` | Convert false-positive cymbals to toms by spectral centroid |
| `apply_drumsep_hits_arbiter` | Cross-check against Demucs drum-separation stems |
| `spectral_reclassify` | Resolve tom-vs-cymbal confusion by harmonic ratio |
| `apply_tom_refinement_filter` | Tom-refinement CNN (`src/models/tom_refinement.py`) |

### 2.4 Post-Processing (`scripts/chart_postprocess.py`)

* Bidirectional iterative streak smoothing (collapses isolated mis-classifications
  inside a uniform streak)
* `kick_suppresses_floor_tom` (kick + floor tom on same frame → drop the tom)
* `enforce_single_tom_per_onset` (one tom marker per onset, max 110/111/112)
* `cap_close_hands` (no two cymbals < 60 ms apart unless tom-roll context)
* `protect_tom_fills` (preserve dense tom rolls from over-aggressive snap)
* Lane conflict resolution + velocity normalization

## 3. Guitar & Bass Pipeline

Both guitar and bass share the same hybrid architecture
(`src/inference/guitar_hybrid_v2.py`). Backend selection via env vars:

| Backend | Onset source | Pitch source | Use case |
|---------|--------------|--------------|----------|
| `hybrid` *(default)* | V2 onset CRNN on Demucs stem | Spotify Basic Pitch | Best balance |
| `neural` | V2 onset CRNN on full mix | V2 fret head | Pure-neural baseline |
| `basicpitch` | Basic Pitch onsets | Basic Pitch | Fallback for sparse onset stems |
| `rule` | librosa onset_detect | pYIN | Legacy, no neural component |

### 3.1 Onset Detection (V1/V2 Guitar CRNN)

`src/models/guitar_v1.py::OnsetCRNN` — single-output onset head trained on
isolated `guitar.ogg` stems from ~5k charts. V2 adds a deeper backbone and
section-aware peak-thresholding. Configs: `guitar_v1.yaml`, `guitar_v2.yaml`.

### 3.2 Polyphonic Pitch (Basic Pitch)

[Spotify Basic Pitch](https://github.com/spotify/basic-pitch) transcribes each
detected onset window into 0–N MIDI pitches (chord support). For bass we
override the model's MIDI range to 24–67 via `STRUM_BP_MIN_PITCH` /
`STRUM_BP_MAX_PITCH`, and use a softer onset peak threshold
(`STRUM_GUITAR_PEAK_THR=0.35`) because bass attacks are weaker than guitar.

### 3.3 Pitch → Fret Mapping

Default: **rule-based register allocation**. Pitches are bucketed into the
five Clone Hero frets by absolute MIDI value, with chord-shape preservation.
HOPO threshold is 170 ms.

Optional: **`PitchToFretMapper` (V4)** — a learned mapper trained on ~5k chart
pitch→fret pairs (`scripts/build_mapper_dataset.py` +
`scripts/train_fret_mapper.py`). Enabled with `STRUM_FRET_MAPPER=1`.

### 3.4 Section Router

`src/inference/section_router.py` predicts per-1-second section labels
(verse/chorus/solo/etc.) using `src/models/section_classifier.py` and modulates
the onset peak threshold per section, preventing over-charting in quiet verses
and under-charting in dense choruses. Optional, gated by checkpoint presence.

## 4. Vocals Pipeline (`scripts/vocals_charter.py`)

1. **Lyric transcription** — OpenAI Whisper extracts word-level timestamps
   from the Demucs vocal stem.
2. **Pitch contour** — `librosa.pyin` tracks vocal F0 at 10 ms resolution.
3. **Word ↔ pitch alignment** — Whisper word boundaries are warped against
   pitch onsets via a dynamic-programming alignment so each word lands on the
   pitch attack instead of the model's centroid estimate.
4. **Lyrics fetching** — Optional synced lyrics from
   [LRCLIB](https://lrclib.net/) and [Lyrics.ovh](https://lyrics.ovh/).
5. **Harmony detection** — Configurable presence threshold (default 30%); a
   pitch line is added to PART HARM2/HARM3 only if it appears in ≥30% of
   vocal sections.

Output dataclasses (`VocalNote`, `VocalPhrase`) use **seconds** for timestamps,
not ms — see §6.5.

## 5. Keys Pipeline (`scripts/keys_charter.py`)

1. **Keyboard detection** — Spectral flatness + harmonic ratio analysis on
   the Demucs `other` stem identifies regions where a keyboard is active.
2. **Onset detection** — `librosa.onset_detect` over the keyboard-active
   regions only.
3. **Pitch extraction** — `librosa.piptrack` per onset window, then Basic
   Pitch refinement when `STRUM_KEYS_BACKEND=basicpitch`.
4. **Dual output** — Both 5-lane simplified PART KEYS and full-range PART
   REAL_KEYS (Pro Keys) are written.

## 6. Tempo Detection & Cross-Instrument Grid Alignment

The single most important system-wide invariant.

### 6.1 BPM Refinement

Initial BPM from `librosa.beat.beat_track`, then a ±5 BPM grid search at
0.1 BPM resolution. For each candidate BPM we measure phase coherence using
**circular statistics** on `(onset_time mod beat_period)`, picking the BPM
with maximum unit-vector magnitude. Reduces grid error from ~175 ms to
~35 ms on typical tracks.

### 6.2 Phase Offset

Once the BPM is locked, `phase_offset_ms` is the time of the **first
detected onset** modulo the beat period. This is more robust than the
circular mean of all onsets, which drifts when the grid-aligned BPM differs
from the librosa estimate.

### 6.3 The Critical Ordering

Every transcriber emits events on the **raw audio time base** so that the six
drum rescue passes (§2.3) can index `mc_frame_probs[time_ms * sr / hop]`
correctly. The phase shift and grid snap happen **after** all rescue passes
have run, in `transcribe_drums()` and the cross-instrument loop in
`batch_pipeline.py`.

```python
for chart in (drums, guitar.notes, guitar.chords,
              bass.notes, bass.chords,
              keys, vocal_phrases, vocal_notes):
    for ev in chart:
        ev.time += phase_offset           # ms or seconds per dataclass
        ev.time = snap_to_grid(ev.time, grid_32nd, roll_window=grid_ms*1.1)
```

### 6.4 Snap-to-Grid with Roll Detection

A naive snap collapses fast double-strokes and tom rolls. The snap function
checks for a same-(lane, is_cymbal) neighbor within
`_roll_window_ms = grid_ms * 1.1` and skips the snap if one exists, preserving
rolls.

### 6.5 Time-Base Inventory

Different transcribers use different time units. The cross-instrument loop
respects this:

| Dataclass | Time field | Unit |
|-----------|------------|------|
| `DrumHit` | `time_ms` | milliseconds |
| `GuitarNote`, `GuitarChord` | `time_ms` | milliseconds |
| `KeysNote` | `time_ms` | milliseconds |
| `VocalNote`, `VocalPhrase` | `start_time`, `end_time` | **seconds** |

## 7. Chart Export

### 7.1 MIDI (`src/export/midi.py`)

Standard MIDI File Type 1, 480 ticks per quarter note. Tracks:

* Track 0 — Tempo map + time signatures
* Track 1 — Section markers
* `PART DRUMS` — Pro drums (lanes 96–100, tom markers 110–112)
* `PART GUITAR` / `PART BASS` — 5-fret (96–100), HOPO/tap modifiers
* `PART VOCALS` — Pitched phrases + lyric meta-events
* `PART KEYS` / `PART REAL_KEYS_X` — 5-lane + Pro Keys

### 7.2 Difficulty Generation (`scripts/chart_enhancer.py`)

| Difficulty | Notes/sec cap | Max chord size | Notes |
|------------|---------------|----------------|-------|
| Expert | 12 | 4 | Full chart |
| Hard   | 9  | 3 | Drop ghost notes |
| Medium | 6  | 2 | Simplify rolls |
| Easy   | 4  | 1 | Downbeats only |

### 7.3 song.ini

Generated with title, artist, charter (`STRUM`), genre, year, BPM, and
per-instrument difficulty ratings. Cover art is fetched from the audio file's
embedded metadata or via the iTunes search API.

## 8. Module Map

```
src/
├── models/
│   ├── drums_v13.py              # TwoStageDrumsCRNN (V14 ckpt)
│   ├── drums_v14_dataset.py      # Drum onset dataset (bg-mel subtraction)
│   ├── onset_classifier.py       # 8-lane drum classifier (ensemble member)
│   ├── onset_classifier_dataset.py, *_cached_dataset.py
│   ├── tom_refinement.py         # Tom-vs-cymbal CNN
│   ├── guitar_v1.py              # Guitar onset CRNN (V1/V2)
│   ├── section_classifier.py     # Per-1s section labeler
│   ├── bg_mel.py                 # Background-mel subtraction utilities
│   └── common.py
├── inference/
│   ├── guitar_hybrid_v2.py       # ★ Production guitar/bass backend
│   ├── guitar_neural.py          # Neural-only backend
│   ├── guitar_bass.py            # Dataclasses + rule backend
│   ├── section_router.py         # Section-aware onset gating
│   └── c3_rules.py               # Clone Hero charting conventions
├── preprocessing/
│   ├── parsers/                  # .mid + .chart parsers
│   ├── alignment.py              # Audio-chart alignment (training)
│   └── separation.py             # Demucs wrapper
├── export/
│   ├── midi.py                   # MIDI writer (all parts)
│   └── chart.py                  # .chart format writer
└── lyrics/                       # LRCLIB + Lyrics.ovh fetcher
```

## 9. Training Inventory

| Model | Trainer | Preprocess | Config |
|-------|---------|------------|--------|
| Drum onset CRNN (V14) | `train_onset_classifier.py` | `preprocess_onset_windows.py` | `drums_v14.yaml` |
| Drum classifier ensemble | `train_onset_classifier.py` | `preprocess_onset_windows.py` | `onset_classifier_*.yaml` |
| Tom-refinement CNN | `train_tom_refinement.py` | (uses Demucs drum stem) | inline |
| Guitar onset CRNN (V1/V2) | `train_guitar_v1.py` | `build_guitar_manifest.py` → `preprocess_guitar_windows.py` | `guitar_v1.yaml`, `guitar_v2.yaml` |
| Pitch→fret mapper (V4) | `train_fret_mapper.py` | `build_mapper_dataset.py` | inline |
| Section classifier | `train_section_classifier.py` | `build_section_labels.py` → `preprocess_section_windows.py` | inline |

All trainers log to W&B (`WANDB_MODE=offline` to disable). Checkpoints land in
`checkpoints/<model>/` and are loaded by name by the production scripts.

## 10. Hardware

Developed on NVIDIA DGX Spark (GB10 GPU, CUDA 12.8). Inference runs in
~real-time on a single 12 GB GPU; training one drum classifier takes
~6–8 hours on the same hardware.
