# STRUM Architecture

## System Overview

STRUM is a modular pipeline for converting audio files into game-compatible charts for Clone Hero and YARG. The system prioritizes accuracy for drums, vocals, and pro keys (which map 1:1 with real performance), while using intelligent reduction for guitar and bass.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STRUM PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │  Audio   │───▶│ Preprocessing │───▶│   Demucs    │───▶│  Instrument   │  │
│  │  Input   │    │  (normalize)  │    │ Separation  │    │   Pipelines   │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────────────┘  │
│                                              │                    │         │
│                          ┌───────────────────┼────────────────────┤         │
│                          ▼                   ▼                    ▼         │
│                    ┌──────────┐        ┌──────────┐        ┌──────────┐    │
│                    │  Drums   │        │  Bass    │        │  Vocals  │    │
│                    │  Stem    │        │  Stem    │        │  Stem    │    │
│                    └────┬─────┘        └────┬─────┘        └────┬─────┘    │
│                         │                   │                    │         │
│                         ▼                   ▼                    ▼         │
│                    ┌──────────┐        ┌──────────┐        ┌──────────┐    │
│                    │  CRNN    │        │  Basic   │        │  Basic   │    │
│                    │  Model   │        │  Pitch   │        │  Pitch   │    │
│                    │(trained) │        │ + Rules  │        │+ Harmony │    │
│                    └────┬─────┘        └────┬─────┘        └────┬─────┘    │
│                         │                   │                    │         │
│                         └───────────────────┼────────────────────┘         │
│                                             ▼                               │
│                                    ┌─────────────────┐                     │
│                                    │  Chart Export   │                     │
│                                    │ (.mid / .chart) │                     │
│                                    └─────────────────┘                     │
│                                             │                               │
│                                             ▼                               │
│                              ┌──────────────────────────┐                  │
│                              │   Difficulty Generation  │                  │
│                              │ (Expert/Hard/Medium/Easy)│                  │
│                              └──────────────────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Module Specifications

### 1. Preprocessing (`src/preprocessing/`)

#### Audio Normalization
- **Sample Rate**: Resample all audio to 44.1 kHz
- **Channels**: Convert to mono for processing (preserve stereo for separation)
- **Loudness**: Normalize to -14 LUFS for consistent model input
- **Format**: Support WAV, MP3, FLAC, OGG input

#### Demucs Separation (`separation.py`)
- **Model**: `htdemucs` (Hybrid Transformer Demucs v4)
- **Stems**: drums, bass, vocals, other
- **Output**: Separated stems as 44.1kHz WAV files
- **GPU**: Utilize CUDA when available (falls back to CPU)

#### Chart Parsing (`parsers/`)

**MIDI Parser** (`.mid` files):
- Parse Standard MIDI File (SMF) Type 0 or 1
- Extract tempo map (microseconds per quarter note)
- Extract time signature events
- Extract note events with timing, pitch, velocity

**Chart Parser** (`.chart` files):
- Parse Clone Hero .chart format
- Extract [Song] metadata section
- Extract [SyncTrack] for tempo/time signatures
- Extract instrument tracks ([ExpertDrums], etc.)

#### Audio-Chart Alignment (`alignment.py`)
- Compute cross-correlation between chart-derived click track and audio
- Detect and correct systematic offset (typical range: -50ms to +50ms)
- Output alignment offset in milliseconds

### 2. Pro Drums Model (`src/models/drums.py`)

#### Architecture: CRNN (Convolutional Recurrent Neural Network)

```
Input: Mel-spectrogram (n_mels=128, hop_length=512, sr=44100)
       Shape: (batch, 1, time_frames, 128)
       
       ┌─────────────────────────────────────────┐
       │         Conv2D Block 1                  │
       │  Conv2D(1, 32, 3x3) → BN → ReLU → Pool  │
       └─────────────────────────────────────────┘
                          │
       ┌─────────────────────────────────────────┐
       │         Conv2D Block 2                  │
       │  Conv2D(32, 64, 3x3) → BN → ReLU → Pool │
       └─────────────────────────────────────────┘
                          │
       ┌─────────────────────────────────────────┐
       │         Conv2D Block 3                  │
       │  Conv2D(64, 128, 3x3) → BN → ReLU → Pool│
       └─────────────────────────────────────────┘
                          │
       ┌─────────────────────────────────────────┐
       │         Reshape + BiLSTM                │
       │  BiLSTM(hidden=256, layers=2, dropout)  │
       └─────────────────────────────────────────┘
                          │
       ┌─────────────────────────────────────────┐
       │         Output Heads (per frame)        │
       │  - 5x Onset logits (kick/R/Y/B/G)       │
       │  - 3x Cymbal flags (Y/B/G)              │
       │  - 5x Velocity (optional regression)    │
       └─────────────────────────────────────────┘

Output Shape: (batch, time_frames, 13)
  - [:, :, 0:5]  = onset probabilities per lane
  - [:, :, 5:8]  = cymbal flags (yellow, blue, green)
  - [:, :, 8:13] = velocity values (0-127 normalized)
```

#### Training Configuration
```yaml
# configs/drums.yaml
model:
  n_mels: 128
  hop_length: 512
  conv_channels: [32, 64, 128]
  lstm_hidden: 256
  lstm_layers: 2
  dropout: 0.3

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  checkpoint_every: 5
  early_stopping_patience: 15

loss:
  onset_weight: 1.0
  cymbal_weight: 0.5
  velocity_weight: 0.3
  pos_weight: 3.0  # Handle class imbalance

augmentation:
  time_stretch: [0.9, 1.1]
  pitch_shift: [-2, 2]  # semitones
  noise_level: 0.01
```

### 3. Clone Hero / YARG Pro Drums MIDI Mapping

#### Standard Pro Drums Note Numbers

| Lane | Note Number | Description | Cymbal Modifier |
|------|-------------|-------------|-----------------|
| Kick | 95 | Bass drum | N/A |
| Red (Snare) | 96 | Snare drum | N/A |
| Yellow | 97 | Hi-hat (cymbal) / High tom | +110 for cymbal |
| Blue | 98 | Ride cymbal / Mid tom | +111 for cymbal |
| Green | 99 | Crash cymbal / Floor tom | +112 for cymbal |

#### Cymbal Markers
- Yellow cymbal: Note 110 simultaneous with note 97
- Blue cymbal: Note 111 simultaneous with note 98  
- Green cymbal: Note 112 simultaneous with note 99

#### Velocity Mapping
- Clone Hero uses velocity 1-127
- Map model velocity output [0, 1] → [1, 127]
- Ghost notes: velocity < 50
- Accent notes: velocity > 100

#### Special Notes
| Note | Description |
|------|-------------|
| 116 | Star Power / Overdrive phrase start |
| 103 | Solo marker start |
| 104 | Solo marker end |
| 105 | Player 1 phrase (for vocals) |
| 106 | Player 2 phrase (for vocals) |

### 4. Guitar/Bass Rule Engine

#### Pitch-to-Fret Mapping (Bass)
```python
# 4-string bass standard tuning: E1-G4 (41-67 MIDI)
BASS_FRET_RANGES = {
    'open':  (0, 44),    # E1 to G#1 → Open/Green
    'fret1': (45, 49),   # A1 to C#2 → Fret 1/Red
    'fret2': (50, 54),   # D2 to F#2 → Fret 2/Yellow
    'fret3': (55, 59),   # G2 to B2  → Fret 3/Blue
    'fret4': (60, 127),  # C3+       → Fret 4/Orange
}
```

#### Pitch-to-Fret Mapping (Guitar)
```python
# 6-string guitar standard tuning: E2-E6 (40-88 MIDI)
GUITAR_FRET_RANGES = {
    'open':  (0, 47),    # E2 to B2  → Open/Green
    'fret1': (48, 54),   # C3 to F#3 → Fret 1/Red
    'fret2': (55, 61),   # G3 to C#4 → Fret 2/Yellow
    'fret3': (62, 68),   # D4 to G#4 → Fret 3/Blue
    'fret4': (69, 127),  # A4+       → Fret 4/Orange
}
```

#### HOPO (Hammer-On/Pull-Off) Detection
```python
HOPO_THRESHOLD_MS = 170  # Notes within 170ms are HOPO candidates

def is_hopo(prev_note, curr_note):
    """Determine if current note should be a HOPO."""
    time_delta = curr_note.time - prev_note.time
    pitch_changed = curr_note.fret != prev_note.fret
    
    return time_delta <= HOPO_THRESHOLD_MS and pitch_changed
```

#### Chord Simplification
```python
def simplify_chord(notes, max_notes=2):
    """Reduce chord to root + highest note."""
    if len(notes) <= max_notes:
        return notes
    
    sorted_notes = sorted(notes, key=lambda n: n.pitch)
    return [sorted_notes[0], sorted_notes[-1]]  # Root + highest
```

### 5. Vocals Pipeline

#### Lead Melody Extraction
- Input: Demucs vocal stem
- Process: Basic Pitch → pitch contour with confidence
- Output: Monophonic melody line (MIDI note events)

#### Harmony Detection
```python
def detect_harmonies(pitch_contours, presence_threshold=0.30):
    """
    Detect harmony lines from multi-pitch output.
    
    Args:
        pitch_contours: List of detected pitch lines with timestamps
        presence_threshold: Minimum % of song for harmony inclusion
        
    Returns:
        lead: Primary melody line
        harmonies: List of harmony lines meeting threshold
    """
    # Cluster simultaneous pitches
    # Identify lead (loudest/most consistent)
    # Calculate presence ratio for each harmony
    # Return harmonies exceeding threshold
```

#### Harmony Presence Calculation
```python
def calculate_presence_ratio(harmony_line, vocal_sections):
    """
    Calculate what % of vocal sections contain this harmony.
    
    Default threshold: 30% (configurable via --harmony-threshold)
    """
    harmony_duration = sum(note.duration for note in harmony_line)
    vocal_duration = sum(section.duration for section in vocal_sections)
    
    return harmony_duration / vocal_duration if vocal_duration > 0 else 0
```

### 6. Chart Export (`src/export/`)

#### MIDI Export Format
- **Type**: Standard MIDI File Type 1 (multiple tracks)
- **Resolution**: 480 ticks per quarter note
- **Tracks**:
  - Track 0: Tempo map + time signatures
  - Track 1: Song metadata (markers, sections)
  - Track 2+: Instrument tracks

#### .chart Export Format
```
[Song]
{
  Name = "Song Title"
  Artist = "Artist Name"
  Charter = "STRUM"
  Resolution = 192
}

[SyncTrack]
{
  0 = TS 4
  0 = B 120000
}

[ExpertDrums]
{
  0 = N 0 0
  480 = N 1 0
  ...
}
```

#### Difficulty Generation
| Difficulty | Max Notes/Sec | Max Chord Size | Special Rules |
|------------|---------------|----------------|---------------|
| Expert | 12 | 4 | Full chart |
| Hard | 9 | 3 | Remove ghost notes |
| Medium | 6 | 2 | Simplify rolls |
| Easy | 4 | 1 | Downbeats only |

### 7. Dataset Format

#### Manifest Schema (`manifest.json`)
```json
{
  "version": "1.0",
  "created": "2025-01-05T00:00:00Z",
  "split": {
    "train": 0.85,
    "test": 0.15,
    "seed": 42
  },
  "songs": [
    {
      "id": "abc123",
      "audio_path": "raw/song.mp3",
      "stems": {
        "drums": "processed/abc123/drums.wav",
        "bass": "processed/abc123/bass.wav",
        "vocals": "processed/abc123/vocals.wav",
        "other": "processed/abc123/other.wav"
      },
      "charts": {
        "drums": "raw/song/notes.mid",
        "guitar": "raw/song/notes.mid",
        "bass": null,
        "vocals": null
      },
      "split": "train",
      "alignment_offset_ms": -12.5
    }
  ],
  "coverage": {
    "drums": 4523,
    "guitar": 4891,
    "bass": 3102,
    "vocals": 892,
    "keys": 234
  }
}
```

#### DrumHit Data Structure
```python
@dataclass
class DrumHit:
    time_ms: float        # Onset time in milliseconds
    lane: int             # 0=kick, 1=red, 2=yellow, 3=blue, 4=green
    is_cymbal: bool       # True if cymbal marker present
    velocity: int         # 1-127 MIDI velocity
    
@dataclass  
class DrumChart:
    hits: List[DrumHit]
    tempo_map: List[TempoEvent]
    time_signatures: List[TimeSignature]
```

### 8. Evaluation Metrics

#### Per-Lane Metrics
- **Onset F1**: Tolerance window ±50ms
- **Lane Accuracy**: Correct lane assignment given correct onset
- **Cymbal F1**: Cymbal flag accuracy for yellow/blue/green

#### Aggregate Metrics
- **Overall F1**: Weighted average across lanes
- **Velocity MAE**: Mean absolute error for velocity prediction
- **Timing Error**: Mean onset timing deviation in ms

#### W&B Logging
```python
wandb.log({
    "train/loss": total_loss,
    "train/onset_loss": onset_loss,
    "train/cymbal_loss": cymbal_loss,
    "val/f1_kick": f1_kick,
    "val/f1_snare": f1_snare,
    "val/f1_overall": f1_overall,
    "val/timing_error_ms": timing_error,
})
```

## API Contracts

### Preprocessing Pipeline
```python
def preprocess(
    input_dir: Path,
    output_dir: Path,
    instruments: List[str] = ["drums"],
    split_ratio: float = 0.85,
    seed: int = 42,
) -> Manifest:
    """
    Preprocess raw audio + charts into training-ready dataset.
    
    Returns:
        Manifest with paths and metadata for all processed songs.
    """
```

### Training Pipeline
```python
def train(
    config_path: Path,
    manifest_path: Path,
    checkpoint_path: Optional[Path] = None,
) -> None:
    """
    Train drums model with W&B logging.
    
    Saves checkpoints to config.output_dir every N epochs.
    """
```

### Inference Pipeline
```python
def infer(
    audio_path: Path,
    model_path: Path,
    output_path: Path,
    harmony_threshold: float = 0.30,
) -> None:
    """
    Generate chart from audio file.
    
    Outputs .mid file with all detected instruments.
    """
```
