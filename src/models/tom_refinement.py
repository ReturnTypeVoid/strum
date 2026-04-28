"""
Tom Refinement Model — classifies candidate frames as {not-tom, HighTom, LowTom, FloorTom}.

Takes ~500ms context windows (mel + CQT features) centered on MC-head tom candidates.
Designed to filter the MC head's high-recall/low-precision tom detections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset

import torchaudio
import soundfile as sf

logger = logging.getLogger(__name__)

# Tom classes from the 8-class drum mapping
# Class 3 = HighTom (lane 2, not cymbal)
# Class 5 = LowTom (lane 3, not cymbal)
# Class 7 = FloorTom (lane 4, not cymbal)
TOM_CLASSES = [3, 5, 7]  # indices into the 8-class system
TOM_CLASS_NAMES = ['NotTom', 'HighTom', 'LowTom', 'FloorTom']

LANE_CYMBAL_TO_CLASS = {
    (0, False): 0, (0, True): 0,    # Kick
    (1, False): 1, (1, True): 1,    # Snare
    (2, True): 2,                     # HiHat (cymbal)
    (2, False): 3,                    # HighTom
    (3, True): 4,                     # Ride (cymbal)
    (3, False): 5,                    # LowTom
    (4, True): 6,                     # Crash (cymbal)
    (4, False): 7,                    # FloorTom
}


class TomRefinementCNN(nn.Module):
    """
    Small CNN classifier for tom refinement.

    Input: (B, 2, n_freq, context_frames) — 2 channels: mel + CQT
    Output: (B, 4) logits — [not_tom, high_tom, low_tom, floor_tom]
    """

    def __init__(
        self,
        n_mel: int = 128,
        n_cqt: int = 84,
        context_frames: int = 43,  # ~500ms at 86 fps
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_mel = n_mel
        self.n_cqt = n_cqt
        self.context_frames = context_frames

        # Mel branch
        self.mel_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 2)),  # (32, 32, 21)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4, 2)),  # (64, 8, 10)

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # (128, 4, 4)
        )

        # CQT branch — lower frequency resolution, good for pitch
        self.cqt_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 2)),  # (32, 21, 21)

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4, 2)),  # (64, 5, 10)

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # (128, 4, 4)
        )

        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # not_tom, high, low, floor
        )

    def forward(self, mel: torch.Tensor, cqt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, 1, n_mel, context_frames) log-mel spectrogram windows
            cqt: (B, 1, n_cqt, context_frames) log-CQT spectrogram windows
        Returns:
            logits: (B, 4) class logits
        """
        mel_feat = self.mel_conv(mel)       # (B, 128, 4, 4)
        cqt_feat = self.cqt_conv(cqt)       # (B, 128, 4, 4)

        mel_flat = mel_feat.flatten(1)      # (B, 2048)
        cqt_flat = cqt_feat.flatten(1)      # (B, 2048)

        combined = torch.cat([mel_flat, cqt_flat], dim=1)  # (B, 4096)
        return self.classifier(combined)


class TomRefinementDataset(Dataset):
    """
    Dataset for tom refinement training.

    Extracts context windows around tom candidates from the full
    drum dataset. Balances positive (real toms) and negative
    (non-tom onsets, random frames) examples.
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        sample_rate: int = 44100,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048,
        context_frames: int = 43,
        n_cqt_bins: int = 84,
        max_songs: int = 0,
        neg_ratio: float = 3.0,
        max_duration_sec: float = 360.0,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.context_frames = context_frames
        self.n_cqt_bins = n_cqt_bins
        self.half_ctx = context_frames // 2
        self.neg_ratio = neg_ratio

        manifest_path = Path(manifest_path)
        self.data_dir = manifest_path.parent

        with open(manifest_path) as f:
            manifest = json.load(f)

        songs = [
            s for s in manifest['songs']
            if s['split'] == split and s['charts'].get('drums')
        ]

        # Validate songs
        valid_songs = []
        for song in songs:
            audio_path = self.data_dir / song['stems']['drums']
            labels_path = self.data_dir / song['id'] / 'drums_labels.json'
            if audio_path.exists() and labels_path.exists():
                if audio_path.stat().st_size > 1000:
                    valid_songs.append(song)

        if max_songs > 0:
            valid_songs = valid_songs[:max_songs]

        # Pre-extract all windows
        self.windows = []  # list of (song_id, frame_idx, label)
        self.song_data = {}  # song_id -> (audio_path, labels)

        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        self._cqt_transform = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_cqt_bins,
            speckwargs={'n_fft': n_fft, 'hop_length': hop_length},
        )

        logger.info(f"Scanning {len(valid_songs)} songs for tom windows...")
        total_pos = [0, 0, 0]  # HighTom, LowTom, FloorTom
        total_neg = 0

        for i, song in enumerate(valid_songs):
            audio_path = self.data_dir / song['stems']['drums']
            labels_path = self.data_dir / song['id'] / 'drums_labels.json'

            try:
                info = sf.info(str(audio_path))
                if info.duration > max_duration_sec or info.duration < 10:
                    continue
            except Exception:
                continue

            with open(labels_path) as f:
                labels = json.load(f)

            hits = labels.get('hits', [])
            if not hits:
                continue

            # Compute total frames for this song
            n_samples = int(info.duration * sample_rate)
            T = n_samples // hop_length + 1

            # Classify each hit
            tom_frames = {}     # frame_idx -> tom_label (1=High, 2=Low, 3=Floor)
            non_tom_frames = [] # frames with non-tom onsets

            for hit in hits:
                time_ms = hit['time_ms']
                frame_idx = int(time_ms / 1000 * sample_rate / hop_length)
                if frame_idx < self.half_ctx or frame_idx >= T - self.half_ctx:
                    continue

                lane = hit['lane']
                is_cymbal = hit['is_cymbal']
                cls = LANE_CYMBAL_TO_CLASS.get((lane, is_cymbal))

                if cls == 3:  # HighTom
                    tom_frames[frame_idx] = 1
                    total_pos[0] += 1
                elif cls == 5:  # LowTom
                    tom_frames[frame_idx] = 2
                    total_pos[1] += 1
                elif cls == 7:  # FloorTom
                    tom_frames[frame_idx] = 3
                    total_pos[2] += 1
                else:
                    non_tom_frames.append(frame_idx)

            if not tom_frames:
                continue

            # Store positive windows
            song_id = song['id']
            self.song_data[song_id] = str(audio_path)

            for frame_idx, label in tom_frames.items():
                self.windows.append((song_id, frame_idx, label))

            # Sample negatives: mix of non-tom onsets + random frames
            n_neg = int(len(tom_frames) * neg_ratio)
            rng = np.random.RandomState(hash(song_id) % (2**31))

            # Half from non-tom onsets (harder negatives)
            n_onset_neg = min(n_neg // 2, len(non_tom_frames))
            if non_tom_frames and n_onset_neg > 0:
                onset_neg_idx = rng.choice(len(non_tom_frames), n_onset_neg, replace=False)
                for idx in onset_neg_idx:
                    f = non_tom_frames[idx]
                    if f not in tom_frames:
                        self.windows.append((song_id, f, 0))
                        total_neg += 1

            # Half from random frames (easier negatives)
            n_random_neg = n_neg - n_onset_neg
            valid_range = list(range(self.half_ctx, T - self.half_ctx))
            if valid_range and n_random_neg > 0:
                random_frames = rng.choice(valid_range, min(n_random_neg, len(valid_range)), replace=False)
                for f in random_frames:
                    if f not in tom_frames:
                        self.windows.append((song_id, int(f), 0))
                        total_neg += 1

            if (i + 1) % 200 == 0:
                logger.info(f"  Scanned {i+1}/{len(valid_songs)} songs...")

        logger.info(
            f"TomRefinementDataset [{split}]: {len(self.windows)} windows | "
            f"HighTom={total_pos[0]}, LowTom={total_pos[1]}, FloorTom={total_pos[2]}, "
            f"NotTom={total_neg}"
        )

        # Precompute all windows into memory tensors
        logger.info("Precomputing windows into memory...")
        self.mel_windows = torch.zeros(len(self.windows), 1, n_mels, context_frames)
        self.cqt_windows = torch.zeros(len(self.windows), 1, n_cqt_bins, context_frames)
        self.labels = torch.zeros(len(self.windows), dtype=torch.long)

        # Group windows by song for efficient loading
        from collections import defaultdict
        song_windows = defaultdict(list)
        for i, (song_id, frame_idx, label) in enumerate(self.windows):
            song_windows[song_id].append((i, frame_idx, label))

        songs_done = 0
        for song_id, wins in song_windows.items():
            audio_path = self.song_data[song_id]
            try:
                audio, sr = sf.read(audio_path, dtype='float32')
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != sample_rate:
                    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
                    audio_t = torchaudio.functional.resample(audio_t, sr, sample_rate)
                    audio = audio_t.squeeze(0).numpy()

                audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
                mel = self._mel_transform(audio_tensor)
                mel = torch.log(mel + 1e-8).squeeze(0)  # (n_mels, T)
                cqt = self._cqt_transform(audio_tensor)
                cqt = cqt.squeeze(0)                      # (n_cqt, T)
                T = mel.shape[-1]

                for idx, frame_idx, label in wins:
                    start = max(0, frame_idx - self.half_ctx)
                    end = min(T, frame_idx + self.half_ctx + 1)

                    mel_w = mel[:, start:end]
                    cqt_w = cqt[:, start:end]

                    # Pad/truncate to context_frames
                    if mel_w.shape[-1] < context_frames:
                        pad = context_frames - mel_w.shape[-1]
                        mel_w = F.pad(mel_w, (0, pad))
                        cqt_w = F.pad(cqt_w, (0, pad))
                    elif mel_w.shape[-1] > context_frames:
                        mel_w = mel_w[:, :context_frames]
                        cqt_w = cqt_w[:, :context_frames]

                    self.mel_windows[idx, 0] = mel_w
                    self.cqt_windows[idx, 0] = cqt_w
                    self.labels[idx] = label

            except Exception as e:
                logger.warning(f"Error processing {song_id}: {e}")

            songs_done += 1
            if songs_done % 200 == 0:
                logger.info(f"  Precomputed {songs_done}/{len(song_windows)} songs...")

        logger.info(
            f"Precomputed {len(self.windows)} windows "
            f"({self.mel_windows.nbytes / 1e9:.1f}GB mel + "
            f"{self.cqt_windows.nbytes / 1e9:.1f}GB cqt)"
        )

        # Free scan data
        self.windows = None
        self.song_data = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.mel_windows[idx], self.cqt_windows[idx], self.labels[idx].item()
