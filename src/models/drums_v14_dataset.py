"""
V14 Full-Song Drums Dataset — loads entire songs for full-context training.

Key differences from DrumsV13Dataset:
  1. Returns entire songs instead of fixed-length segments
  2. Batch size = 1 (variable-length), gradient accumulation for effective batch
  3. No segmentation or overlap logic
  4. Augmentation limited to gain/noise/SpecAugment (no time-stretch — too slow on full songs)
  5. Songs sorted by duration for more predictable memory usage
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Reuse mappings from V13
LANE_CYMBAL_TO_CLASS = {
    (0, False): 0,  # Kick
    (0, True): 0,   # Kick
    (1, False): 1,  # Snare
    (1, True): 1,   # Snare
    (2, True): 2,   # HiHat (cymbal)
    (2, False): 3,  # HighTom (tom)
    (3, True): 4,   # Ride (cymbal)
    (3, False): 5,  # LowTom (tom)
    (4, True): 6,   # Crash (cymbal)
    (4, False): 7,  # FloorTom (tom)
}

CLASS_NAMES = [
    'Kick', 'Snare', 'HiHat', 'HighTom',
    'Ride', 'LowTom', 'Crash', 'FloorTom',
]


class DrumsV14FullSongDataset(Dataset):
    """
    Dataset that returns entire songs for full-context drum transcription.

    Each item is one complete song:
      - mel_spec: (1, n_mels, T) log-mel spectrogram (variable T)
      - onset_target: (T, 1) binary onset
      - class_target: (T, 8) per-class binary
      - velocity_target: (T, 8) velocity values
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        sample_rate: int = 44100,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 2048,
        augment: bool = False,
        augment_config: Optional[dict] = None,
        max_duration_sec: float = 600.0,
        source_filter: Optional[set[str]] = None,
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.augment = augment
        self.augment_config = augment_config or {}
        self.max_duration_sec = max_duration_sec

        manifest_path = Path(manifest_path)
        with open(manifest_path) as f:
            manifest_data = json.load(f)

        self.data_dir = manifest_path.parent

        # Filter songs by split that have drums
        all_songs = [
            s for s in manifest_data["songs"]
            if s["split"] == split and s["charts"].get("drums")
        ]

        # Optional: filter by source game
        if source_filter:
            all_songs = [s for s in all_songs if s.get("source_game") in source_filter]

        # Validate and get durations
        self.songs = []
        self.durations = []
        skipped = 0

        for song in all_songs:
            audio_path = self.data_dir / song["stems"]["drums"]
            # Use pseudo labels if marked
            if song.get("pseudo_labels"):
                labels_path = self.data_dir / song["id"] / "drums_labels_pseudo.json"
                if not labels_path.exists():
                    labels_path = self.data_dir / song["id"] / "drums_labels.json"
            else:
                labels_path = self.data_dir / song["id"] / "drums_labels.json"

            if not audio_path.exists() or not labels_path.exists():
                skipped += 1
                continue

            if audio_path.stat().st_size < 1000:
                skipped += 1
                continue

            try:
                info = sf.info(str(audio_path))
                duration = info.duration
            except Exception as e:
                logger.warning(f"Skipping {song['id']}: cannot read audio - {e}")
                skipped += 1
                continue

            if duration > max_duration_sec or duration < 10.0:
                skipped += 1
                continue

            # Verify labels have hits
            try:
                with open(labels_path) as f:
                    labels = json.load(f)
                if not labels.get("hits"):
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            self.songs.append(song)
            self.durations.append(duration)

        if skipped:
            logger.info(f"Skipped {skipped} songs (missing data/too short/too long)")

        # MelSpectrogram created lazily per worker (fork-safe)
        self._mel_transform = None
        self._worker_pid = None

        # Duration stats
        durs = np.array(self.durations)
        logger.info(
            f"DrumsV14FullSongDataset [{split}]: {len(self.songs)} songs | "
            f"Duration: mean={durs.mean():.0f}s, median={np.median(durs):.0f}s, "
            f"max={durs.max():.0f}s, min={durs.min():.0f}s"
        )

    @property
    def mel_transform(self) -> torchaudio.transforms.MelSpectrogram:
        """Lazy-init mel transform per worker process (fork-safe)."""
        pid = os.getpid()
        if self._mel_transform is None or self._worker_pid != pid:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            self._worker_pid = pid
        return self._mel_transform

    def __len__(self) -> int:
        return len(self.songs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        song = self.songs[idx]

        # Load full audio
        audio_path = str(self.data_dir / song["stems"]["drums"])
        audio = self._load_full_audio(audio_path)

        # Compute mel spectrogram
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)
        power_mel = self.mel_transform(audio_tensor)  # (1, n_mels, T) linear power

        # Optional background subtraction (matches inference path exactly).
        # Honors STRUM_MEL_BG_SUBTRACT/ALPHA/PCTL/WINDOW_SEC env vars.
        from src.models.bg_mel import bg_subtract_log_mel
        mel_spec = bg_subtract_log_mel(
            power_mel,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            model_tag="v14",
        )

        T = mel_spec.shape[-1]

        # Load labels for full song
        onset_target, class_target, velocity_target = self._load_labels(song, T)

        # Apply augmentation (subset safe for full songs)
        if self.augment:
            mel_spec, onset_target, class_target, velocity_target = self._apply_augmentation(
                mel_spec, onset_target, class_target, velocity_target
            )

        targets = {
            'onset': onset_target,        # (T, 1)
            'classes': class_target,       # (T, 8)
            'velocities': velocity_target, # (T, 8)
        }

        return mel_spec, targets

    def _load_full_audio(self, audio_path: str) -> np.ndarray:
        """Load entire audio file using soundfile (fork-safe)."""
        try:
            audio, sr = sf.read(audio_path, dtype='float32')
        except Exception as e:
            logger.warning(f"Error reading {audio_path}: {e}")
            # Return 30s of silence as fallback
            return np.zeros(int(30 * self.sample_rate), dtype=np.float32)

        # Convert to mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.sample_rate)
            audio = audio_tensor.squeeze(0).numpy()

        return audio

    def _load_labels(
        self,
        song: dict,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load labels for the full song.

        Returns:
            onset_target: (T, 1) binary onset
            class_target: (T, 8) per-class binary
            velocity_target: (T, 8) velocity values
        """
        # Use pseudo labels if available for this song
        if song.get("pseudo_labels"):
            labels_path = self.data_dir / song["id"] / "drums_labels_pseudo.json"
            if not labels_path.exists():
                labels_path = self.data_dir / song["id"] / "drums_labels.json"
        else:
            labels_path = self.data_dir / song["id"] / "drums_labels.json"

        with open(labels_path) as f:
            labels_data = json.load(f)

        onset_target = torch.zeros(T, 1)
        class_target = torch.zeros(T, 8)
        velocity_target = torch.zeros(T, 8)

        for hit in labels_data["hits"]:
            time_ms = hit["time_ms"]
            frame_idx = int(time_ms / 1000 * self.sample_rate / self.hop_length)

            if 0 <= frame_idx < T:
                lane = hit["lane"]
                is_cymbal = hit["is_cymbal"]
                velocity = hit["velocity"] / 127.0

                cls = LANE_CYMBAL_TO_CLASS.get((lane, is_cymbal))
                if cls is not None:
                    onset_target[frame_idx, 0] = 1.0
                    class_target[frame_idx, cls] = 1.0
                    velocity_target[frame_idx, cls] = velocity

        return onset_target, class_target, velocity_target

    def _apply_augmentation(
        self,
        mel_spec: torch.Tensor,
        onset_target: torch.Tensor,
        class_target: torch.Tensor,
        velocity_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Augmentation for full songs — lightweight ops only."""

        # Pitch shift (mel bin shift) — configurable range
        pitch_range = self.augment_config.get('pitch_shift_range', 4)
        if np.random.random() < 0.3:
            shift = np.random.randint(-pitch_range, pitch_range + 1)
            if shift != 0:
                min_val = mel_spec.min().item()
                if shift > 0:
                    mel_spec = F.pad(mel_spec[:, :-shift, :], (0, 0, shift, 0), value=min_val)
                else:
                    mel_spec = F.pad(mel_spec[:, -shift:, :], (0, 0, 0, -shift), value=min_val)

        # Additive noise — configurable max std
        noise_max = self.augment_config.get('noise_std_max', 0.05)
        if np.random.random() < 0.3:
            noise = torch.randn_like(mel_spec) * np.random.uniform(0.0, noise_max)
            mel_spec = mel_spec + noise

        # Gain variation — configurable range
        gain_range = self.augment_config.get('gain_db_range', 6)
        if np.random.random() < 0.5:
            gain_db = np.random.uniform(-gain_range, gain_range)
            mel_spec = mel_spec * (10 ** (gain_db / 20))

        # SpecAugment: Frequency masking
        if np.random.random() < 0.5:
            for _ in range(np.random.randint(1, 3)):
                f_param = self.augment_config.get('freq_mask_param', 27)
                f = np.random.randint(0, f_param)
                f0 = np.random.randint(0, mel_spec.shape[1] - f)
                mel_spec[:, f0:f0 + f, :] = mel_spec.min().item()

        # SpecAugment: Time masking (proportional to song length)
        if np.random.random() < 0.5:
            T = mel_spec.shape[-1]
            for _ in range(np.random.randint(1, 4)):
                t_param = self.augment_config.get('time_mask_param', 50)
                t = np.random.randint(0, min(t_param, T // 8))
                t0 = np.random.randint(0, T - t)
                mel_spec[:, :, t0:t0 + t] = mel_spec.min().item()
                onset_target[t0:t0 + t] = 0.0
                class_target[t0:t0 + t] = 0.0
                velocity_target[t0:t0 + t] = 0.0

        return mel_spec, onset_target, class_target, velocity_target
