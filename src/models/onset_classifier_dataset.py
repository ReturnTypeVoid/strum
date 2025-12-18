"""
Onset Window Dataset — extracts small mel windows around ground-truth onsets.

Each training example is a ~300ms mel spectrogram patch centered on one onset,
labeled with the drum class. Two mel resolutions are computed per window:
  - Fine: n_fft=1024 (~23ms) — captures transient/attack shape
  - Coarse: n_fft=4096 (~93ms) — captures tonal content, pitch

Class-balanced sampling is built in: `get_sampler()` returns a
WeightedRandomSampler that gives equal probability to all 8 classes,
completely eliminating the brutal 41:1 Kick:HighTom imbalance.

Onset context: for each onset, we encode the classes of the nearest
N previous and N next onsets as one-hot vectors — giving the model
pattern context without needing long sequences.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

# Lane+cymbal → 8-class mapping (same as V13)
LANE_CYMBAL_TO_CLASS = {
    (0, False): 0,  (0, True): 0,   # Kick
    (1, False): 1,  (1, True): 1,   # Snare
    (2, True): 2,   (2, False): 3,  # HiHat / HighTom
    (3, True): 4,   (3, False): 5,  # Ride / LowTom
    (4, True): 6,   (4, False): 7,  # Crash / FloorTom
}

CLASS_NAMES = [
    'Kick', 'Snare', 'HiHat', 'HighTom',
    'Ride', 'LowTom', 'Crash', 'FloorTom',
]


class OnsetWindowDataset(Dataset):
    """
    Dataset of individual onset windows with dual-resolution mel spectrograms.

    Each item returns:
        mel_fine: (1, n_mels, T_fine) — fine-resolution log-mel
        mel_coarse: (1, n_mels, T_coarse) — coarse-resolution log-mel
        context: (2 * context_size * 8,) — flattened neighbor one-hots
        label: (8,) — multi-hot class vector (usually single class)
    """

    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        sample_rate: int = 44100,
        n_mels: int = 128,
        # Dual resolution
        fine_n_fft: int = 1024,
        fine_hop: int = 256,
        coarse_n_fft: int = 4096,
        coarse_hop: int = 512,
        # Window around onset
        window_before_ms: float = 50.0,
        window_after_ms: float = 200.0,
        # Context
        context_size: int = 4,
        # Augmentation
        augment: bool = False,
        noise_std: float = 0.005,
        gain_range: tuple[float, float] = (0.8, 1.2),
    ):
        self.split = split
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fine_n_fft = fine_n_fft
        self.fine_hop = fine_hop
        self.coarse_n_fft = coarse_n_fft
        self.coarse_hop = coarse_hop
        self.context_size = context_size
        self.augment = augment
        self.noise_std = noise_std
        self.gain_range = gain_range

        # Window size in samples
        self.window_before_samples = int(window_before_ms / 1000 * sample_rate)
        self.window_after_samples = int(window_after_ms / 1000 * sample_rate)
        self.window_samples = self.window_before_samples + self.window_after_samples

        # Expected output frames per resolution
        self.fine_frames = self.window_samples // fine_hop + 1
        self.coarse_frames = self.window_samples // coarse_hop + 1

        # Mel transforms (created lazily per worker for fork-safety)
        self._fine_mel = None
        self._coarse_mel = None
        self._worker_pid = None

        manifest_path = Path(manifest_path)
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        self.data_dir = manifest_path.parent

        songs = [
            s for s in manifest_data["songs"]
            if s["split"] == split and s["charts"].get("drums")
        ]

        # Build onset index: list of (song_info, onset_time_ms, class_idx, neighbor_classes)
        self.onsets = []
        self.class_counts = Counter()
        self._build_onset_index(songs)

        logger.info(
            f"OnsetWindowDataset [{split}]: {len(self.onsets)} onsets from "
            f"{len(songs)} songs"
        )
        for i, name in enumerate(CLASS_NAMES):
            logger.info(f"  {name}: {self.class_counts[i]}")

    def _get_mel_transforms(self):
        """Create mel transforms lazily (fork-safe with forkserver)."""
        import os
        pid = os.getpid()
        if self._fine_mel is None or self._worker_pid != pid:
            self._worker_pid = pid
            self._fine_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.fine_n_fft,
                hop_length=self.fine_hop,
                n_mels=self.n_mels,
                power=2.0,
            )
            self._coarse_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.coarse_n_fft,
                hop_length=self.coarse_hop,
                n_mels=self.n_mels,
                power=2.0,
            )
        return self._fine_mel, self._coarse_mel

    def _build_onset_index(self, songs: list[dict]) -> None:
        """Build flat index of all onsets with class labels and neighbor context."""
        skipped_songs = 0

        for song in songs:
            labels_path = self.data_dir / song["id"] / "drums_labels.json"
            audio_path = self.data_dir / song["stems"]["drums"]

            if not labels_path.exists() or not audio_path.exists():
                skipped_songs += 1
                continue

            if audio_path.stat().st_size < 1000:
                skipped_songs += 1
                continue

            with open(labels_path) as f:
                labels = json.load(f)

            if not labels["hits"]:
                skipped_songs += 1
                continue

            # Get audio duration
            try:
                info = sf.info(str(audio_path))
                audio_duration_ms = info.duration * 1000
            except Exception:
                skipped_songs += 1
                continue

            # Group simultaneous hits (within 5ms) and sort by time
            hits_by_time = {}
            for hit in labels["hits"]:
                t = hit["time_ms"]
                cls = LANE_CYMBAL_TO_CLASS.get((hit["lane"], hit["is_cymbal"]))
                if cls is None:
                    continue
                # Quantize to 5ms bins for simultaneous hit grouping
                bin_t = round(t / 5) * 5
                if bin_t not in hits_by_time:
                    hits_by_time[bin_t] = {"time_ms": t, "classes": set()}
                hits_by_time[bin_t]["classes"].add(cls)

            sorted_times = sorted(hits_by_time.keys())
            onset_list = [(hits_by_time[t]["time_ms"], hits_by_time[t]["classes"])
                          for t in sorted_times]

            # For each onset, record it with its neighbor context
            for i, (time_ms, classes) in enumerate(onset_list):
                # Skip onsets too close to audio boundaries
                if time_ms < self.window_before_samples / self.sample_rate * 1000:
                    continue
                if time_ms > audio_duration_ms - self.window_after_samples / self.sample_rate * 1000:
                    continue

                # Build neighbor context (previous + next N onsets)
                neighbor_classes = []
                for offset in range(-self.context_size, 0):
                    idx = i + offset
                    if 0 <= idx < len(onset_list):
                        neighbor_classes.append(onset_list[idx][1])
                    else:
                        neighbor_classes.append(set())  # padding

                for offset in range(1, self.context_size + 1):
                    idx = i + offset
                    if 0 <= idx < len(onset_list):
                        neighbor_classes.append(onset_list[idx][1])
                    else:
                        neighbor_classes.append(set())

                # Store one entry per onset (multi-label for simultaneous hits)
                primary_class = min(classes)  # For sampling weight purposes
                self.class_counts[primary_class] += 1

                self.onsets.append({
                    "audio_path": str(audio_path),
                    "time_ms": time_ms,
                    "classes": classes,
                    "primary_class": primary_class,
                    "neighbors": neighbor_classes,
                })

        if skipped_songs:
            logger.info(f"Skipped {skipped_songs} songs (missing/invalid data)")

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Class-balanced sampler: every class gets equal sampling probability.

        This is THE fix for class imbalance. Instead of seeing HighTom in
        0.7% of examples, the sampler draws HighTom-containing onsets ~12.5%
        of the time (1/8 classes).
        """
        # Weight per sample = 1 / count_of_its_class
        weights = []
        for onset in self.onsets:
            cls = onset["primary_class"]
            w = 1.0 / max(self.class_counts[cls], 1)
            weights.append(w)

        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.onsets),
            replacement=True,
        )

    def get_hard_negative_sampler(
        self, confusion_pairs: Optional[dict[tuple[int, int], float]] = None
    ) -> WeightedRandomSampler:
        """
        Enhanced sampler that upweights confused class pairs.

        Args:
            confusion_pairs: dict mapping (true_class, pred_class) → confusion_rate.
                             Higher rate = more upweighting for that true class.
        """
        if confusion_pairs is None:
            return self.get_sampler()

        # Start with balanced weights
        base_weights = []
        for onset in self.onsets:
            cls = onset["primary_class"]
            w = 1.0 / max(self.class_counts[cls], 1)

            # Boost weight for classes involved in confusion
            boost = 1.0
            for (true_cls, pred_cls), rate in confusion_pairs.items():
                if cls == true_cls:
                    boost = max(boost, 1.0 + rate * 2.0)
            base_weights.append(w * boost)

        return WeightedRandomSampler(
            weights=base_weights,
            num_samples=len(self.onsets),
            replacement=True,
        )

    def __len__(self) -> int:
        return len(self.onsets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        onset = self.onsets[idx]

        # Load audio window
        audio = self._load_audio_window(onset["audio_path"], onset["time_ms"])

        # Augmentation
        if self.augment:
            audio = self._augment_audio(audio)

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)

        # Dual-resolution mel spectrograms
        fine_mel_transform, coarse_mel_transform = self._get_mel_transforms()
        mel_fine = torch.log(fine_mel_transform(audio_tensor) + 1e-8)    # (1, n_mels, T_fine)
        mel_coarse = torch.log(coarse_mel_transform(audio_tensor) + 1e-8)  # (1, n_mels, T_coarse)

        # Pad/trim to consistent size
        mel_fine = self._pad_or_trim(mel_fine, self.fine_frames)
        mel_coarse = self._pad_or_trim(mel_coarse, self.coarse_frames)

        # Context encoding
        context = self._encode_context(onset["neighbors"])

        # Label (multi-hot)
        label = torch.zeros(8)
        for cls in onset["classes"]:
            label[cls] = 1.0

        return mel_fine, mel_coarse, context, label

    def _load_audio_window(self, audio_path: str, time_ms: float) -> np.ndarray:
        """Load audio window centered on onset time."""
        center_sample = int(time_ms / 1000 * self.sample_rate)
        start_sample = max(0, center_sample - self.window_before_samples)
        num_samples = self.window_samples

        try:
            info = sf.info(audio_path)
            file_sr = info.samplerate

            if file_sr != self.sample_rate:
                start_file = int(start_sample * file_sr / self.sample_rate)
                num_file = int(num_samples * file_sr / self.sample_rate)
            else:
                start_file = start_sample
                num_file = num_samples

            audio, sr = sf.read(audio_path, start=start_file, frames=num_file, dtype='float32')
        except Exception as e:
            logger.warning(f"Error reading {audio_path}: {e}")
            return np.zeros(self.window_samples, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if len(audio) == 0:
            return np.zeros(self.window_samples, dtype=np.float32)

        # Resample if needed
        if sr != self.sample_rate:
            audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, self.sample_rate)
            audio = audio_t.squeeze(0).numpy()

        # Pad/trim
        if len(audio) < self.window_samples:
            audio = np.pad(audio, (0, self.window_samples - len(audio)))
        elif len(audio) > self.window_samples:
            audio = audio[:self.window_samples]

        return audio

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Simple audio augmentation: noise + gain."""
        rng = np.random.default_rng()

        # Additive noise
        if self.noise_std > 0:
            noise = rng.normal(0, self.noise_std, size=audio.shape).astype(np.float32)
            audio = audio + noise

        # Random gain
        lo, hi = self.gain_range
        gain = rng.uniform(lo, hi)
        audio = audio * gain

        return audio

    def _pad_or_trim(self, mel: torch.Tensor, target_frames: int) -> torch.Tensor:
        """Ensure mel has exactly target_frames time frames."""
        T = mel.shape[-1]
        if T > target_frames:
            mel = mel[..., :target_frames]
        elif T < target_frames:
            pad = target_frames - T
            mel = torch.nn.functional.pad(mel, (0, pad), value=mel.min().item())
        return mel

    def _encode_context(self, neighbors: list[set[int]]) -> torch.Tensor:
        """Encode neighbor onset classes as flattened one-hot vector."""
        parts = []
        for cls_set in neighbors:
            one_hot = torch.zeros(8)
            for cls in cls_set:
                one_hot[cls] = 1.0
            parts.append(one_hot)
        return torch.cat(parts)  # (2 * context_size * 8,)
