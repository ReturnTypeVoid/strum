"""
Cached Onset Window Dataset — loads pre-extracted onset windows from memmap files.

Created by scripts/preprocess_onset_windows.py, this dataset memory-maps
.npy files for near-zero load time and minimal memory footprint (~0 MB
resident, OS manages page cache from the ~46 GB files).

Class-balanced sampling via get_sampler() / get_hard_negative_sampler().
"""

import json
import logging
import mmap as mmap_module
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)

CLASS_NAMES = [
    'Kick', 'Snare', 'HiHat', 'HighTom',
    'Ride', 'LowTom', 'Crash', 'FloorTom',
]


class CachedOnsetDataset(Dataset):
    """
    Fast onset window dataset backed by memory-mapped .npy files.

    Data stays on disk and is paged in by the OS as needed.
    Each __getitem__ just indexes numpy memmaps and converts to tensors.
    """

    def __init__(
        self,
        cache_dir: str,
        split: str,
        augment: bool = False,
        noise_std: float = 0.005,
        gain_db_range: tuple[float, float] = (-2.0, 2.0),
        spec_augment: bool = False,
        freq_mask_width: int = 12,
        time_mask_width: int = 8,
        clean_mask_path: str | None = None,
        bleed_cache_dir: str | None = None,
        bleed_prob: float = 0.5,
    ):
        cache_dir = Path(cache_dir)
        index_path = cache_dir / f"{split}_index.json"
        logger.info(f"Loading onset cache index: {index_path}")

        with open(index_path) as f:
            index = json.load(f)

        self.total = index["total_onsets"]

        # Memory-map the arrays (readonly)
        self.mel_fine = np.load(
            str(cache_dir / index["files"]["mel_fine"]), mmap_mode='r')
        self.mel_coarse = np.load(
            str(cache_dir / index["files"]["mel_coarse"]), mmap_mode='r')
        # CQT or low-freq mel (backward compat: try CQT first, then mel_lowfreq)
        cqt_file = index["files"].get("cqt")
        lowfreq_file = index["files"].get("mel_lowfreq")
        if cqt_file and (cache_dir / cqt_file).exists():
            self.mel_lowfreq = np.load(
                str(cache_dir / cqt_file), mmap_mode='r')
            logger.info(f"  Loaded CQT branch: {cqt_file}")
        elif lowfreq_file and (cache_dir / lowfreq_file).exists():
            self.mel_lowfreq = np.load(
                str(cache_dir / lowfreq_file), mmap_mode='r')
        else:
            self.mel_lowfreq = None
        self.labels = np.load(
            str(cache_dir / index["files"]["labels"]), mmap_mode='r')
        self.contexts = np.load(
            str(cache_dir / index["files"]["contexts"]), mmap_mode='r')

        # Track mmap handles for periodic page cache flushing
        # (144 GB of mmap files vs 122.5 GB RAM → CUDA unified memory deadlock)
        self._mmaps: list = []
        for arr in [self.mel_fine, self.mel_coarse, self.mel_lowfreq,
                     self.labels, self.contexts]:
            if arr is not None and hasattr(arr, '_mmap'):
                self._mmaps.append(arr._mmap)

        self.augment = augment
        self.noise_std = noise_std
        self.gain_db_range = gain_db_range
        self.spec_augment = spec_augment
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width

        # Build class counts from index (avoid scanning all labels)
        self.class_counts = Counter()
        for k, v in index.get("class_counts", {}).items():
            self.class_counts[int(k)] += v

        # Primary class per onset (scan labels — fast even for 2.7M with memmap)
        self._primary_classes = self.labels[:self.total].argmax(axis=1)

        # Optional clean mask: boolean array excluding suspected mislabeled songs
        self.clean_mask: np.ndarray | None = None
        if clean_mask_path and Path(clean_mask_path).exists():
            self.clean_mask = np.load(clean_mask_path)[:self.total]
            n_excluded = (~self.clean_mask).sum()
            logger.info(f"  Clean mask: excluding {n_excluded} onsets "
                        f"({n_excluded/self.total*100:.1f}%)")

        logger.info(f"  Loaded {self.total} {split} onsets (memmap)")
        for i, name in enumerate(CLASS_NAMES):
            logger.info(f"  {name}: {self.class_counts.get(i, 0)}")

        # Optional bleed cache: augmented mel/CQT with instrument bleed
        self.bleed_prob = bleed_prob
        self._bleed_fine = None
        self._bleed_coarse = None
        self._bleed_lowfreq = None
        if bleed_cache_dir and Path(bleed_cache_dir).exists():
            bleed_dir = Path(bleed_cache_dir)
            bleed_idx_path = bleed_dir / f"{split}_index.json"
            if bleed_idx_path.exists():
                with open(bleed_idx_path) as f:
                    bleed_index = json.load(f)
                bleed_total = bleed_index["total_onsets"]
                # Bleed cache may have fewer onsets (songs without non-drum stems skipped)
                self._bleed_total = min(bleed_total, self.total)
                bf = bleed_dir / bleed_index["files"]["mel_fine"]
                bc = bleed_dir / bleed_index["files"]["mel_coarse"]
                if bf.exists():
                    self._bleed_fine = np.load(str(bf), mmap_mode='r')
                if bc.exists():
                    self._bleed_coarse = np.load(str(bc), mmap_mode='r')
                # CQT from bleed cache
                bcqt = bleed_index["files"].get("cqt")
                if bcqt and (bleed_dir / bcqt).exists():
                    self._bleed_lowfreq = np.load(str(bleed_dir / bcqt), mmap_mode='r')
                # Track mmaps for page cache flushing
                for arr in [self._bleed_fine, self._bleed_coarse, self._bleed_lowfreq]:
                    if arr is not None and hasattr(arr, '_mmap'):
                        self._mmaps.append(arr._mmap)
                logger.info(f"  Bleed cache loaded: {bleed_dir} "
                            f"({self._bleed_total} onsets, prob={bleed_prob})")

    def drop_page_cache(self) -> None:
        """Release cached mmap pages to prevent CUDA unified memory starvation."""
        for mm in self._mmaps:
            try:
                mm.madvise(mmap_module.MADV_DONTNEED)
            except Exception:
                pass

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # With bleed_prob, use bleed-augmented features instead of clean
        use_bleed = (self._bleed_fine is not None
                     and idx < self._bleed_total
                     and np.random.random() < self.bleed_prob)

        if use_bleed:
            mel_fine = self._bleed_fine[idx].astype(np.float32)
            mel_coarse = self._bleed_coarse[idx].astype(np.float32)
            mel_lowfreq = self._bleed_lowfreq[idx].astype(np.float32) if self._bleed_lowfreq is not None else None
        else:
            mel_fine = self.mel_fine[idx].astype(np.float32)
            mel_coarse = self.mel_coarse[idx].astype(np.float32)
            mel_lowfreq = self.mel_lowfreq[idx].astype(np.float32) if self.mel_lowfreq is not None else None
        context = self.contexts[idx].astype(np.float32)
        label = self.labels[idx].astype(np.float32)

        if self.augment:
            primary_class = int(self._primary_classes[idx])
            mel_fine, mel_coarse, mel_lowfreq = self._augment(mel_fine, mel_coarse, primary_class, mel_lowfreq)

        mel_fine = torch.from_numpy(mel_fine.copy()).unsqueeze(0)
        mel_coarse = torch.from_numpy(mel_coarse.copy()).unsqueeze(0)
        if mel_lowfreq is not None:
            mel_lowfreq = torch.from_numpy(mel_lowfreq.copy()).unsqueeze(0)
        context = torch.from_numpy(context.copy())
        label = torch.from_numpy(label.copy())

        return mel_fine, mel_coarse, context, label, mel_lowfreq

    def _augment(
        self, mel_fine: np.ndarray, mel_coarse: np.ndarray, primary_class: int,
        mel_lowfreq: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Augment in log-mel space: noise + gain + pitch-shift + SpecAugment."""
        rng = np.random.default_rng()

        if self.noise_std > 0:
            mel_fine = mel_fine + rng.standard_normal(mel_fine.shape, dtype=np.float32) * np.float32(self.noise_std)
            mel_coarse = mel_coarse + rng.standard_normal(mel_coarse.shape, dtype=np.float32) * np.float32(self.noise_std)
            if mel_lowfreq is not None:
                mel_lowfreq = mel_lowfreq + rng.standard_normal(mel_lowfreq.shape, dtype=np.float32) * np.float32(self.noise_std)

        lo, hi = self.gain_db_range
        gain_log = np.float32(rng.uniform(lo, hi)) * np.float32(np.log(10) / 20)
        mel_fine = mel_fine + gain_log
        mel_coarse = mel_coarse + gain_log
        if mel_lowfreq is not None:
            mel_lowfreq = mel_lowfreq + gain_log

        # Pitch-shift augmentation: shift mel bins up/down
        # Applied more aggressively to rare classes (toms, crash)
        # HighTom=3, LowTom=5, FloorTom=7, Crash=6
        rare_classes = {3, 5, 6, 7}
        if primary_class in rare_classes and rng.random() < 0.5:
            shift = rng.integers(-4, 5)  # -4 to +4 mel bins
            if shift != 0:
                mel_fine = np.roll(mel_fine, shift, axis=0)
                mel_coarse = np.roll(mel_coarse, shift, axis=0)
                # Zero out the wrapped-around bins
                if shift > 0:
                    mel_fine[:shift, :] = mel_fine[shift, :]
                    mel_coarse[:shift, :] = mel_coarse[shift, :]
                else:
                    mel_fine[shift:, :] = mel_fine[shift - 1, :]
                    mel_coarse[shift:, :] = mel_coarse[shift - 1, :]
                # Low-freq mel: shift by proportionally more bins since
                # it covers a narrower freq range (each bin = smaller interval)
                if mel_lowfreq is not None:
                    lf_shift = shift * 3  # ~3x more bins per Hz
                    mel_lowfreq = np.roll(mel_lowfreq, lf_shift, axis=0)
                    if lf_shift > 0:
                        mel_lowfreq[:lf_shift, :] = mel_lowfreq[lf_shift, :]
                    else:
                        mel_lowfreq[lf_shift:, :] = mel_lowfreq[lf_shift - 1, :]

        # SpecAugment: frequency and time masking
        if self.spec_augment:
            F_bins, T_fine = mel_fine.shape
            _, T_coarse = mel_coarse.shape

            # Frequency masking (1-2 masks)
            n_freq_masks = rng.integers(1, 3)
            for _ in range(n_freq_masks):
                f_width = rng.integers(1, self.freq_mask_width + 1)
                f_start = rng.integers(0, max(F_bins - f_width, 1))
                fill_val = mel_fine.mean()
                mel_fine[f_start:f_start + f_width, :] = fill_val
                mel_coarse[f_start:f_start + f_width, :] = fill_val
                if mel_lowfreq is not None:
                    mel_lowfreq[f_start:f_start + f_width, :] = mel_lowfreq.mean()

            # Time masking (1-2 masks)
            n_time_masks = rng.integers(1, 3)
            for _ in range(n_time_masks):
                t_width_fine = rng.integers(1, min(self.time_mask_width + 1, T_fine))
                t_start_fine = rng.integers(0, max(T_fine - t_width_fine, 1))
                fill_val = mel_fine.mean()
                mel_fine[:, t_start_fine:t_start_fine + t_width_fine] = fill_val

                t_width_coarse = max(t_width_fine // 2, 1)
                t_start_coarse = min(t_start_fine // 2, max(T_coarse - t_width_coarse, 0))
                mel_coarse[:, t_start_coarse:t_start_coarse + t_width_coarse] = fill_val

                if mel_lowfreq is not None:
                    mel_lowfreq[:, t_start_coarse:t_start_coarse + t_width_coarse] = mel_lowfreq.mean()

        return mel_fine, mel_coarse, mel_lowfreq

    def _apply_clean_mask(self, weights: np.ndarray) -> np.ndarray:
        """Zero out weights for excluded onsets if clean_mask is set."""
        if self.clean_mask is not None:
            weights = weights * self.clean_mask.astype(weights.dtype)
        return weights

    def get_sampler(self, num_samples: int | None = None) -> WeightedRandomSampler:
        """Class-balanced sampler: equal probability for all 8 classes."""
        # Vectorized: build inverse-frequency lookup, index by primary class
        inv_freq = np.array([1.0 / max(self.class_counts.get(c, 1), 1)
                             for c in range(8)], dtype=np.float64)
        weights = inv_freq[self._primary_classes[:self.total]]
        weights = self._apply_clean_mask(weights)

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=num_samples or len(self),
            replacement=True,
        )

    def get_tom_boosted_sampler(self, tom_fraction: float = 0.5, num_samples: int | None = None) -> WeightedRandomSampler:
        """Sampler that ensures tom-positive examples dominate each batch.

        Args:
            tom_fraction: Target fraction of batch that should be tom-positive.
                Tom-positive = any of HighTom (3), LowTom (5), FloorTom (7) active.
        """
        TOM_INDICES = [3, 5, 7]
        labels = self.labels[:self.total]

        # Identify tom-positive examples
        is_tom = np.zeros(self.total, dtype=bool)
        for ti in TOM_INDICES:
            is_tom |= (labels[:, ti] > 0.5)

        # Vectorized inverse-frequency weights
        inv_freq = np.array([1.0 / max(self.class_counts.get(c, 1), 1)
                             for c in range(8)], dtype=np.float64)
        all_weights = inv_freq[self._primary_classes[:self.total]]

        tom_w = np.where(is_tom, all_weights, 0.0)
        nontom_w = np.where(~is_tom, all_weights, 0.0)

        # Normalize so tom group gets exactly tom_fraction of total weight
        tom_total = tom_w.sum()
        nontom_total = nontom_w.sum()
        if tom_total > 0:
            tom_w *= tom_fraction / tom_total
        if nontom_total > 0:
            nontom_w *= (1.0 - tom_fraction) / nontom_total
        weights = tom_w + nontom_w
        weights = self._apply_clean_mask(weights)

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=num_samples or len(self),
            replacement=True,
        )

    def get_hard_negative_sampler(
        self, confusion_pairs: Optional[dict[tuple[int, int], float]] = None,
        num_samples: int | None = None,
    ) -> WeightedRandomSampler:
        """Enhanced sampler upweighting confused class pairs."""
        if confusion_pairs is None:
            return self.get_sampler(num_samples=num_samples)

        # Vectorized: build per-class boost from confusion pairs
        class_boost = np.ones(8, dtype=np.float64)
        for (true_cls, pred_cls), rate in confusion_pairs.items():
            class_boost[true_cls] = max(class_boost[true_cls], 1.0 + rate * 2.0)

        inv_freq = np.array([1.0 / max(self.class_counts.get(c, 1), 1)
                             for c in range(8)], dtype=np.float64)
        cls_weights = inv_freq * class_boost  # shape (8,)

        weights = cls_weights[self._primary_classes[:self.total]]
        weights = self._apply_clean_mask(weights)

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=num_samples or len(self),
            replacement=True,
        )
