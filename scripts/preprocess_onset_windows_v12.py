#!/usr/bin/env python3
"""
V12 onset window preprocessing with CQT + longer windows.

Changes from V1 preprocessor:
  1. Longer window: 100ms before + 600ms after = 700ms (was 500ms)
     → captures tom sustain/decay characteristics
  2. CQT (Constant-Q Transform) replaces low-freq mel:
     30-2000 Hz, 24 bins/octave = 144 bins
     → logarithmic frequency resolution gives ~6x better separation
       in the kick/tom fundamental range (40-300 Hz)
  3. Fine/coarse mels unchanged in params but have more frames due to longer window

Output files in cache_v12/:
  {split}_mel_fine.npy    - (N, 128, 121) float16  (was 87 frames)
  {split}_mel_coarse.npy  - (N, 128, 61) float16   (was 44 frames)
  {split}_cqt.npy         - (N, 144, 61) float16   (NEW: replaces mel_lowfreq)
  {split}_labels.npy      - (N, 8) uint8
  {split}_contexts.npy    - (N, 64) float16
  {split}_index.json      - metadata
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.onset_classifier_dataset import LANE_CYMBAL_TO_CLASS, CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Audio parameters ──────────────────────────────────────────
SAMPLE_RATE = 44100
N_MELS = 128

# Mel spectrogram params (unchanged from V1)
FINE_N_FFT = 1024
FINE_HOP = 256
COARSE_N_FFT = 4096
COARSE_HOP = 512

# CQT parameters (replaces low-freq mel)
CQT_FMIN = 30.0       # Lowest frequency (covers kick fundamentals)
CQT_BINS_PER_OCT = 24  # 24 bins/octave = ~3% frequency resolution
CQT_N_BINS = 144       # 6 octaves covers 30-1920 Hz
CQT_HOP = 512          # Same hop as coarse mel

# Longer window: captures tom sustain/decay (toms sustain 200-500ms)
WINDOW_BEFORE_MS = 100.0
WINDOW_AFTER_MS = 600.0   # Was 400ms → now 600ms
CONTEXT_SIZE = 4

WINDOW_BEFORE_SAMP = int(WINDOW_BEFORE_MS / 1000 * SAMPLE_RATE)
WINDOW_AFTER_SAMP = int(WINDOW_AFTER_MS / 1000 * SAMPLE_RATE)
WINDOW_SAMPLES = WINDOW_BEFORE_SAMP + WINDOW_AFTER_SAMP
FINE_FRAMES = WINDOW_SAMPLES // FINE_HOP + 1    # 121
COARSE_FRAMES = WINDOW_SAMPLES // COARSE_HOP + 1  # 61
CQT_FRAMES = WINDOW_SAMPLES // CQT_HOP + 1      # 61


def parse_onsets(labels_path: Path) -> list[tuple[float, set[int]]]:
    """Parse label JSON into grouped onset list."""
    with open(labels_path) as f:
        label_data = json.load(f)

    hits_by_time: dict[int, dict] = {}
    for hit in label_data["hits"]:
        t = hit["time_ms"]
        cls = LANE_CYMBAL_TO_CLASS.get((hit["lane"], hit["is_cymbal"]))
        if cls is None:
            continue
        bin_t = round(t / 5) * 5
        if bin_t not in hits_by_time:
            hits_by_time[bin_t] = {"time_ms": t, "classes": set()}
        hits_by_time[bin_t]["classes"].add(cls)

    sorted_times = sorted(hits_by_time.keys())
    return [(hits_by_time[t]["time_ms"], hits_by_time[t]["classes"])
            for t in sorted_times]


def count_valid_onsets(onset_list: list, audio_len_samples: int) -> int:
    """Count onsets whose window fits within the audio."""
    count = 0
    for time_ms, _ in onset_list:
        center = int(time_ms / 1000 * SAMPLE_RATE)
        if center - WINDOW_BEFORE_SAMP >= 0 and center + WINDOW_AFTER_SAMP <= audio_len_samples:
            count += 1
    return count


def phase1_count(songs: list[dict], data_dir: Path) -> tuple[int, list]:
    """Phase 1: Fast counting pass."""
    logger.info("Phase 1: Counting valid onsets...")
    total = 0
    song_info = []
    skipped = 0

    for i, song in enumerate(songs):
        if i % 200 == 0:
            logger.info(f"  [{i}/{len(songs)}] counted {total} onsets...")

        labels_path = data_dir / song["id"] / "drums_labels.json"
        audio_path = data_dir / song["stems"]["drums"]

        if not labels_path.exists() or not audio_path.exists():
            skipped += 1
            continue

        try:
            info = sf.info(str(audio_path))
            audio_len = int(info.frames)
            if info.samplerate != SAMPLE_RATE:
                audio_len = int(info.duration * SAMPLE_RATE)
        except Exception:
            skipped += 1
            continue

        onset_list = parse_onsets(labels_path)
        n = count_valid_onsets(onset_list, audio_len)

        if n > 0:
            song_info.append({
                "song": song,
                "onset_count": n,
                "onset_list": onset_list,
            })
            total += n
        else:
            skipped += 1

    logger.info(f"  Count complete: {total} onsets from {len(song_info)} songs "
                f"(skipped {skipped})")
    return total, song_info


def compute_full_song_cqt(audio_np: np.ndarray) -> np.ndarray:
    """Compute log-CQT for the full song audio. Returns (CQT_N_BINS, T) float32."""
    cqt = librosa.cqt(
        y=audio_np,
        sr=SAMPLE_RATE,
        hop_length=CQT_HOP,
        fmin=CQT_FMIN,
        n_bins=CQT_N_BINS,
        bins_per_octave=CQT_BINS_PER_OCT,
    )
    return np.log(np.abs(cqt) + 1e-8).astype(np.float32)


def slice_cqt_window(full_cqt: np.ndarray, start_sample: int) -> np.ndarray:
    """Slice a CQT window from the full-song CQT spectrogram.

    Args:
        full_cqt: (CQT_N_BINS, T_full) full-song CQT
        start_sample: window start in audio samples

    Returns:
        (CQT_N_BINS, CQT_FRAMES) log-CQT slice
    """
    start_frame = start_sample // CQT_HOP
    end_frame = start_frame + CQT_FRAMES
    total_frames = full_cqt.shape[1]

    if end_frame <= total_frames:
        cqt_slice = full_cqt[:, start_frame:end_frame]
    else:
        # Pad at the end if needed
        available = full_cqt[:, start_frame:total_frames]
        pad_width = CQT_FRAMES - available.shape[1]
        cqt_slice = np.pad(available, ((0, 0), (0, pad_width)),
                           mode='constant', constant_values=available.min())

    # Ensure exact frame count
    if cqt_slice.shape[1] > CQT_FRAMES:
        cqt_slice = cqt_slice[:, :CQT_FRAMES]
    return cqt_slice


def phase2_extract(
    song_info: list,
    data_dir: Path,
    total_onsets: int,
    cache_dir: Path,
    split: str,
) -> dict:
    """Phase 2: Extract features and write to memmap files."""
    logger.info(f"Phase 2: Extracting {total_onsets} onset windows to memmap...")

    # Pre-allocate memmap files
    mf_path = cache_dir / f"{split}_mel_fine.npy"
    mc_path = cache_dir / f"{split}_mel_coarse.npy"
    cqt_path = cache_dir / f"{split}_cqt.npy"
    lb_path = cache_dir / f"{split}_labels.npy"
    ctx_path = cache_dir / f"{split}_contexts.npy"

    mf_shape = (total_onsets, N_MELS, FINE_FRAMES)
    mc_shape = (total_onsets, N_MELS, COARSE_FRAMES)
    cqt_shape = (total_onsets, CQT_N_BINS, CQT_FRAMES)
    lb_shape = (total_onsets, 8)
    ctx_shape = (total_onsets, 2 * CONTEXT_SIZE * 8)

    # Create empty .npy files
    for path, shape, dtype in [
        (mf_path, mf_shape, np.float16),
        (mc_path, mc_shape, np.float16),
        (cqt_path, cqt_shape, np.float16),
        (lb_path, lb_shape, np.uint8),
        (ctx_path, ctx_shape, np.float16),
    ]:
        fp = np.lib.format.open_memmap(str(path), mode='w+', dtype=dtype, shape=shape)
        del fp

    mm_fine = np.lib.format.open_memmap(str(mf_path), mode='r+')
    mm_coarse = np.lib.format.open_memmap(str(mc_path), mode='r+')
    mm_cqt = np.lib.format.open_memmap(str(cqt_path), mode='r+')
    mm_labels = np.lib.format.open_memmap(str(lb_path), mode='r+')
    mm_ctx = np.lib.format.open_memmap(str(ctx_path), mode='r+')

    # Mel transforms
    fine_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=FINE_N_FFT, hop_length=FINE_HOP,
        n_mels=N_MELS, power=2.0,
    )
    coarse_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=COARSE_N_FFT, hop_length=COARSE_HOP,
        n_mels=N_MELS, power=2.0,
    )

    offset = 0
    class_counts = Counter()
    failed_songs = 0
    t_start = time.time()

    for si, sinfo in enumerate(song_info):
        if si % 100 == 0:
            elapsed = time.time() - t_start
            rate = si / elapsed if elapsed > 0 else 0
            eta = (len(song_info) - si) / rate / 60 if rate > 0 else 0
            logger.info(f"  [{si}/{len(song_info)}] {offset}/{total_onsets} onsets "
                        f"({elapsed/60:.1f}min elapsed, ~{eta:.0f}min remaining)")

        song = sinfo["song"]
        onset_list = sinfo["onset_list"]
        audio_path = data_dir / song["stems"]["drums"]

        try:
            audio, sr = sf.read(str(audio_path), dtype='float32')
        except Exception:
            failed_songs += 1
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            failed_songs += 1
            continue

        if sr != SAMPLE_RATE:
            audio_t = torch.from_numpy(audio).float().unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, SAMPLE_RATE)
            audio = audio_t.squeeze(0).numpy()

        # Compute all spectrograms once for the entire song, then slice per onset
        full_cqt = compute_full_song_cqt(audio)

        audio_t = torch.from_numpy(audio).float().unsqueeze(0)
        with torch.no_grad():
            full_fine = torch.log(fine_mel(audio_t) + 1e-8).squeeze(0).numpy()   # (128, T_fine)
            full_coarse = torch.log(coarse_mel(audio_t) + 1e-8).squeeze(0).numpy()  # (128, T_coarse)

        for i, (time_ms, classes) in enumerate(onset_list):
            center = int(time_ms / 1000 * SAMPLE_RATE)
            start = center - WINDOW_BEFORE_SAMP

            if start < 0 or center + WINDOW_AFTER_SAMP > len(audio):
                continue
            if offset >= total_onsets:
                break

            # Slice mel_fine from full-song spectrogram
            fine_start = start // FINE_HOP
            fine_end = fine_start + FINE_FRAMES
            if fine_end <= full_fine.shape[1]:
                mf = full_fine[:, fine_start:fine_end]
            else:
                avail = full_fine[:, fine_start:full_fine.shape[1]]
                mf = np.pad(avail, ((0, 0), (0, FINE_FRAMES - avail.shape[1])),
                            mode='constant', constant_values=avail.min())

            # Slice mel_coarse from full-song spectrogram
            coarse_start = start // COARSE_HOP
            coarse_end = coarse_start + COARSE_FRAMES
            if coarse_end <= full_coarse.shape[1]:
                mc = full_coarse[:, coarse_start:coarse_end]
            else:
                avail = full_coarse[:, coarse_start:full_coarse.shape[1]]
                mc = np.pad(avail, ((0, 0), (0, COARSE_FRAMES - avail.shape[1])),
                            mode='constant', constant_values=avail.min())

            # CQT: slice from full-song CQT
            cqt = slice_cqt_window(full_cqt, start)
            # Context (neighbor one-hots)
            ctx_parts = []
            for off in list(range(-CONTEXT_SIZE, 0)) + list(range(1, CONTEXT_SIZE + 1)):
                idx = i + off
                if 0 <= idx < len(onset_list):
                    oh = np.zeros(8, dtype=np.float16)
                    for c in onset_list[idx][1]:
                        oh[c] = 1.0
                    ctx_parts.append(oh)
                else:
                    ctx_parts.append(np.zeros(8, dtype=np.float16))
            context = np.concatenate(ctx_parts)

            # Label (multi-hot)
            label = np.zeros(8, dtype=np.uint8)
            for c in classes:
                label[c] = 1

            # Write to memmap
            mm_fine[offset] = mf.astype(np.float16)
            mm_coarse[offset] = mc.astype(np.float16)
            mm_cqt[offset] = cqt.astype(np.float16)
            mm_labels[offset] = label
            mm_ctx[offset] = context

            primary = min(classes)
            class_counts[primary] += 1
            offset += 1

    # Flush memmaps
    del mm_fine, mm_coarse, mm_cqt, mm_labels, mm_ctx

    actual_count = offset
    logger.info(f"\n  Written: {actual_count}/{total_onsets} onsets "
                f"(failed songs: {failed_songs})")

    # Truncate if needed
    if actual_count < total_onsets:
        logger.info(f"  Truncating memmap files from {total_onsets} to {actual_count}...")
        for path, orig_shape, dtype in [
            (mf_path, mf_shape, np.float16),
            (mc_path, mc_shape, np.float16),
            (cqt_path, cqt_shape, np.float16),
            (lb_path, lb_shape, np.uint8),
            (ctx_path, ctx_shape, np.float16),
        ]:
            new_shape = (actual_count,) + orig_shape[1:]
            data = np.lib.format.open_memmap(str(path), mode='r+')[:actual_count].copy()
            fp = np.lib.format.open_memmap(str(path), mode='w+', dtype=dtype, shape=new_shape)
            fp[:] = data
            del fp, data

    return {"actual_count": actual_count, "class_counts": dict(class_counts)}


def extract_split(manifest_path: Path, split: str, cache_dir: Path):
    """Full extraction pipeline for one split."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    data_dir = manifest_path.parent
    songs = [s for s in manifest["songs"]
             if s["split"] == split and s["charts"].get("drums")]

    logger.info(f"Found {len(songs)} {split} songs with drum charts")

    total_onsets, song_info = phase1_count(songs, data_dir)
    if total_onsets == 0:
        logger.warning("No onsets found!")
        return

    # Disk estimate
    bytes_per_onset = (N_MELS * FINE_FRAMES * 2 + N_MELS * COARSE_FRAMES * 2
                       + CQT_N_BINS * CQT_FRAMES * 2
                       + 8 + 2 * CONTEXT_SIZE * 8 * 2)
    disk_gb = total_onsets * bytes_per_onset / 1e9
    logger.info(f"  Estimated disk usage: {disk_gb:.1f} GB")

    result = phase2_extract(song_info, data_dir, total_onsets, cache_dir, split)

    # Save index
    index = {
        "split": split,
        "total_onsets": result["actual_count"],
        "allocated_onsets": total_onsets,
        "class_counts": result["class_counts"],
        "fine_frames": FINE_FRAMES,
        "coarse_frames": COARSE_FRAMES,
        "cqt_frames": CQT_FRAMES,
        "cqt_bins": CQT_N_BINS,
        "n_mels": N_MELS,
        "sample_rate": SAMPLE_RATE,
        "window_before_ms": WINDOW_BEFORE_MS,
        "window_after_ms": WINDOW_AFTER_MS,
        "context_size": CONTEXT_SIZE,
        "files": {
            "mel_fine": f"{split}_mel_fine.npy",
            "mel_coarse": f"{split}_mel_coarse.npy",
            "cqt": f"{split}_cqt.npy",
            "labels": f"{split}_labels.npy",
            "contexts": f"{split}_contexts.npy",
        },
    }
    index_path = cache_dir / f"{split}_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"\nExtraction complete ({split}):")
    logger.info(f"  Total onsets: {result['actual_count']}")
    for i, name in enumerate(CLASS_NAMES):
        logger.info(f"  {name}: {result['class_counts'].get(str(i), result['class_counts'].get(i, 0))}")
    logger.info(f"  Index: {index_path}")

    for key, fname in index["files"].items():
        fpath = cache_dir / fname
        if fpath.exists():
            logger.info(f"  {fname}: {fpath.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/mnt/ml-data/dataset_drums/manifest.json")
    parser.add_argument("--output-dir", default="outputs/onset_classifier")
    parser.add_argument("--split", default="both", choices=["train", "test", "both"])
    args = parser.parse_args()

    cache_dir = Path(args.output_dir) / "cache_v12"
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"V12 Extracting {split} onset windows (CQT + 700ms window)...")
        logger.info(f"{'='*60}")
        extract_split(Path(args.manifest), split, cache_dir)
        logger.info(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
