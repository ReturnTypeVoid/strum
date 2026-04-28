#!/usr/bin/env python3
"""
Pre-extract onset windows from drum stems into memory-mapped cache files.

Two-phase approach for minimal memory usage (~300MB peak vs 93GB before):
  Phase 1: Count valid onsets per song (fast, header-only audio reads)
  Phase 2: Pre-allocate memmap .npy files, extract mels, write at offsets

Output files in {output_dir}/cache/:
  {split}_mel_fine.npy    - (N, 128, 44) float16, memory-mappable
  {split}_mel_coarse.npy  - (N, 128, 22) float16, memory-mappable
  {split}_labels.npy      - (N, 8) uint8
  {split}_contexts.npy    - (N, 64) float16
  {split}_index.json      - metadata (counts, class distribution)
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.onset_classifier_dataset import LANE_CYMBAL_TO_CLASS, CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Fixed dimensions (must match config & model expectations)
SAMPLE_RATE = 44100
N_MELS = 128
FINE_N_FFT = 1024
FINE_HOP = 256
COARSE_N_FFT = 4096
COARSE_HOP = 512
# Low-frequency focused mel: 30-2000 Hz with 128 bins
# Gives ~6x more resolution in the tom/kick/snare fundamental range
LOWFREQ_N_FFT = 4096
LOWFREQ_HOP = 512
LOWFREQ_FMIN = 30.0
LOWFREQ_FMAX = 2000.0
WINDOW_BEFORE_MS = 100.0
WINDOW_AFTER_MS = 400.0
CONTEXT_SIZE = 4

WINDOW_BEFORE_SAMP = int(WINDOW_BEFORE_MS / 1000 * SAMPLE_RATE)
WINDOW_AFTER_SAMP = int(WINDOW_AFTER_MS / 1000 * SAMPLE_RATE)
WINDOW_SAMPLES = WINDOW_BEFORE_SAMP + WINDOW_AFTER_SAMP
FINE_FRAMES = WINDOW_SAMPLES // FINE_HOP + 1   # 87
COARSE_FRAMES = WINDOW_SAMPLES // COARSE_HOP + 1  # 44
LOWFREQ_FRAMES = WINDOW_SAMPLES // LOWFREQ_HOP + 1  # 44 (same as coarse)


def parse_onsets(labels_path: Path) -> list[tuple[float, set[int]]]:
    """Parse label JSON into grouped onset list: [(time_ms, {class_ids}), ...]."""
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
    """Phase 1: Fast counting pass. Returns (total_count, song_info_list)."""
    logger.info("Phase 1: Counting valid onsets (header-only, no mel computation)...")
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


def phase2_extract(
    song_info: list,
    data_dir: Path,
    total_onsets: int,
    cache_dir: Path,
    split: str,
) -> dict:
    """Phase 2: Extract mels and write directly to pre-allocated memmap files."""
    logger.info(f"Phase 2: Extracting {total_onsets} onset windows to memmap...")

    # Pre-allocate memmap files
    mf_path = cache_dir / f"{split}_mel_fine.npy"
    mc_path = cache_dir / f"{split}_mel_coarse.npy"
    ml_path = cache_dir / f"{split}_mel_lowfreq.npy"
    cf_path = cache_dir / f"{split}_crash_flux.npy"
    lb_path = cache_dir / f"{split}_labels.npy"
    ctx_path = cache_dir / f"{split}_contexts.npy"

    mf_shape = (total_onsets, N_MELS, FINE_FRAMES)
    mc_shape = (total_onsets, N_MELS, COARSE_FRAMES)
    ml_shape = (total_onsets, N_MELS, LOWFREQ_FRAMES)
    cf_shape = (total_onsets, 2, FINE_FRAMES)  # crash-band energy + flux
    lb_shape = (total_onsets, 8)
    ctx_shape = (total_onsets, 2 * CONTEXT_SIZE * 8)

    # Create empty .npy files with headers, then memmap them
    for path, shape, dtype in [
        (mf_path, mf_shape, np.float16),
        (mc_path, mc_shape, np.float16),
        (ml_path, ml_shape, np.float16),
        (cf_path, cf_shape, np.float16),
        (lb_path, lb_shape, np.uint8),
        (ctx_path, ctx_shape, np.float16),
    ]:
        fp = np.lib.format.open_memmap(str(path), mode='w+', dtype=dtype, shape=shape)
        del fp  # flush

    mm_fine = np.lib.format.open_memmap(str(mf_path), mode='r+')
    mm_coarse = np.lib.format.open_memmap(str(mc_path), mode='r+')
    mm_lowfreq = np.lib.format.open_memmap(str(ml_path), mode='r+')
    mm_crash_flux = np.lib.format.open_memmap(str(cf_path), mode='r+')
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
    lowfreq_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=LOWFREQ_N_FFT, hop_length=LOWFREQ_HOP,
        n_mels=N_MELS, f_min=LOWFREQ_FMIN, f_max=LOWFREQ_FMAX, power=2.0,
    )

    offset = 0
    class_counts = Counter()
    failed_songs = 0

    for si, sinfo in enumerate(song_info):
        if si % 100 == 0:
            logger.info(f"  [{si}/{len(song_info)}] {offset}/{total_onsets} onsets written...")

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

        for i, (time_ms, classes) in enumerate(onset_list):
            center = int(time_ms / 1000 * SAMPLE_RATE)
            start = center - WINDOW_BEFORE_SAMP

            if start < 0 or center + WINDOW_AFTER_SAMP > len(audio):
                continue
            if offset >= total_onsets:
                break

            # Audio window
            window = audio[start:start + WINDOW_SAMPLES]
            if len(window) < WINDOW_SAMPLES:
                window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))

            # Triple-resolution mel
            audio_t = torch.from_numpy(window).float().unsqueeze(0)
            with torch.no_grad():
                mf = torch.log(fine_mel(audio_t) + 1e-8).squeeze(0)
                mc = torch.log(coarse_mel(audio_t) + 1e-8).squeeze(0)
                ml = torch.log(lowfreq_mel(audio_t) + 1e-8).squeeze(0)

            # Pad/trim to fixed size
            if mf.shape[-1] > FINE_FRAMES:
                mf = mf[:, :FINE_FRAMES]
            elif mf.shape[-1] < FINE_FRAMES:
                mf = torch.nn.functional.pad(mf, (0, FINE_FRAMES - mf.shape[-1]),
                                             value=mf.min().item())
            if mc.shape[-1] > COARSE_FRAMES:
                mc = mc[:, :COARSE_FRAMES]
            elif mc.shape[-1] < COARSE_FRAMES:
                mc = torch.nn.functional.pad(mc, (0, COARSE_FRAMES - mc.shape[-1]),
                                             value=mc.min().item())
            if ml.shape[-1] > LOWFREQ_FRAMES:
                ml = ml[:, :LOWFREQ_FRAMES]
            elif ml.shape[-1] < LOWFREQ_FRAMES:
                ml = torch.nn.functional.pad(ml, (0, LOWFREQ_FRAMES - ml.shape[-1]),
                                             value=ml.min().item())

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

            # Write directly to memmap
            mm_fine[offset] = mf.numpy().astype(np.float16)
            mm_coarse[offset] = mc.numpy().astype(np.float16)
            mm_lowfreq[offset] = ml.numpy().astype(np.float16)
            mm_labels[offset] = label
            mm_ctx[offset] = context

            # Crash-band spectral flux (derived from fine mel, bins 80-127)
            crash_band = mf[80:128, :]  # (48, FINE_FRAMES)
            crash_energy = crash_band.mean(dim=0)  # (FINE_FRAMES,)
            crash_diff = torch.diff(crash_energy, prepend=crash_energy[:1])
            crash_flux = torch.relu(crash_diff)
            mm_crash_flux[offset, 0, :] = crash_energy.numpy().astype(np.float16)
            mm_crash_flux[offset, 1, :] = crash_flux.numpy().astype(np.float16)

            primary = min(classes)
            class_counts[primary] += 1
            offset += 1

    # Flush memmaps
    del mm_fine, mm_coarse, mm_lowfreq, mm_crash_flux, mm_labels, mm_ctx

    actual_count = offset
    logger.info(f"\n  Written: {actual_count}/{total_onsets} onsets "
                f"(failed songs: {failed_songs})")

    # If actual < allocated, truncate by rewriting headers
    if actual_count < total_onsets:
        logger.info(f"  Truncating memmap files from {total_onsets} to {actual_count}...")
        for path, orig_shape, dtype in [
            (mf_path, mf_shape, np.float16),
            (mc_path, mc_shape, np.float16),
            (ml_path, ml_shape, np.float16),
            (cf_path, cf_shape, np.float16),
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

    # Phase 1: Count
    total_onsets, song_info = phase1_count(songs, data_dir)
    if total_onsets == 0:
        logger.warning("No onsets found!")
        return

    # Disk estimate
    bytes_per_onset = (N_MELS * FINE_FRAMES * 2 + N_MELS * COARSE_FRAMES * 2
                       + N_MELS * LOWFREQ_FRAMES * 2
                       + 8 + 2 * CONTEXT_SIZE * 8 * 2)
    disk_gb = total_onsets * bytes_per_onset / 1e9
    logger.info(f"  Estimated disk usage: {disk_gb:.1f} GB")

    # Phase 2: Extract
    result = phase2_extract(song_info, data_dir, total_onsets, cache_dir, split)

    # Save index
    index = {
        "split": split,
        "total_onsets": result["actual_count"],
        "allocated_onsets": total_onsets,
        "class_counts": result["class_counts"],
        "fine_frames": FINE_FRAMES,
        "coarse_frames": COARSE_FRAMES,
        "lowfreq_frames": LOWFREQ_FRAMES,
        "n_mels": N_MELS,
        "sample_rate": SAMPLE_RATE,
        "window_before_ms": WINDOW_BEFORE_MS,
        "window_after_ms": WINDOW_AFTER_MS,
        "context_size": CONTEXT_SIZE,
        "files": {
            "mel_fine": f"{split}_mel_fine.npy",
            "mel_coarse": f"{split}_mel_coarse.npy",
            "mel_lowfreq": f"{split}_mel_lowfreq.npy",
            "crash_flux": f"{split}_crash_flux.npy",
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

    # File sizes
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

    cache_dir = Path(args.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        t0 = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Extracting {split} onset windows...")
        logger.info(f"{'='*60}")
        extract_split(Path(args.manifest), split, cache_dir)
        logger.info(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
