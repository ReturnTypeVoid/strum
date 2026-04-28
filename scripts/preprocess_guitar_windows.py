#!/usr/bin/env python3
"""Preprocess guitar training data into memory-mapped caches.

Reads `configs/guitar_v1_manifest.json` and produces two caches per split,
sharing a single audio-load pass per song:

  STAGE 1 — Onset CRNN (frame-level binary onset detection)
    {cache}/onset/{split}_audio_segments.npy  (N_seg, n_mels, T_frames)  fp16
    {cache}/onset/{split}_onset_labels.npy    (N_seg, T_frames)          uint8
    {cache}/onset/{split}_segment_meta.json   metadata

  STAGE 2 — Fret Classifier (per-onset window → 5-bit multi-label)
    {cache}/fret/{split}_mel_window.npy       (N_win, n_mels, W_frames)  fp16
    {cache}/fret/{split}_fret_labels.npy      (N_win, 5)                 uint8
    {cache}/fret/{split}_chord_flag.npy       (N_win,)                   uint8
    {cache}/fret/{split}_window_meta.json     metadata

Audio is normalized to mono at 22050 Hz (matches inference). Mel is
log-power, n_fft=2048, hop=512 → 23.2 ms/frame.

Usage:
    python scripts/preprocess_guitar_windows.py \
        --manifest configs/guitar_v1_manifest.json \
        --cache-dir /mnt/ml-data/guitar_v1_cache \
        --splits train val test \
        [--limit-songs N]   # debug
        [--no-onset-cache | --no-fret-cache]
        [--workers N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("guitar_preprocess")

# ─────────────────────────── Audio / mel constants ──────────────────────────
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 30.0
FMAX = 8000.0
FRAME_MS = 1000.0 * HOP_LENGTH / SAMPLE_RATE   # 23.22 ms
FRAMES_PER_SEC = SAMPLE_RATE / HOP_LENGTH       # 43.07

# ─────────────────────────── Stage 1: segments ──────────────────────────────
# 5-second segments with 50% overlap → matches OnsetCRNN config in
# train_guitar_onset.py (segment_duration_sec=5.0, overlap=0.5).
SEG_DURATION_SEC = 5.0
SEG_OVERLAP = 0.5
SEG_FRAMES = int(SEG_DURATION_SEC * FRAMES_PER_SEC) + 1   # 216
SEG_HOP_FRAMES = int(SEG_FRAMES * (1 - SEG_OVERLAP))      # 108
SEG_HOP_SAMPLES = int(SEG_HOP_FRAMES * HOP_LENGTH)
SEG_SAMPLES = (SEG_FRAMES - 1) * HOP_LENGTH

# Smear the binary onset target ±1 frame → soft target stabilises training
ONSET_SMEAR_FRAMES = 1

# ─────────────────────────── Stage 2: windows ───────────────────────────────
# 100 ms before, 400 ms after the onset (same as drums Stage 2).
WIN_BEFORE_MS = 100.0
WIN_AFTER_MS = 400.0
WIN_BEFORE_SAMP = int(WIN_BEFORE_MS / 1000.0 * SAMPLE_RATE)
WIN_AFTER_SAMP = int(WIN_AFTER_MS / 1000.0 * SAMPLE_RATE)
WIN_SAMPLES = WIN_BEFORE_SAMP + WIN_AFTER_SAMP
WIN_FRAMES = WIN_SAMPLES // HOP_LENGTH + 1                # 22

# Group simultaneous Expert notes into one onset (mirrors manifest builder)
CHORD_GROUP_MS = 25.0


# ─────────────────────────── Helpers ────────────────────────────────────────
def load_audio_mono_22050(path: Path) -> Optional[np.ndarray]:
    """Read any sf-supported audio → mono float32 @ 22050 Hz."""
    try:
        audio, sr = sf.read(str(path), dtype="float32")
    except Exception as exc:
        log.debug("audio read failed for %s: %s", path, exc)
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) == 0:
        return None
    if sr != SAMPLE_RATE:
        t = torch.from_numpy(audio).float().unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)
        audio = t.squeeze(0).numpy()
    return audio


_MEL: torchaudio.transforms.MelSpectrogram | None = None


def get_mel() -> torchaudio.transforms.MelSpectrogram:
    global _MEL
    if _MEL is None:
        _MEL = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=FMIN,
            f_max=FMAX,
            power=2.0,
        )
    return _MEL


def compute_log_mel(audio: np.ndarray) -> torch.Tensor:
    """Returns (n_mels, T) log-mel."""
    with torch.no_grad():
        x = torch.from_numpy(audio).float().unsqueeze(0)
        m = get_mel()(x).squeeze(0)              # (n_mels, T)
        m = torch.log(m + 1e-8)
    return m


def parse_onsets_from_manifest(midi_path: Path) -> list[tuple[float, set[int]]]:
    """Re-parse Expert PART GUITAR onsets from notes.mid (cheaper than
    re-running build_guitar_manifest, but uses identical logic)."""
    import mido
    EXPERT = {96: 0, 97: 1, 98: 2, 99: 3, 100: 4}
    OPEN = 95

    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception:
        return []

    track = None
    for tr in mid.tracks:
        if tr.name == "PART GUITAR":
            track = tr
            break
    if track is None:
        return []

    # Tempo map
    tempo_changes: list[tuple[int, int]] = [(0, 500_000)]
    for tr in mid.tracks:
        tk = 0
        for msg in tr:
            tk += msg.time
            if msg.type == "set_tempo":
                tempo_changes.append((tk, msg.tempo))
    tempo_changes.sort()
    dedup = [tempo_changes[0]]
    for tk, tm in tempo_changes[1:]:
        if tk == dedup[-1][0]:
            dedup[-1] = (tk, tm)
        else:
            dedup.append((tk, tm))
    tempo_changes = dedup
    tpb = mid.ticks_per_beat

    def t2ms(target: int) -> float:
        ms = 0.0
        last_t = 0
        last_tempo = tempo_changes[0][1]
        for et, em in tempo_changes[1:]:
            if et >= target:
                break
            ms += (et - last_t) * last_tempo / tpb / 1000.0
            last_t, last_tempo = et, em
        ms += (target - last_t) * last_tempo / tpb / 1000.0
        return ms

    raw: list[tuple[float, int]] = []
    tk = 0
    for msg in track:
        tk += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.note in EXPERT:
                raw.append((t2ms(tk), EXPERT[msg.note]))
            elif msg.note == OPEN:
                raw.append((t2ms(tk), -1))

    if not raw:
        return []
    raw.sort()
    out: list[tuple[float, set[int]]] = []
    cur_t, cur = raw[0][0], {raw[0][1]}
    for t, f in raw[1:]:
        if t - cur_t <= CHORD_GROUP_MS:
            cur.add(f)
        else:
            out.append((cur_t, cur))
            cur_t, cur = t, {f}
    out.append((cur_t, cur))
    return out


# ─────────────────────────── Phase counters ─────────────────────────────────
def count_segments(audio_len: int) -> int:
    """How many overlapping segments fit in this audio."""
    if audio_len < SEG_SAMPLES:
        return 0
    return 1 + (audio_len - SEG_SAMPLES) // SEG_HOP_SAMPLES


def count_valid_windows(onsets: list[tuple[float, set[int]]],
                        audio_len: int) -> int:
    n = 0
    for tms, _ in onsets:
        center = int(tms / 1000.0 * SAMPLE_RATE)
        if center - WIN_BEFORE_SAMP >= 0 and center + WIN_AFTER_SAMP <= audio_len:
            n += 1
    return n


# ─────────────────────────── Pre-flight (Phase 1) ───────────────────────────
def phase1(songs: list[dict]) -> tuple[list[dict], int, int]:
    """Cheap header-only pass to size memmaps."""
    info = []
    total_segs = 0
    total_wins = 0
    skipped = 0
    for song in tqdm(songs, desc="size"):
        audio_path = Path(song["audio_path"])
        midi_path = Path(song["midi_path"])
        if not audio_path.exists() or not midi_path.exists():
            skipped += 1
            continue
        try:
            ainfo = sf.info(str(audio_path))
            # Account for resample to 22050
            audio_len = int(ainfo.duration * SAMPLE_RATE)
        except Exception:
            skipped += 1
            continue
        onsets = parse_onsets_from_manifest(midi_path)
        if not onsets:
            skipped += 1
            continue

        n_seg = count_segments(audio_len)
        n_win = count_valid_windows(onsets, audio_len)
        if n_seg == 0 and n_win == 0:
            skipped += 1
            continue

        info.append({
            "song": song,
            "audio_len_samples": audio_len,
            "onsets": onsets,
            "n_segments": n_seg,
            "n_windows": n_win,
        })
        total_segs += n_seg
        total_wins += n_win

    log.info("phase1: %d songs / %d segments / %d windows (skipped %d)",
             len(info), total_segs, total_wins, skipped)
    return info, total_segs, total_wins


# ─────────────────────────── Extract (Phase 2) ──────────────────────────────
def phase2_extract(
    song_info: list[dict],
    cache_dir: Path,
    split: str,
    total_segs: int,
    total_wins: int,
    do_onset: bool,
    do_fret: bool,
) -> dict:
    onset_dir = cache_dir / "onset"
    fret_dir = cache_dir / "fret"
    onset_dir.mkdir(parents=True, exist_ok=True)
    fret_dir.mkdir(parents=True, exist_ok=True)

    # ─── Allocate memmaps
    if do_onset:
        seg_audio_path = onset_dir / f"{split}_segments_mel.npy"
        seg_label_path = onset_dir / f"{split}_segments_onset.npy"
        seg_audio = np.lib.format.open_memmap(
            str(seg_audio_path), mode="w+", dtype=np.float16,
            shape=(total_segs, N_MELS, SEG_FRAMES),
        )
        seg_label = np.lib.format.open_memmap(
            str(seg_label_path), mode="w+", dtype=np.float16,
            shape=(total_segs, SEG_FRAMES),
        )
    if do_fret:
        win_mel_path = fret_dir / f"{split}_window_mel.npy"
        win_fret_path = fret_dir / f"{split}_window_fret.npy"
        win_meta_path = fret_dir / f"{split}_window_chord.npy"
        win_mel = np.lib.format.open_memmap(
            str(win_mel_path), mode="w+", dtype=np.float16,
            shape=(total_wins, N_MELS, WIN_FRAMES),
        )
        win_fret = np.lib.format.open_memmap(
            str(win_fret_path), mode="w+", dtype=np.uint8,
            shape=(total_wins, 5),
        )
        win_chord = np.lib.format.open_memmap(
            str(win_meta_path), mode="w+", dtype=np.uint8,
            shape=(total_wins,),
        )

    seg_off = 0
    win_off = 0
    failed = 0
    fret_counts = np.zeros(5, dtype=np.int64)
    chord_count = 0

    for sinfo in tqdm(song_info, desc=f"extract[{split}]"):
        song = sinfo["song"]
        audio_path = Path(song["audio_path"])
        audio = load_audio_mono_22050(audio_path)
        if audio is None:
            failed += 1
            continue

        full_mel = compute_log_mel(audio)            # (n_mels, T)
        T_full = full_mel.shape[-1]

        # ── Onset frame target (full-song)
        onset_frames_target = None
        if do_onset:
            onset_frames_target = np.zeros(T_full, dtype=np.float32)
            for tms, _ in sinfo["onsets"]:
                fr = int(round(tms / 1000.0 * FRAMES_PER_SEC))
                lo = max(0, fr - ONSET_SMEAR_FRAMES)
                hi = min(T_full, fr + ONSET_SMEAR_FRAMES + 1)
                onset_frames_target[lo:hi] = 1.0

        # ── Stage 1: write overlapping segments
        if do_onset and sinfo["n_segments"] > 0:
            for s_idx in range(sinfo["n_segments"]):
                fstart = s_idx * SEG_HOP_FRAMES
                fend = fstart + SEG_FRAMES
                if fend > T_full:
                    break
                seg = full_mel[:, fstart:fend]                  # (n_mels, SEG_FRAMES)
                lab = onset_frames_target[fstart:fend]          # (SEG_FRAMES,)
                if seg_off >= total_segs:
                    break
                seg_audio[seg_off] = seg.numpy().astype(np.float16)
                seg_label[seg_off] = lab.astype(np.float16)
                seg_off += 1

        # ── Stage 2: per-onset windows
        if do_fret:
            for tms, frets in sinfo["onsets"]:
                if win_off >= total_wins:
                    break
                center = int(tms / 1000.0 * SAMPLE_RATE)
                if center - WIN_BEFORE_SAMP < 0 or center + WIN_AFTER_SAMP > len(audio):
                    continue
                fstart = (center - WIN_BEFORE_SAMP) // HOP_LENGTH
                fend = fstart + WIN_FRAMES
                if fend > T_full:
                    continue
                win = full_mel[:, fstart:fend]
                if win.shape[-1] != WIN_FRAMES:
                    continue
                win_mel[win_off] = win.numpy().astype(np.float16)

                playable = {f for f in frets if 0 <= f <= 4}
                lab = np.zeros(5, dtype=np.uint8)
                for f in playable:
                    lab[f] = 1
                    fret_counts[f] += 1
                win_fret[win_off] = lab
                is_chord = 1 if len(playable) >= 2 else 0
                win_chord[win_off] = is_chord
                if is_chord:
                    chord_count += 1
                win_off += 1

    if do_onset:
        del seg_audio, seg_label
    if do_fret:
        del win_mel, win_fret, win_chord

    # Truncate if we wrote fewer than allocated
    if do_onset and seg_off < total_segs:
        _truncate(onset_dir / f"{split}_segments_mel.npy",
                  (seg_off, N_MELS, SEG_FRAMES), np.float16)
        _truncate(onset_dir / f"{split}_segments_onset.npy",
                  (seg_off, SEG_FRAMES), np.float16)
    if do_fret and win_off < total_wins:
        _truncate(fret_dir / f"{split}_window_mel.npy",
                  (win_off, N_MELS, WIN_FRAMES), np.float16)
        _truncate(fret_dir / f"{split}_window_fret.npy",
                  (win_off, 5), np.uint8)
        _truncate(fret_dir / f"{split}_window_chord.npy",
                  (win_off,), np.uint8)

    meta = {
        "split": split,
        "n_segments": int(seg_off),
        "n_windows": int(win_off),
        "failed_songs": int(failed),
        "fret_counts": fret_counts.tolist(),
        "chord_windows": int(chord_count),
        "single_windows": int(win_off - chord_count),
        "constants": {
            "sample_rate": SAMPLE_RATE,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "fmin": FMIN, "fmax": FMAX,
            "frame_ms": FRAME_MS,
            "seg_frames": SEG_FRAMES,
            "seg_overlap": SEG_OVERLAP,
            "win_before_ms": WIN_BEFORE_MS,
            "win_after_ms": WIN_AFTER_MS,
            "win_frames": WIN_FRAMES,
            "onset_smear_frames": ONSET_SMEAR_FRAMES,
        },
    }
    if do_onset:
        with (onset_dir / f"{split}_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
    if do_fret:
        with (fret_dir / f"{split}_meta.json").open("w") as f:
            json.dump(meta, f, indent=2)
    log.info("split=%s: wrote %d segments / %d windows  fret_counts=%s",
             split, seg_off, win_off, fret_counts.tolist())
    return meta


def _truncate(path: Path, new_shape: tuple, dtype) -> None:
    data = np.lib.format.open_memmap(str(path), mode="r+")[:new_shape[0]].copy()
    fp = np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=new_shape)
    fp[:] = data
    del fp


# ─────────────────────────── Main ───────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/guitar_v1_manifest.json")
    ap.add_argument("--cache-dir", default="/mnt/ml-data/guitar_v1_cache")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test"])
    ap.add_argument("--limit-songs", type=int, default=0,
                    help="Per-split cap (debug)")
    ap.add_argument("--no-onset-cache", action="store_true")
    ap.add_argument("--no-fret-cache", action="store_true")
    args = ap.parse_args()

    do_onset = not args.no_onset_cache
    do_fret = not args.no_fret_cache
    if not do_onset and not do_fret:
        log.error("nothing to do (both caches disabled)")
        return 2

    manifest = json.loads(Path(args.manifest).read_text())
    all_songs = manifest["songs"]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("manifest: %d songs total", len(all_songs))

    overall: dict[str, dict] = {}
    for split in args.splits:
        songs = [s for s in all_songs if s["split"] == split]
        if args.limit_songs:
            songs = songs[: args.limit_songs]
        log.info("=== split=%s : %d songs ===", split, len(songs))
        if not songs:
            continue
        t0 = time.time()
        info, total_segs, total_wins = phase1(songs)
        if not do_onset:
            total_segs = 0
        if not do_fret:
            total_wins = 0
        overall[split] = phase2_extract(
            info, cache_dir, split, total_segs, total_wins, do_onset, do_fret,
        )
        log.info("split=%s done in %.1fs", split, time.time() - t0)

    # Write overall summary
    summary_path = cache_dir / "preprocess_summary.json"
    with summary_path.open("w") as f:
        json.dump({"per_split": overall}, f, indent=2)
    log.info("summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
