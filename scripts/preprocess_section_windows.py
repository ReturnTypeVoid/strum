"""
Preprocess section-classifier windows.

Reads configs/guitar_section_labels.json and extracts a 2-s log-mel patch per
window from each song's audio, saving to a memmap cache.

Output:
    {cache}/{split}_section_mel.npy   (N, n_mels, T)  fp16
    {cache}/{split}_section_label.npy (N,)            int8
    {cache}/{split}_section_meta.json (song_id, t_start_s for each)

Run:
    python scripts/preprocess_section_windows.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from preprocess_guitar_windows import (  # noqa: E402
    HOP_LENGTH, N_MELS, SAMPLE_RATE, compute_log_mel,
    load_audio_mono_22050 as load_audio_mono22k,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("section_pre")

LABELS = ["silence", "constant_strum", "chord_stab", "lead_line", "single_notes", "mixed"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}

WINDOW_S = 2.0
WINDOW_SAMPLES = int(WINDOW_S * SAMPLE_RATE)
WINDOW_FRAMES = WINDOW_SAMPLES // HOP_LENGTH + 1   # ~87


def process_split(records: list[dict], split: str, cache_dir: Path) -> None:
    # Group by audio path so we load each file once
    by_audio: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["split"] != split:
            continue
        by_audio[r["audio_path"]].append(r)

    if not by_audio:
        log.warning("no records for split=%s", split)
        return

    n_total = sum(len(v) for v in by_audio.values())
    log.info("split=%s songs=%d windows=%d", split, len(by_audio), n_total)

    mel_path = cache_dir / f"{split}_section_mel.npy"
    lab_path = cache_dir / f"{split}_section_label.npy"
    meta_path = cache_dir / f"{split}_section_meta.json"

    mel_mm = np.lib.format.open_memmap(
        mel_path, mode="w+", dtype=np.float16, shape=(n_total, N_MELS, WINDOW_FRAMES),
    )
    lab_mm = np.zeros(n_total, dtype=np.int8)
    meta: list[dict] = []

    cur = 0
    for ai, (audio_path, recs) in enumerate(by_audio.items()):
        if ai % 100 == 0:
            log.info("[%d/%d] %s", ai, len(by_audio), Path(audio_path).name)
        try:
            audio = load_audio_mono22k(Path(audio_path))
        except Exception as exc:
            log.warning("load failed %s: %s", audio_path, exc)
            continue
        if audio is None:
            continue

        for r in recs:
            t_start = float(r["t_start_s"])
            s = int(t_start * SAMPLE_RATE)
            e = s + WINDOW_SAMPLES
            if e > len(audio):
                continue
            patch = audio[s:e]
            mel = compute_log_mel(patch).numpy()  # (n_mels, T)
            if mel.shape[1] < WINDOW_FRAMES:
                pad = WINDOW_FRAMES - mel.shape[1]
                mel = np.pad(mel, ((0, 0), (0, pad)), mode="edge")
            elif mel.shape[1] > WINDOW_FRAMES:
                mel = mel[:, :WINDOW_FRAMES]
            mel_mm[cur] = mel.astype(np.float16)
            lab_mm[cur] = LABEL_TO_IDX[r["label"]]
            meta.append({
                "song_id": r["song_id"],
                "t_start_s": t_start,
                "label": r["label"],
            })
            cur += 1

    # Trim arrays to actual size
    mel_mm.flush()
    del mel_mm
    if cur < n_total:
        # Re-open to truncate
        full = np.load(mel_path, mmap_mode="r")
        trimmed = np.array(full[:cur])
        np.save(mel_path, trimmed)
        log.info("trimmed mel array %d -> %d", n_total, cur)
        del full, trimmed
    np.save(lab_path, lab_mm[:cur])
    meta_path.write_text(json.dumps(meta))

    counter = Counter(LABELS[l] for l in lab_mm[:cur])
    log.info("split=%s saved=%d labels=%s", split, cur, dict(counter))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="configs/guitar_section_labels.json")
    ap.add_argument("--cache-dir", default="/mnt/ml-data/guitar_section_cache")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.labels).read_text())
    records = data["records"]
    log.info("loaded %d records", len(records))

    for split in ("val", "test", "train"):
        process_split(records, split, cache_dir)

    log.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
