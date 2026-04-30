"""
Build section-type labels from PART GUITAR Expert ground truth.

For every WINDOW_S window (HOP_S step) in each song's GT, derive a label in
{silence, constant_strum, chord_stab, lead_line, single_notes, mixed}
from the GT note pattern alone.

Output: configs/guitar_section_labels.json with per-window records:
    {song_id, audio_path, t_start_s, t_end_s, label, features}

Run: python scripts/build_section_labels.py [--max-songs N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

# Add scripts/ to path so we can import the manifest parser
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_guitar_manifest import parse_part_guitar_expert  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("section_labels")

WINDOW_S = 2.0
HOP_S = 1.0


def label_window(onsets_in_window: list[frozenset[int]], window_s: float) -> tuple[str, dict]:
    """Return (label, features) for a list of fret-sets in one window."""
    n = len(onsets_in_window)
    nps = n / window_s

    # Drop open-only (-1) hits when computing playable set
    playable_sets = [frozenset(f for f in s if 0 <= f <= 4) for s in onsets_in_window]
    playable_sets = [s for s in playable_sets if s]

    if len(playable_sets) < 2:
        return "silence", {"n": n, "nps": round(nps, 2)}

    chord_count = sum(1 for s in playable_sets if len(s) >= 2)
    chord_ratio = chord_count / len(playable_sets)

    # Pattern repetition: longest run of identical consecutive fret-sets
    max_run = 1
    cur_run = 1
    for i in range(1, len(playable_sets)):
        if playable_sets[i] == playable_sets[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    repetition = max_run / len(playable_sets)

    # Unique-set ratio (variability)
    unique_ratio = len(set(playable_sets)) / len(playable_sets)

    feats = {
        "n": n,
        "nps": round(nps, 2),
        "chord_ratio": round(chord_ratio, 3),
        "repetition": round(repetition, 3),
        "unique_ratio": round(unique_ratio, 3),
    }

    # Decision tree (ordered, mutually exclusive)
    # 1. Constant strum: dense + repeating same chord/note
    if nps >= 4.0 and repetition >= 0.5:
        return "constant_strum", feats
    # 2. Chord stab: sparse + chordy
    if nps < 3.0 and chord_ratio >= 0.5:
        return "chord_stab", feats
    # 3. Lead line: dense, melodic, varied
    if nps >= 3.0 and chord_ratio < 0.2 and unique_ratio > 0.4:
        return "lead_line", feats
    # 4. Single notes: low density, no chords, not very varied
    if nps < 4.0 and chord_ratio < 0.25:
        return "single_notes", feats
    # 5. Catch-all
    return "mixed", feats


def process_song(song_record: dict) -> list[dict]:
    midi_path = Path(song_record["midi_path"])
    if not midi_path.exists():
        return []

    onsets = parse_part_guitar_expert(midi_path)  # list[(time_ms, set[int])]
    if not onsets:
        return []

    # Convert to (t_s, frozenset)
    events = [(t / 1000.0, frozenset(s)) for t, s in onsets]
    duration = float(song_record["duration_sec"])

    records = []
    t = 0.0
    while t + WINDOW_S <= duration:
        in_window = [s for ts, s in events if t <= ts < t + WINDOW_S]
        label, feats = label_window(in_window, WINDOW_S)
        records.append({
            "song_id": song_record["id"],
            "audio_path": song_record["audio_path"],
            "t_start_s": round(t, 3),
            "t_end_s": round(t + WINDOW_S, 3),
            "label": label,
            "features": feats,
            "split": song_record["split"],
        })
        t += HOP_S
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/guitar_v1_manifest.json")
    ap.add_argument("--out", default="configs/guitar_section_labels.json")
    ap.add_argument("--max-songs", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    songs = manifest["songs"]
    if args.max_songs > 0:
        songs = songs[: args.max_songs]
    log.info("Processing %d songs", len(songs))

    all_records: list[dict] = []
    for i, sr in enumerate(songs):
        if i % 200 == 0:
            log.info("[%d/%d] %s", i, len(songs), sr["id"])
        try:
            recs = process_song(sr)
            all_records.extend(recs)
        except Exception as exc:
            log.warning("failed %s: %s", sr.get("id"), exc)

    # Stats
    label_counter = Counter(r["label"] for r in all_records)
    split_counter = Counter((r["split"], r["label"]) for r in all_records)
    log.info("Total windows: %d", len(all_records))
    log.info("Label distribution:")
    for lab, c in label_counter.most_common():
        log.info("  %-15s %7d  (%.1f%%)", lab, c, 100 * c / len(all_records))
    log.info("Per-split:")
    for split in ("train", "val", "test"):
        sub = [c for (s, l), c in split_counter.items() if s == split]
        log.info("  %-5s total=%d", split, sum(sub))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"window_s": WINDOW_S, "hop_s": HOP_S, "records": all_records}))
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
