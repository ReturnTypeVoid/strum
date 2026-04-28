#!/usr/bin/env python3
"""Build a training manifest for the Guitar V1 neural pipeline.

Walks `/mnt/ml-data/training_songs/{C3,RockBand4,Custom,FortniteFestival,...}`
for songs that have:
  * an audio source (preferring isolated `guitar.ogg`, falling back to
    `song.ogg`/`song.opus`/etc. with a flag),
  * a `notes.mid` containing `PART GUITAR` with at least one Expert note.

Outputs a single JSON manifest:
    {
      "songs": [
        {
          "id": "<root>__<song_dir>",
          "song_dir": "/abs/path",
          "audio_path": "/abs/path/guitar.ogg",
          "audio_kind": "isolated_guitar" | "rhythm" | "full_mix",
          "midi_path": "/abs/path/notes.mid",
          "duration_sec": 234.1,
          "n_expert_notes": 412,
          "n_chord_onsets": 38,
          "fret_distribution": [120, 95, 80, 70, 47],   # frets 0..4
          "split": "train" | "val" | "test"
        }, ...
      ],
      "summary": {...}
    }

Splits are deterministic (hash of song_id) so adding songs later doesn't
shuffle existing assignments.

Usage:
    python scripts/build_guitar_manifest.py \
        --roots /mnt/ml-data/training_songs/C3 \
                /mnt/ml-data/training_songs/RockBand4 \
                /mnt/ml-data/training_songs/Custom \
                /mnt/ml-data/training_songs/FortniteFestival \
        --output configs/guitar_v1_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import mido
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("guitar_manifest")

# Clone Hero / RB MIDI conventions for PART GUITAR Expert
EXPERT_NOTES = {96, 97, 98, 99, 100}      # frets 0 (green) .. 4 (orange)
NOTE_TO_FRET = {96: 0, 97: 1, 98: 2, 99: 3, 100: 4}
# Note 95 = open note (no fret) on some charts. We keep but tag as fret -1.
OPEN_NOTE = 95
# Note 105/106 etc. are forced HOPO/strum and lyric markers — ignored.

# Audio source preference (highest first)
AUDIO_PREFERENCE = [
    ("guitar.ogg", "isolated_guitar"),
    ("guitar.opus", "isolated_guitar"),
    ("guitar.wav", "isolated_guitar"),
    ("guitar.mp3", "isolated_guitar"),
    ("rhythm.ogg", "rhythm"),
    ("rhythm.opus", "rhythm"),
    ("rhythm.wav", "rhythm"),
    ("song.ogg", "full_mix"),
    ("song.opus", "full_mix"),
    ("song.wav", "full_mix"),
    ("song.mp3", "full_mix"),
]

# Chord-detection threshold: notes within this many ms count as one onset
CHORD_GROUP_MS = 25.0


def find_audio(song_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    files = {p.name.lower(): p for p in song_dir.iterdir() if p.is_file()}
    for name, kind in AUDIO_PREFERENCE:
        if name in files:
            return files[name], kind
    return None, None


def parse_part_guitar_expert(midi_path: Path) -> Optional[list[tuple[float, set[int]]]]:
    """Return list of (time_ms, {fret_indices}) for Expert PART GUITAR.

    Frets are 0..4 for notes 96..100. Open-strum (note 95) is mapped to -1 so
    callers can decide how to handle it; we currently include it in the onset
    list but exclude from fret labels.
    """
    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception as exc:
        log.debug("parse failed for %s: %s", midi_path, exc)
        return None

    track = None
    for tr in mid.tracks:
        if tr.name == "PART GUITAR":
            track = tr
            break
    if track is None:
        return None

    # Build tempo map from any TEMPO track or this track itself
    tempo_changes: list[tuple[int, int]] = [(0, 500_000)]  # (tick, microsec/beat)
    for tr in mid.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.type == "set_tempo":
                tempo_changes.append((tick, msg.tempo))
    tempo_changes.sort()
    # Deduplicate
    dedup = [tempo_changes[0]]
    for tk, tm in tempo_changes[1:]:
        if tk == dedup[-1][0]:
            dedup[-1] = (tk, tm)
        else:
            dedup.append((tk, tm))
    tempo_changes = dedup

    tpb = mid.ticks_per_beat

    def tick_to_ms(target_tick: int) -> float:
        ms = 0.0
        cur_tick = 0
        cur_tempo = tempo_changes[0][1]
        for nxt_tick, nxt_tempo in tempo_changes[1:]:
            if nxt_tick >= target_tick:
                break
            seg_ticks = nxt_tick - cur_tick
            ms += seg_ticks * cur_tempo / tpb / 1000.0
            cur_tick = nxt_tick
            cur_tempo = nxt_tempo
        ms += (target_tick - cur_tick) * cur_tempo / tpb / 1000.0
        return ms

    # Walk track collecting note_on events for Expert range
    raw_hits: list[tuple[float, int]] = []  # (time_ms, fret or -1)
    tick = 0
    for msg in track:
        tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            if msg.note in EXPERT_NOTES:
                raw_hits.append((tick_to_ms(tick), NOTE_TO_FRET[msg.note]))
            elif msg.note == OPEN_NOTE:
                raw_hits.append((tick_to_ms(tick), -1))

    if not raw_hits:
        return None

    # Group simultaneous hits into chord onsets
    raw_hits.sort()
    grouped: list[tuple[float, set[int]]] = []
    cur_t, cur_frets = raw_hits[0][0], {raw_hits[0][1]}
    for t, fret in raw_hits[1:]:
        if t - cur_t <= CHORD_GROUP_MS:
            cur_frets.add(fret)
        else:
            grouped.append((cur_t, cur_frets))
            cur_t, cur_frets = t, {fret}
    grouped.append((cur_t, cur_frets))
    return grouped


def deterministic_split(song_id: str, train_ratio=0.85, val_ratio=0.075) -> str:
    h = int(hashlib.md5(song_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < train_ratio:
        return "train"
    if h < train_ratio + val_ratio:
        return "val"
    return "test"


def process_song(root_name: str, song_dir: Path) -> Optional[dict]:
    audio_path, audio_kind = find_audio(song_dir)
    midi_path = song_dir / "notes.mid"
    if audio_path is None or not midi_path.exists():
        return None

    onsets = parse_part_guitar_expert(midi_path)
    if not onsets:
        return None

    # Probe audio duration cheaply (header only)
    try:
        info = sf.info(str(audio_path))
        duration_sec = float(info.duration)
    except Exception as exc:
        log.debug("audio probe failed for %s: %s", audio_path, exc)
        return None

    if duration_sec < 30 or duration_sec > 900:
        # Skip extremely short or extremely long entries (likely intros/loops/podcasts)
        return None

    # Drop onsets past audio end (defensive)
    onsets = [(t, frets) for t, frets in onsets if t / 1000.0 < duration_sec - 0.1]
    if len(onsets) < 20:
        return None

    fret_counts = [0, 0, 0, 0, 0]
    chord_count = 0
    for _, frets in onsets:
        playable = {f for f in frets if 0 <= f <= 4}
        if len(playable) >= 2:
            chord_count += 1
        for f in playable:
            fret_counts[f] += 1

    safe_song_name = song_dir.name.replace("/", "_")
    song_id = f"{root_name}__{safe_song_name}"

    return {
        "id": song_id,
        "song_dir": str(song_dir),
        "audio_path": str(audio_path),
        "audio_kind": audio_kind,
        "midi_path": str(midi_path),
        "duration_sec": round(duration_sec, 3),
        "n_expert_notes": len(onsets),
        "n_chord_onsets": chord_count,
        "fret_distribution": fret_counts,
        "split": deterministic_split(song_id),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        nargs="+",
        default=[
            "/mnt/ml-data/training_songs/C3",
            "/mnt/ml-data/training_songs/RockBand4",
            "/mnt/ml-data/training_songs/Custom",
            "/mnt/ml-data/training_songs/FortniteFestival",
            "/mnt/ml-data/training_songs/Rock Band",
            "/mnt/ml-data/training_songs/Guitar Hero",
        ],
        help="Root directories to scan for song folders (recursively)",
    )
    ap.add_argument(
        "--output",
        default="configs/guitar_v1_manifest.json",
        help="Output manifest JSON",
    )
    ap.add_argument(
        "--require-isolated",
        action="store_true",
        help="Only keep songs with isolated guitar.ogg/rhythm.ogg (skip full_mix)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap song count (debug)")
    args = ap.parse_args()

    candidate_dirs: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for root in args.roots:
        rp = Path(root)
        if not rp.is_dir():
            log.warning("root does not exist: %s", rp)
            continue
        # Recursively find every directory that contains a notes.mid
        for mid in rp.rglob("notes.mid"):
            sd = mid.parent
            if sd in seen:
                continue
            seen.add(sd)
            candidate_dirs.append((rp.name, sd))

    log.info("scanning %d candidate song dirs across %d roots",
             len(candidate_dirs), len(args.roots))

    if args.limit:
        candidate_dirs = candidate_dirs[: args.limit]

    songs: list[dict] = []
    skipped_reasons: Counter = Counter()
    for root_name, sd in tqdm(candidate_dirs, desc="manifest"):
        try:
            entry = process_song(root_name, sd)
        except Exception as exc:
            skipped_reasons["exception"] += 1
            log.debug("error on %s: %s", sd, exc)
            continue
        if entry is None:
            skipped_reasons["no_audio_or_chart_or_too_short"] += 1
            continue
        if args.require_isolated and entry["audio_kind"] == "full_mix":
            skipped_reasons["full_mix_excluded"] += 1
            continue
        songs.append(entry)

    # Summary
    by_split = Counter(s["split"] for s in songs)
    by_kind = Counter(s["audio_kind"] for s in songs)
    fret_totals = [0, 0, 0, 0, 0]
    chord_total = 0
    note_total = 0
    for s in songs:
        for i, c in enumerate(s["fret_distribution"]):
            fret_totals[i] += c
        chord_total += s["n_chord_onsets"]
        note_total += s["n_expert_notes"]

    summary = {
        "total_kept": len(songs),
        "total_skipped": sum(skipped_reasons.values()),
        "skipped_reasons": dict(skipped_reasons),
        "by_split": dict(by_split),
        "by_audio_kind": dict(by_kind),
        "fret_totals": fret_totals,
        "chord_onsets": chord_total,
        "note_total_onsets": note_total,
        "avg_notes_per_song": round(note_total / max(len(songs), 1), 1),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"songs": songs, "summary": summary}, f, indent=2)

    log.info("=" * 60)
    log.info("MANIFEST: %s", out_path)
    log.info("kept %d / skipped %d", summary["total_kept"], summary["total_skipped"])
    for k, v in summary["by_split"].items():
        log.info("  split %s: %d", k, v)
    for k, v in summary["by_audio_kind"].items():
        log.info("  audio_kind %s: %d", k, v)
    log.info("  fret distribution: %s", fret_totals)
    log.info("  chord onsets: %d (%.1f%% of notes)",
             chord_total, 100.0 * chord_total / max(note_total, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
