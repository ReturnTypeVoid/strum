"""Emit dataset statistics for the paper's Data section.

Walks the training manifest and prints (and writes JSON):
  * per-split song count, hours of audio, distinct id/source counts
  * per-class onset count (from the parsed ground-truth MIDI files)
  * tempo distribution (median / p10 / p90 BPM) per split

Usage:
    python scripts/paper/dataset_stats.py \
        --manifest /mnt/ml-data/dataset_drums/manifest.json \
        --out      logs/dataset_stats.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

# 8-class drum taxonomy used by the OnsetClassifier ensemble.
LANE_NAMES = {
    0: "kick", 1: "snare", 2: "hi-hat",
    3: "high-tom", 4: "ride",
    5: "low-tom", 6: "crash", 7: "floor-tom",
}


def audio_seconds(path: Path) -> float:
    try:
        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return 0.0


def labels_stats(path: Path) -> tuple[Counter, list[float]]:
    """Return (per-class hit counts, list of tempo BPMs from tempo_events)."""
    counts: Counter = Counter()
    tempos: list[float] = []
    try:
        d = json.loads(path.read_text())
    except Exception:
        return counts, tempos
    for hit in d.get("hits", []):
        lane = hit.get("lane")
        if lane is not None:
            counts[lane] += 1
    tpb = d.get("ticks_per_beat") or 480
    for ev in d.get("tempo_events", []):
        # tempo_events entries are typically {"time_ms": ..., "us_per_beat": ...}
        # or {"tick": ..., "tempo": ...}; cover both.
        us = ev.get("us_per_beat") or ev.get("tempo")
        if us:
            tempos.append(60_000_000.0 / float(us))
    return counts, tempos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("logs/dataset_stats.json"))
    ap.add_argument("--skip-audio", action="store_true",
                    help="Skip audio duration probing (faster, but no hours total).")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    base = args.manifest.parent
    songs = manifest.get("songs", manifest if isinstance(manifest, list) else [])

    by_split: dict[str, dict] = defaultdict(lambda: {
        "n_songs": 0,
        "hours": 0.0,
        "ids": set(),
        "sources": Counter(),
        "lane_counts": Counter(),
        "tempos": [],
        "missing_audio": 0,
        "missing_labels": 0,
        "num_hits_total": 0,
    })

    for i, s in enumerate(songs):
        split = s.get("split", "unknown")
        bucket = by_split[split]
        bucket["n_songs"] += 1
        bucket["ids"].add(s.get("id", f"_idx{i}"))
        bucket["sources"][s.get("source_game", "unknown")] += 1
        bucket["num_hits_total"] += int(s.get("num_hits", 0) or 0)

        drums_audio = s.get("stems", {}).get("drums")
        if drums_audio:
            audio_path = (base / drums_audio).resolve()
            if audio_path.exists():
                if not args.skip_audio:
                    bucket["hours"] += audio_seconds(audio_path) / 3600.0
                # drums_labels.json sits next to drums.ogg
                labels_path = audio_path.parent / "drums_labels.json"
                if labels_path.exists():
                    lanes, tempos = labels_stats(labels_path)
                    bucket["lane_counts"].update(lanes)
                    bucket["tempos"].extend(tempos)
                else:
                    bucket["missing_labels"] += 1
            else:
                bucket["missing_audio"] += 1

        if (i + 1) % 200 == 0:
            print(f"  scanned {i+1}/{len(songs)}")

    report: dict[str, dict] = {}
    for split, b in by_split.items():
        tempos = b["tempos"]
        report[split] = {
            "n_songs": b["n_songs"],
            "hours_audio": round(b["hours"], 2),
            "n_unique_ids": len(b["ids"]),
            "sources": dict(b["sources"]),
            "missing_audio_files": b["missing_audio"],
            "missing_labels_files": b["missing_labels"],
            "num_hits_total": b["num_hits_total"],
            "lane_counts": {LANE_NAMES.get(k, str(k)): v
                            for k, v in b["lane_counts"].most_common()},
            "tempo_bpm": {
                "n": len(tempos),
                "median": round(statistics.median(tempos), 2) if tempos else None,
                "p10": round(statistics.quantiles(tempos, n=10)[0], 2) if len(tempos) >= 10 else None,
                "p90": round(statistics.quantiles(tempos, n=10)[8], 2) if len(tempos) >= 10 else None,
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    print("\n=== Dataset summary ===")
    for split, r in report.items():
        print(f"\n[{split}]")
        print(f"  songs:        {r['n_songs']}")
        print(f"  hours audio:  {r['hours_audio']}")
        print(f"  unique ids:   {r['n_unique_ids']}")
        print(f"  sources:      {r['sources']}")
        print(f"  total hits:   {r['num_hits_total']}")
        print(f"  tempo (BPM):  median={r['tempo_bpm']['median']}  p10={r['tempo_bpm']['p10']}  p90={r['tempo_bpm']['p90']}")
        print(f"  per-class onset counts:")
        for lane, n in r["lane_counts"].items():
            print(f"    {lane:<10} {n}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
