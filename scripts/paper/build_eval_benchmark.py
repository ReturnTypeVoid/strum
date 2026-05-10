"""Build the paper's evaluation benchmark.

Inputs:
  --candidates   CSV of candidate songs you own audio + GT chart for.
                 Required columns:
                     title, artist, genre, audio_path, midi_path
                 Optional:
                     duration_s, source
  --train-manifest  The training manifest JSON used by drums_v14 / classifiers.
                    Used to verify holdout (rejects any song whose id, audio
                    sha256, or artist appears in train).
  --target-per-genre  Stratified target count per genre tag (default 3).
  --out          Output benchmark manifest JSON.

Behaviour:
  1. Hashes every candidate audio file (sha256) and every train audio file.
  2. Hard-rejects candidates whose audio_path or audio sha256 is in train.
  3. Soft-rejects candidates whose artist appears in train (logged, requires
     --allow-artist-overlap to keep them).
  4. Stratified-samples remaining candidates to hit the per-genre target.
  5. Writes a deterministic manifest with a shuffle seed.

Usage:
    python scripts/paper/build_eval_benchmark.py \
        --candidates  paper/benchmark_candidates.csv \
        --train-manifest /mnt/ml-data/dataset_drums/manifest.json \
        --out         paper/benchmark_manifest.json \
        --target-per-genre 3
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def norm_artist(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def load_train_signatures(manifest_path: Path) -> tuple[set[str], set[str], set[str]]:
    """Return (set of audio sha256, set of normalised audio paths, set of normalised artists).

    Artists are inferred from song id (we assume the manifest id encodes
    "Artist - Title" or similar; if not, the artist set will be empty and we'll
    fall back to path/hash-only checks).
    """
    manifest = json.loads(manifest_path.read_text())
    songs = manifest.get("songs", manifest if isinstance(manifest, list) else [])
    base = manifest_path.parent

    sha_set: set[str] = set()
    path_set: set[str] = set()
    artist_set: set[str] = set()

    for s in songs:
        sid = str(s.get("id", ""))
        if " - " in sid:
            artist_set.add(norm_artist(sid.split(" - ", 1)[0]))
        for stem_kind in ("drums", "guitar", "bass", "vocals", "mix"):
            rel = s.get("stems", {}).get(stem_kind)
            if not rel:
                continue
            ap = (base / rel).resolve()
            path_set.add(str(ap).lower())
            if ap.exists():
                try:
                    sha_set.add(sha256(ap))
                except OSError:
                    pass
    return sha_set, path_set, artist_set


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--train-manifest", required=True, type=Path)
    ap.add_argument("--target-per-genre", type=int, default=3)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260509)
    ap.add_argument("--allow-artist-overlap", action="store_true",
                    help="Don't reject candidates whose artist also appears in train.")
    ap.add_argument("--skip-train-hash", action="store_true",
                    help="Skip sha256-ing every train audio file (uses path+artist only). "
                         "Faster, weaker holdout guarantee.")
    args = ap.parse_args()

    print(f"loading train manifest from {args.train_manifest}")
    if args.skip_train_hash:
        manifest = json.loads(args.train_manifest.read_text())
        songs = manifest.get("songs", manifest if isinstance(manifest, list) else [])
        base = args.train_manifest.parent
        train_sha: set[str] = set()
        train_paths = {
            str((base / s.get("stems", {}).get(k, "x")).resolve()).lower()
            for s in songs for k in ("drums", "guitar", "bass", "vocals", "mix")
            if s.get("stems", {}).get(k)
        }
        train_artists = {
            norm_artist(str(s.get("id", "")).split(" - ", 1)[0])
            for s in songs if " - " in str(s.get("id", ""))
        }
    else:
        train_sha, train_paths, train_artists = load_train_signatures(args.train_manifest)
    print(f"  train: {len(train_sha)} hashes, {len(train_paths)} paths, {len(train_artists)} artists")

    rows = list(csv.DictReader(args.candidates.open()))
    print(f"loaded {len(rows)} candidate rows")

    accepted: list[dict] = []
    rejected: list[dict] = []

    for r in rows:
        ap_path = Path(r["audio_path"]).expanduser().resolve()
        mp_path = Path(r["midi_path"]).expanduser().resolve()
        reasons: list[str] = []

        if not ap_path.exists():
            reasons.append("audio missing")
        if not mp_path.exists():
            reasons.append("midi missing")

        audio_hash = ""
        if ap_path.exists():
            audio_hash = sha256(ap_path)
            if audio_hash in train_sha:
                reasons.append("audio sha256 in train")
            if str(ap_path).lower() in train_paths:
                reasons.append("audio path in train")

        artist = norm_artist(r.get("artist", ""))
        if artist and artist in train_artists and not args.allow_artist_overlap:
            reasons.append("artist in train")

        entry = {
            "title": r["title"].strip(),
            "artist": r["artist"].strip(),
            "genre": r["genre"].strip().lower(),
            "audio_path": str(ap_path),
            "midi_path": str(mp_path),
            "audio_sha256": audio_hash,
            "duration_s": float(r["duration_s"]) if r.get("duration_s") else None,
            "source": r.get("source", "").strip() or None,
        }
        if reasons:
            entry["reject_reasons"] = reasons
            rejected.append(entry)
        else:
            accepted.append(entry)

    print(f"\nholdout: {len(accepted)} accepted, {len(rejected)} rejected")

    by_genre: dict[str, list[dict]] = defaultdict(list)
    for e in accepted:
        by_genre[e["genre"]].append(e)

    rng = random.Random(args.seed)
    sampled: list[dict] = []
    for genre, pool in sorted(by_genre.items()):
        rng.shuffle(pool)
        take = pool[: args.target_per_genre]
        sampled.extend(take)
        print(f"  {genre:<20} pool={len(pool):>3}  taken={len(take)}")

    out = {
        "seed": args.seed,
        "tolerance_ms": 100,
        "n_songs": len(sampled),
        "songs": sampled,
        "rejected": rejected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}  ({len(sampled)} songs total)")


if __name__ == "__main__":
    main()
