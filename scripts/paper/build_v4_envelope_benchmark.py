"""
Build v4 in-envelope benchmark:
  1. Sample ~65 candidates from paper/benchmark_candidates_strict.csv (excluding v3 songs)
  2. Run Demucs (drums-only, two-stems mode) on each to get drum stem
  3. Compute median 1-second-window RMS on the drum stem
  4. Keep songs with RMS >= --rms-threshold up to --target-n
  5. Stage audio + GT to /mnt/ml-data/benchmark-v4-{audio,gt}/  for full pipeline run
  6. Write paper/benchmark_manifest_v4.json with kept + rejected lists

Usage:
  python scripts/paper/build_v4_envelope_benchmark.py \\
      --candidates paper/benchmark_candidates_strict.csv \\
      --exclude    paper/benchmark_manifest_v3.json \\
      --rms-threshold 0.018 \\
      --target-n 30 \\
      --pool-size 65 \\
      --seed 20260510
"""
from __future__ import annotations
import argparse, csv, json, random, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch
import warnings; warnings.filterwarnings("ignore")


_DEMUCS_MODEL = None

def _get_demucs():
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is None:
        from demucs.pretrained import get_model
        m = get_model("htdemucs_6s")
        m.to("cuda" if torch.cuda.is_available() else "cpu")
        m.eval()
        _DEMUCS_MODEL = m
    return _DEMUCS_MODEL


def screen_drumstem_rms(audio_path: Path, work_dir: Path) -> float:
    """Run Demucs (Python API, matching scripts/batch_pipeline.py:separate_stems)
    and return median 1s-window RMS of the drums stem at 22050 Hz mono."""
    from demucs.apply import apply_model

    model = _get_demucs()
    device = next(model.parameters()).device
    target_sr = model.samplerate
    y, sr_orig = librosa.load(str(audio_path), sr=None, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    if sr_orig != target_sr:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)
    waveform = torch.from_numpy(y).float()
    ref = waveform.mean(0)
    waveform_norm = (waveform - ref.mean()) / ref.std()
    with torch.no_grad():
        sources = apply_model(model, waveform_norm[None].to(device), device=device)[0]
    sources = sources * ref.std() + ref.mean()
    drums_idx = list(model.sources).index("drums")
    drums = sources[drums_idx].cpu().numpy().mean(axis=0)  # mono
    if target_sr != 22050:
        drums = librosa.resample(drums, orig_sr=target_sr, target_sr=22050)
    sr = 22050
    win = sr
    n_win = max(1, len(drums) // win)
    rms_per = np.array([np.sqrt(np.mean(drums[i*win:(i+1)*win]**2)) for i in range(n_win)])
    return float(np.median(rms_per))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--exclude", type=Path, help="manifest json whose songs to skip")
    ap.add_argument("--rms-threshold", type=float, default=0.018)
    ap.add_argument("--target-n", type=int, default=30)
    ap.add_argument("--pool-size", type=int, default=65,
                    help="how many candidates to screen with demucs")
    ap.add_argument("--seed", type=int, default=20260510)
    ap.add_argument("--audio-out", type=Path, default=Path("/mnt/ml-data/benchmark-v4-audio"))
    ap.add_argument("--gt-out",    type=Path, default=Path("/mnt/ml-data/benchmark-v4-gt"))
    ap.add_argument("--screen-cache", type=Path, default=Path("/mnt/ml-data/benchmark-v4-screen"))
    ap.add_argument("--manifest-out", type=Path, default=Path("paper/benchmark_manifest_v4.json"))
    ap.add_argument("--max-per-genre", type=int, default=6,
                    help="cap kept-songs per genre to avoid one-genre dominance")
    args = ap.parse_args()

    # Load candidates
    rows = list(csv.DictReader(open(args.candidates)))
    print(f"loaded {len(rows)} strict candidates")

    # Exclude already-used songs
    excluded_keys = set()
    if args.exclude and args.exclude.exists():
        ex = json.load(open(args.exclude))
        for s in ex.get("songs", []):
            excluded_keys.add((s["title"].strip().lower(), s["artist"].strip().lower()))
        print(f"excluding {len(excluded_keys)} v3 songs")
    rows = [r for r in rows if (r["title"].strip().lower(), r["artist"].strip().lower()) not in excluded_keys]
    print(f"{len(rows)} candidates after exclusion")

    # Stratified sample by genre, then top up randomly
    rng = random.Random(args.seed)
    by_genre: dict[str, list] = {}
    for r in rows:
        by_genre.setdefault(r["genre"], []).append(r)
    for g in by_genre.values():
        rng.shuffle(g)

    pool = []
    # Round-robin across genres until pool full
    while len(pool) < args.pool_size and any(by_genre.values()):
        for g in list(by_genre.keys()):
            if not by_genre[g]:
                continue
            pool.append(by_genre[g].pop())
            if len(pool) >= args.pool_size:
                break
    print(f"sampled {len(pool)} candidates for screening")

    # Screen each via demucs, in deterministic order
    args.screen_cache.mkdir(parents=True, exist_ok=True)
    cache_file = args.screen_cache / "rms_cache.json"
    cache: dict[str, float] = {}
    if cache_file.exists():
        cache = json.load(open(cache_file))
        print(f"loaded {len(cache)} cached RMS values")

    screened = []
    for i, r in enumerate(pool, 1):
        audio_path = Path(r["audio_path"])
        if not audio_path.exists():
            print(f"  [{i:3d}/{len(pool)}] MISS audio: {audio_path}")
            continue
        midi_path = Path(r["midi_path"])
        if not midi_path.exists():
            print(f"  [{i:3d}/{len(pool)}] MISS midi:  {midi_path}")
            continue
        cache_key = str(audio_path)
        if cache_key in cache:
            rms = cache[cache_key]
            print(f"  [{i:3d}/{len(pool)}] cached  rms={rms:.4f}  {r['artist']} - {r['title']}")
        else:
            song_work = args.screen_cache / f"{i:03d}_{audio_path.stem[:40]}"
            song_work.mkdir(parents=True, exist_ok=True)
            try:
                rms = screen_drumstem_rms(audio_path, song_work)
            except Exception as e:
                print(f"  [{i:3d}/{len(pool)}] FAIL    {r['artist']} - {r['title']}: {e}")
                continue
            cache[cache_key] = rms
            json.dump(cache, open(cache_file, "w"), indent=2)
            # Cleanup demucs output once we have RMS (keeps disk small)
            shutil.rmtree(song_work / "demucs", ignore_errors=True)
            mark = "PASS" if rms >= args.rms_threshold else "drop"
            print(f"  [{i:3d}/{len(pool)}] {mark}    rms={rms:.4f}  {r['artist']} - {r['title']}")
        screened.append({**r, "drumstem_rms_med": rms, "passed": rms >= args.rms_threshold})

    # Filter passes, cap per-genre, cap to target-n
    passes = [s for s in screened if s["passed"]]
    by_genre_pass: dict[str, list] = {}
    for s in passes:
        by_genre_pass.setdefault(s["genre"], []).append(s)
    for g in by_genre_pass.values():
        rng.shuffle(g)

    kept = []
    while len(kept) < args.target_n and any(by_genre_pass.values()):
        for g in list(by_genre_pass.keys()):
            if not by_genre_pass[g]:
                continue
            if sum(1 for k in kept if k["genre"] == g) >= args.max_per_genre:
                by_genre_pass[g] = []
                continue
            kept.append(by_genre_pass[g].pop())
            if len(kept) >= args.target_n:
                break

    print(f"\nscreened {len(screened)}, passed {len(passes)}, kept {len(kept)} (target {args.target_n})")
    print("kept genre breakdown:")
    g_counts: dict[str,int] = {}
    for k in kept:
        g_counts[k["genre"]] = g_counts.get(k["genre"], 0) + 1
    for g, c in sorted(g_counts.items(), key=lambda x: -x[1]):
        print(f"  {g:<12}  {c}")

    # Stage symlinks
    args.audio_out.mkdir(parents=True, exist_ok=True)
    args.gt_out.mkdir(parents=True, exist_ok=True)
    for k in kept:
        slug = f"{k['artist']} - {k['title']}".replace("/", "_")
        audio_dst = args.audio_out / f"{slug}{Path(k['audio_path']).suffix}"
        gt_dst_dir = args.gt_out / slug
        gt_dst_dir.mkdir(parents=True, exist_ok=True)
        gt_dst = gt_dst_dir / "notes.mid"
        if audio_dst.exists() or audio_dst.is_symlink():
            audio_dst.unlink()
        if gt_dst.exists() or gt_dst.is_symlink():
            gt_dst.unlink()
        audio_dst.symlink_to(k["audio_path"])
        gt_dst.symlink_to(k["midi_path"])

    # Write manifest
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "seed": args.seed,
        "rms_threshold": args.rms_threshold,
        "target_n": args.target_n,
        "n_screened": len(screened),
        "n_passed": len(passes),
        "n_kept": len(kept),
        "kept": kept,
        "rejected_screen": [s for s in screened if not s["passed"]],
    }, open(args.manifest_out, "w"), indent=2)
    print(f"\nwrote {args.manifest_out}")
    print(f"audio staged at {args.audio_out}")
    print(f"gt    staged at {args.gt_out}")


if __name__ == "__main__":
    main()
