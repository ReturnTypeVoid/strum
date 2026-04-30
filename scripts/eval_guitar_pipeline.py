#!/usr/bin/env python3
"""Honest end-to-end eval for the rule-based guitar pipeline.

For N val songs:
  1. Separate stems with Demucs (cached at /mnt/ml-data/eval_stems_cache/)
  2. Run transcribe_guitar on the htdemucs guitar/other stem
  3. Compare against PART GUITAR Expert ground truth
  4. Report: density ratio, onset F1, fret-bit F1, timing offset distribution
  5. Per-section breakdown (silence / constant_strum / chord_stab / etc.) using
     the section router on the GT timeline so we know WHICH sections are bad.

Usage:
    python scripts/eval_guitar_pipeline.py --max-songs 10
    STRUM_GB_USE_ROUTER=0 python scripts/eval_guitar_pipeline.py --max-songs 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

import preprocess_guitar_windows as pgw  # noqa: E402
from src.inference.guitar_bass import transcribe_guitar  # noqa: E402

CACHE_ROOT = Path("/mnt/ml-data/eval_stems_cache")


def load_audio(path: Path, sr: int = 22050) -> np.ndarray:
    y, file_sr = sf.read(str(path), always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def parse_gt(midi_path: Path, offset_ms: float):
    raw = pgw.parse_onsets_from_manifest(midi_path)
    out = []
    for tms, frets in raw:
        t_sec = (tms - offset_ms) / 1000.0
        if t_sec < 0:
            continue
        playable = frozenset(f for f in frets if 0 <= f <= 4)
        if not playable:
            continue
        out.append((t_sec, playable))
    out.sort()
    return out


def get_or_separate_stem(song_id: str, audio_path: Path) -> Path | None:
    """Return path to guitar stem (htdemucs_6s), running Demucs if cached miss."""
    cache_dir = CACHE_ROOT / song_id.replace("/", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    guitar_path = cache_dir / "guitar.wav"
    other_path = cache_dir / "other.wav"

    # Prefer guitar stem if present (htdemucs_6s splits it out); fall back to other.
    if guitar_path.exists():
        return guitar_path
    if other_path.exists():
        return other_path

    # Run Demucs htdemucs_6s
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        import torch
        import librosa

        model = get_model("htdemucs_6s")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        y, sr_orig = librosa.load(str(audio_path), sr=None, mono=False)
        if y.ndim == 1:
            y = np.stack([y, y])
        target_sr = model.samplerate
        if sr_orig != target_sr:
            y = librosa.resample(y, orig_sr=sr_orig, target_sr=target_sr)
        wav = torch.from_numpy(y).float()
        ref = wav.mean(0)
        wav_n = (wav - ref.mean()) / ref.std()
        sources = apply_model(model, wav_n[None].to(device), device=device)[0]
        sources = sources * ref.std() + ref.mean()
        for i, name in enumerate(model.sources):
            if name in ("guitar", "other"):
                sf.write(str(cache_dir / f"{name}.wav"), sources[i].cpu().numpy().T, target_sr)
        if guitar_path.exists():
            return guitar_path
        if other_path.exists():
            return other_path
    except Exception as exc:
        print(f"  [demucs failed] {song_id}: {exc}")
        return None
    return None


def _chart_events(chart):
    """Flatten chart.notes + chart.chords into a list of (time_s, frozenset[int])."""
    events = []
    notes = getattr(chart, "notes", None) or []
    chords = getattr(chart, "chords", None) or []
    for n in notes:
        events.append((n.time_ms / 1000.0, frozenset([n.fret])))
    for c in chords:
        events.append((c.time_ms / 1000.0, frozenset(c.frets)))
    events.sort()
    return events


def evaluate_pair(pred_events, gt_events, tolerance_s=0.05):
    """Greedy match. pred_events and gt_events are both list[(t_s, frozenset[int])]."""
    pred_times = np.array([t for t, _ in pred_events], dtype=np.float64)
    pred_frets = [f for _, f in pred_events]

    gt_times = np.array([t for t, _ in gt_events], dtype=np.float64)
    gt_frets = [f for _, f in gt_events]

    n_pred, n_gt = len(pred_times), len(gt_times)
    used_gt = np.zeros(n_gt, dtype=bool)

    onset_tp = 0
    fret_bit_tp = fret_bit_fp = fret_bit_fn = 0
    event_tp = 0
    matches = []

    order = np.argsort(pred_times)
    j_start = 0
    for idx in order:
        pt = pred_times[idx]
        pf = pred_frets[idx]
        while j_start < n_gt and gt_times[j_start] < pt - tolerance_s:
            j_start += 1
        best_j = -1
        best_diff = tolerance_s + 1
        j = j_start
        while j < n_gt and gt_times[j] <= pt + tolerance_s:
            if not used_gt[j]:
                d = abs(gt_times[j] - pt)
                if d < best_diff:
                    best_diff = d
                    best_j = j
            j += 1
        if best_j < 0:
            continue
        used_gt[best_j] = True
        onset_tp += 1
        gset = gt_frets[best_j]
        fret_bit_tp += len(pf & gset)
        fret_bit_fp += len(pf - gset)
        fret_bit_fn += len(gset - pf)
        if pf == gset:
            event_tp += 1
        matches.append((pt, gt_times[best_j]))

    return {
        "onset_tp": onset_tp,
        "onset_fp": n_pred - onset_tp,
        "onset_fn": n_gt - onset_tp,
        "fret_bit_tp": fret_bit_tp,
        "fret_bit_fp": fret_bit_fp,
        "fret_bit_fn": fret_bit_fn,
        "event_tp": event_tp,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "matches": matches,
    }


def f1(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def per_section_breakdown(gt_events, pred_events_flat, sections):
    """Group both pred and GT events by section type, then F1 per group.
    pred_events_flat: list[(t_s, frozenset[int])]"""
    def label_at(t):
        for s in sections:
            if s.t_start_s <= t < s.t_end_s:
                if float(s.probs.max()) >= 0.55:
                    return s.label
                return "low_conf"
        return "low_conf"

    gt_by = defaultdict(list)
    for t, frets in gt_events:
        gt_by[label_at(t)].append((t, frets))

    pred_by = defaultdict(list)
    for t, frets in pred_events_flat:
        pred_by[label_at(t)].append((t, frets))

    out = {}
    keys = sorted(set(gt_by) | set(pred_by))
    for k in keys:
        stats = evaluate_pair(sorted(pred_by[k]), sorted(gt_by[k]))
        p, r, fsc = f1(stats["onset_tp"], stats["onset_fp"], stats["onset_fn"])
        out[k] = {
            "n_gt": stats["n_gt"],
            "n_pred": stats["n_pred"],
            "p": p, "r": r, "f1": fsc,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="configs/guitar_v1_manifest.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-songs", type=int, default=10)
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--no-section-breakdown", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    songs = [s for s in manifest["songs"] if s["split"] == args.split]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(songs)
    songs = songs[: args.max_songs]

    print(f"Eval: {len(songs)} {args.split} songs, tolerance={args.tolerance_ms}ms")
    print(f"Router: {'ON' if os.environ.get('STRUM_GB_USE_ROUTER', '1') != '0' else 'OFF'}")
    print()

    agg = {k: 0 for k in [
        "onset_tp", "onset_fp", "onset_fn",
        "fret_bit_tp", "fret_bit_fp", "fret_bit_fn",
        "event_tp", "n_pred", "n_gt",
    ]}
    timing_offsets_ms = []
    section_agg = defaultdict(lambda: {"n_gt": 0, "n_pred": 0, "tp": 0})
    skipped = 0
    t0 = time.time()

    for i, s in enumerate(songs):
        sid = s["id"]
        audio_path = Path(s["audio_path"])
        if not audio_path.exists():
            print(f"[skip] {sid}: audio missing")
            skipped += 1
            continue
        try:
            stem = get_or_separate_stem(sid, audio_path)
            if stem is None:
                skipped += 1
                continue
            offset_ms = float(s.get("audio_offset_ms", 0) or 0)
            gt = parse_gt(Path(s["midi_path"]), offset_ms)
            chart = transcribe_guitar(stem)
            pred_flat = _chart_events(chart)
            stats = evaluate_pair(pred_flat, gt, args.tolerance_ms / 1000.0)

            for k in agg:
                if k in stats:
                    agg[k] += stats[k]
            for pt, gt_t in stats["matches"]:
                timing_offsets_ms.append((pt - gt_t) * 1000.0)

            density = stats["n_pred"] / max(stats["n_gt"], 1)
            p, r, fsc = f1(stats["onset_tp"], stats["onset_fp"], stats["onset_fn"])
            print(f"  {sid[:60]:60} pred={stats['n_pred']:4d} gt={stats['n_gt']:4d} "
                  f"dens={density:.2f} F1={fsc:.3f} (P={p:.2f} R={r:.2f})")

            # Per-section breakdown
            if not args.no_section_breakdown:
                try:
                    from src.inference.section_router import get_router
                    router = get_router()
                    if router is not None:
                        y = load_audio(stem)
                        sections = router.predict(y, 22050)
                        per = per_section_breakdown(gt, pred_flat, sections)
                        for label, st in per.items():
                            section_agg[label]["n_gt"] += st["n_gt"]
                            section_agg[label]["n_pred"] += st["n_pred"]
                            section_agg[label]["tp"] += int(round(st["r"] * st["n_gt"]))
                except Exception as exc:
                    print(f"    [section breakdown failed] {exc}")

        except Exception as exc:
            print(f"[skip] {sid}: {exc}")
            skipped += 1

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s ({len(songs) - skipped} ok, {skipped} skipped)")

    on_p, on_r, on_f = f1(agg["onset_tp"], agg["onset_fp"], agg["onset_fn"])
    fb_p, fb_r, fb_f = f1(agg["fret_bit_tp"], agg["fret_bit_fp"], agg["fret_bit_fn"])
    event_f = 2 * agg["event_tp"] / max(agg["n_pred"] + agg["n_gt"], 1)
    overall_density = agg["n_pred"] / max(agg["n_gt"], 1)

    print("\n" + "=" * 70)
    print(f"  OVERALL ({args.tolerance_ms:.0f}ms tol)")
    print(f"    Density (pred/GT): {overall_density:.2f}  ({agg['n_pred']} / {agg['n_gt']})")
    print(f"    Onset    P={on_p:.3f} R={on_r:.3f} F1={on_f:.3f}")
    print(f"    Fret-bit P={fb_p:.3f} R={fb_r:.3f} F1={fb_f:.3f}")
    print(f"    Event    F1={event_f:.3f} (exact fret-set @ matched onset)")

    if timing_offsets_ms:
        offs = np.array(timing_offsets_ms)
        print(f"    Timing offset (pred-GT, matched only): "
              f"median={np.median(offs):+.1f}ms  "
              f"|q25={np.quantile(offs, 0.25):+.1f}|"
              f"|q75={np.quantile(offs, 0.75):+.1f}| "
              f"mean={offs.mean():+.1f}±{offs.std():.1f}")

    if section_agg:
        print("\n  PER-SECTION BREAKDOWN (label = router @ conf>=0.55)")
        print(f"    {'label':<18}{'n_gt':>8}{'n_pred':>10}{'dens':>8}{'recall':>10}")
        for label in sorted(section_agg.keys()):
            st = section_agg[label]
            d = st["n_pred"] / max(st["n_gt"], 1)
            r = st["tp"] / max(st["n_gt"], 1)
            print(f"    {label:<18}{st['n_gt']:>8d}{st['n_pred']:>10d}{d:>8.2f}{r:>10.2f}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main() or 0)
