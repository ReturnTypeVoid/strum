#!/usr/bin/env python3
"""End-to-end eval for the V2 guitar neural pipeline.

For each song in the manifest split:
  - Load audio, run GuitarNeuralCharter.transcribe()
  - Load GT events from notes.mid (offset-corrected via audio_offset_ms)
  - Compute three metrics:
      1. ONSET F1     (50 ms tolerance, ignores frets)
      2. FRET-BIT F1  (per matched onset, micro-averaged 5-bit BCE)
      3. EVENT F1     (matched onset AND exact fret-set match)

Usage:
    python scripts/eval_guitar_neural.py --split val --max-songs 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.inference.guitar_neural import GuitarNeuralCharter  # noqa: E402
import preprocess_guitar_windows as pgw  # noqa: E402


def load_audio(path: Path, sr: int = 22050) -> np.ndarray:
    y, file_sr = sf.read(str(path), always_2d=False, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        import librosa
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def parse_gt(midi_path: Path, offset_ms: float) -> list[tuple[float, frozenset[int]]]:
    """Returns [(time_sec, frozenset(fret_bits))]. Open notes are dropped
    (their fret label is -1, no playable bit). Matches preprocess behavior."""
    raw = pgw.parse_onsets_from_manifest(midi_path)
    out: list[tuple[float, frozenset[int]]] = []
    for tms, frets in raw:
        t_sec = (tms - offset_ms) / 1000.0
        if t_sec < 0:
            continue
        playable = frozenset(f for f in frets if 0 <= f <= 4)
        if not playable:
            # open-only event — skip (model can't represent it via 5 bits)
            continue
        out.append((t_sec, playable))
    out.sort()
    return out


def evaluate_song(
    pred_events,
    gt_events: list[tuple[float, frozenset[int]]],
    tolerance_s: float = 0.05,
):
    """Greedy time-match. Returns aggregated counters dict."""
    pred_times = np.array([e.time_sec for e in pred_events], dtype=np.float64)
    pred_frets = [frozenset(e.frets) for e in pred_events]
    gt_times = np.array([t for t, _ in gt_events], dtype=np.float64)
    gt_frets = [f for _, f in gt_events]

    n_pred, n_gt = len(pred_times), len(gt_times)
    used_gt = np.zeros(n_gt, dtype=bool)

    onset_tp = 0
    fret_bit_tp = fret_bit_fp = fret_bit_fn = 0
    event_tp = 0

    j_start = 0
    # Pred ordered by emit; ensure sort
    order = np.argsort(pred_times)
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
        gt_set = gt_frets[best_j]
        # bit-level
        bp = pf
        bg = gt_set
        fret_bit_tp += len(bp & bg)
        fret_bit_fp += len(bp - bg)
        fret_bit_fn += len(bg - bp)
        if pf == gt_set:
            event_tp += 1

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
    }


def f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/guitar_v2.yaml")
    ap.add_argument("--manifest",
                    default="configs/guitar_v1_manifest_aligned_filtered.json")
    ap.add_argument("--onset-ckpt",
                    default="checkpoints/guitar_v2/guitar_v2_onset/best.pt")
    ap.add_argument("--fret-ckpt",
                    default="checkpoints/guitar_v2/guitar_v2_fret/best.pt")
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--max-songs", type=int, default=20)
    ap.add_argument("--onset-threshold", type=float, default=0.35)
    ap.add_argument("--fret-threshold", type=float, default=0.5)
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--export-midi-dir", type=Path, default=None,
                    help="If set, write predicted PART GUITAR MIDI per song.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    charter = GuitarNeuralCharter(
        onset_ckpt=Path(args.onset_ckpt),
        fret_ckpt=Path(args.fret_ckpt),
        config_path=Path(args.config),
        device=args.device,
    )
    print(f"loaded onset {args.onset_ckpt} (val_f1={charter.onset_meta.get('val_f1')})")
    print(f"loaded fret  {args.fret_ckpt} (val_f1={charter.fret_meta.get('val_f1')})")

    manifest = json.load(open(args.manifest))
    songs = [s for s in manifest["songs"] if s["split"] == args.split][:args.max_songs]
    print(f"evaluating {len(songs)} {args.split} songs "
          f"(thr_on={args.onset_threshold} thr_fr={args.fret_threshold} "
          f"tol={args.tolerance_ms}ms)")

    if args.export_midi_dir is not None:
        from src.inference.guitar_neural import export_events_to_midi
        args.export_midi_dir.mkdir(parents=True, exist_ok=True)

    agg = {k: 0 for k in [
        "onset_tp", "onset_fp", "onset_fn",
        "fret_bit_tp", "fret_bit_fp", "fret_bit_fn",
        "event_tp", "n_pred", "n_gt",
    ]}
    skipped = 0
    t0 = time.time()
    per_song = []

    for i, s in enumerate(songs):
        try:
            y = load_audio(Path(s["audio_path"]))
            offset_ms = float(s.get("audio_offset_ms", 0) or 0)
            gt = parse_gt(Path(s["midi_path"]), offset_ms)
            pred = charter.transcribe(
                y,
                onset_threshold=args.onset_threshold,
                fret_threshold=args.fret_threshold,
            )
            stats = evaluate_song(pred, gt, args.tolerance_ms / 1000.0)
            for k, v in stats.items():
                agg[k] += v
            p, r, f = f1(stats["onset_tp"], stats["onset_fp"], stats["onset_fn"])
            per_song.append({
                "id": s["id"], "n_pred": stats["n_pred"], "n_gt": stats["n_gt"],
                "onset_f1": round(f, 3),
                "event_f1": round(2 * stats["event_tp"] / max(stats["n_pred"] + stats["n_gt"], 1), 3),
            })

            if args.export_midi_dir is not None:
                export_events_to_midi(
                    pred, args.export_midi_dir / f"{s['id']}.mid",
                )
        except Exception as exc:
            print(f"  [skip] {s['id']}: {exc}")
            skipped += 1
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(songs)}  ({time.time() - t0:.1f}s)")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s  ({len(songs) - skipped} ok, {skipped} skipped)")

    # ─── Aggregate metrics ─────────────────────────────────────────────────
    on_p, on_r, on_f = f1(agg["onset_tp"], agg["onset_fp"], agg["onset_fn"])
    fb_p, fb_r, fb_f = f1(agg["fret_bit_tp"], agg["fret_bit_fp"], agg["fret_bit_fn"])
    # Event = exact fret-set match @ matched onset; treat as F1 over predicted+GT
    event_f = 2 * agg["event_tp"] / max(agg["n_pred"] + agg["n_gt"], 1)

    print()
    print("════════════════════════════════════════════════")
    print(f"  ONSET (50 ms tol):    P={on_p:.3f}  R={on_r:.3f}  F1={on_f:.3f}")
    print(f"                       TP={agg['onset_tp']}  FP={agg['onset_fp']}  FN={agg['onset_fn']}")
    print(f"  FRET-BIT (matched):   P={fb_p:.3f}  R={fb_r:.3f}  F1={fb_f:.3f}")
    print(f"                       TP={agg['fret_bit_tp']}  FP={agg['fret_bit_fp']}  FN={agg['fret_bit_fn']}")
    print(f"  EVENT (exact match):  F1={event_f:.3f}")
    print(f"                       event_tp={agg['event_tp']} of {agg['onset_tp']} matched onsets")
    print("════════════════════════════════════════════════")

    # Show worst songs by event_f1
    per_song.sort(key=lambda x: x["event_f1"])
    print("\nWorst 5 songs (by event F1):")
    for s in per_song[:5]:
        print(f"  {s['id']:50s}  pred={s['n_pred']:4d} gt={s['n_gt']:4d}  "
              f"on={s['onset_f1']:.3f} ev={s['event_f1']:.3f}")
    print("\nBest 5:")
    for s in per_song[-5:]:
        print(f"  {s['id']:50s}  pred={s['n_pred']:4d} gt={s['n_gt']:4d}  "
              f"on={s['onset_f1']:.3f} ev={s['event_f1']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
