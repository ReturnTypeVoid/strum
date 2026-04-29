#!/usr/bin/env python3
"""Eval the hybrid V2 guitar pipeline (V2 onset + basic-pitch + rule-mapper).

Reports the same metrics as eval_guitar_neural.py:
  - ONSET F1     (50 ms tolerance)
  - FRET-BIT F1  (per matched onset, 5-bit micro-avg)
  - EVENT F1     (matched onset + exact fret-set)

Plus optional MIDI export per song.
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

from src.inference.guitar_hybrid_v2 import (  # noqa: E402
    GuitarHybridV2Charter, export_events_to_midi_with_sustain,
)
import preprocess_guitar_windows as pgw  # noqa: E402
from eval_guitar_neural import load_audio, parse_gt, evaluate_song, f1  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/guitar_v2.yaml")
    ap.add_argument("--manifest",
                    default="configs/guitar_v1_manifest_aligned_filtered.json")
    ap.add_argument("--onset-ckpt",
                    default="checkpoints/guitar_v2/guitar_v2_onset/best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-songs", type=int, default=20)
    ap.add_argument("--onset-threshold", type=float, default=0.35)
    ap.add_argument("--snap-window-ms", type=float, default=75.0)
    ap.add_argument("--min-pitch-amp", type=float, default=0.3)
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--export-midi-dir", type=Path, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    charter = GuitarHybridV2Charter(
        onset_ckpt=Path(args.onset_ckpt),
        config_path=Path(args.config),
        device=args.device,
    )
    print(f"loaded onset {args.onset_ckpt} (val_f1={charter.onset_meta.get('val_f1')})")
    print(f"basic-pitch model loaded")

    manifest = json.load(open(args.manifest))
    songs = [s for s in manifest["songs"] if s["split"] == args.split][:args.max_songs]
    print(f"evaluating {len(songs)} {args.split} songs "
          f"(thr_on={args.onset_threshold} snap={args.snap_window_ms}ms "
          f"min_amp={args.min_pitch_amp})")

    if args.export_midi_dir is not None:
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
                y, audio_path=Path(s["audio_path"]),
                onset_threshold=args.onset_threshold,
                snap_window_s=args.snap_window_ms / 1000.0,
                min_pitch_amplitude=args.min_pitch_amp,
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
                export_events_to_midi_with_sustain(
                    pred, args.export_midi_dir / f"{s['id']}.mid",
                )
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  [skip] {s['id']}: {exc}")
            skipped += 1
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(songs)}  ({time.time() - t0:.1f}s)")

    dt = time.time() - t0
    print(f"\ndone in {dt:.1f}s  ({len(songs) - skipped} ok, {skipped} skipped)")

    on_p, on_r, on_f = f1(agg["onset_tp"], agg["onset_fp"], agg["onset_fn"])
    fb_p, fb_r, fb_f = f1(agg["fret_bit_tp"], agg["fret_bit_fp"], agg["fret_bit_fn"])
    event_f = 2 * agg["event_tp"] / max(agg["n_pred"] + agg["n_gt"], 1)

    print()
    print("════════════════════════════════════════════════")
    print(f"  ONSET (50 ms tol):    P={on_p:.3f}  R={on_r:.3f}  F1={on_f:.3f}")
    print(f"                       TP={agg['onset_tp']}  FP={agg['onset_fp']}  FN={agg['onset_fn']}")
    print(f"  FRET-BIT (matched):   P={fb_p:.3f}  R={fb_r:.3f}  F1={fb_f:.3f}")
    print(f"                       TP={agg['fret_bit_tp']}  FP={agg['fret_bit_fp']}  FN={agg['fret_bit_fn']}")
    print(f"  EVENT (exact match):  F1={event_f:.3f}  ({agg['event_tp']} of {agg['onset_tp']} matched onsets)")
    print("════════════════════════════════════════════════")

    per_song.sort(key=lambda x: x["event_f1"])
    print("\nWorst 5 (by event F1):")
    for s in per_song[:5]:
        print(f"  {s['id']:55s}  pred={s['n_pred']:4d} gt={s['n_gt']:4d}  on={s['onset_f1']:.3f} ev={s['event_f1']:.3f}")
    print("\nBest 5:")
    for s in per_song[-5:]:
        print(f"  {s['id']:55s}  pred={s['n_pred']:4d} gt={s['n_gt']:4d}  on={s['onset_f1']:.3f} ev={s['event_f1']:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
