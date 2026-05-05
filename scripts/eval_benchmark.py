"""Per-instrument F1 benchmark: STRUM predictions vs. ground-truth charts.

Usage:
    python scripts/eval_benchmark.py \\
        --gt-dir  /mnt/ml-data/benchmark-game-charts-gt \\
        --pred-dir /mnt/ml-data/benchmark-pred-v18-aligned \\
        --tolerance-ms 50

Computes onset F1 (existence + within tolerance) and lane/pitch agreement
across PART DRUMS, PART GUITAR, PART BASS, PART VOCALS for every paired
(GT, prediction) folder.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import mido


# Clone Hero / YARG MIDI conventions
DRUM_LANES = {96: "kick", 97: "red", 98: "yellow", 99: "blue", 100: "green"}
TOM_MARKERS = {110, 111, 112}  # 110=yellow tom, 111=blue tom, 112=green tom
GUITAR_BASS_LANES = {96, 97, 98, 99, 100}  # green/red/yellow/blue/orange


def _midi_to_seconds(mid: mido.MidiFile, ticks: int, tempo_us_per_beat: int) -> float:
    return mido.tick2second(ticks, mid.ticks_per_beat, tempo_us_per_beat)


def parse_track_events(
    path: Path,
    track_name: str,
) -> dict[str, list[tuple[float, int]]]:
    """Return {'onsets': [(time_s, pitch), ...]} for the named track.

    Onsets are pitch-coded so we can compute lane / pitch agreement on top of
    onset F1.  Tom markers (110-112) are kept separately so we can fold them
    back into PART DRUMS as cymbal-vs-tom flags.
    """
    try:
        mid = mido.MidiFile(str(path), clip=True)
    except (EOFError, OSError, ValueError) as e:
        print(f"  [warn] failed to parse {path.name}: {e}")
        return {"onsets": [], "toms": []}
    out: dict[str, list[tuple[float, int]]] = {"onsets": [], "toms": []}
    for track in mid.tracks:
        name = next((m.name for m in track if m.type == "track_name"), None)
        if name != track_name:
            continue
        tempo = 500_000  # default 120 bpm in microseconds/beat
        elapsed_ticks = 0
        elapsed_s = 0.0
        # build a tempo-aware time line
        for msg in track:
            elapsed_ticks += msg.time
            elapsed_s += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                if msg.note in TOM_MARKERS:
                    out["toms"].append((elapsed_s, msg.note))
                else:
                    out["onsets"].append((elapsed_s, msg.note))
        # only first matching track
        break
    out["onsets"].sort()
    out["toms"].sort()
    return out


def onset_f1(
    gt: list[tuple[float, int]],
    pred: list[tuple[float, int]],
    tolerance_s: float,
    require_lane_match: bool,
) -> dict[str, float]:
    """Greedy match GT to predictions within tolerance.

    If `require_lane_match`, the pitch (lane) must also match for a true
    positive — otherwise it's counted as an onset hit but lane miss.
    """
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    lane_correct = 0

    # gt and pred are time-sorted; sweep with a moving window
    for i, (g_time, g_pitch) in enumerate(gt):
        # find candidate preds in window
        best_j = -1
        best_dt = float("inf")
        # linear scan; small enough at <10k events
        for j, (p_time, p_pitch) in enumerate(pred):
            if j in matched_pred:
                continue
            dt = abs(p_time - g_time)
            if dt > tolerance_s:
                if p_time > g_time + tolerance_s:
                    break
                continue
            if require_lane_match and p_pitch != g_pitch:
                continue
            if dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j >= 0:
            matched_gt.add(i)
            matched_pred.add(best_j)
            if pred[best_j][1] == g_pitch:
                lane_correct += 1

    tp = len(matched_gt)
    fn = len(gt) - tp
    fp = len(pred) - len(matched_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    lane_acc = lane_correct / tp if tp else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "lane_accuracy": lane_acc,
        "gt_count": len(gt),
        "pred_count": len(pred),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


PARTS = [
    # (label, midi-track-name, pitch_filter)
    # Pitch filter excludes phrase/overdrive/lower-difficulty markers and keeps
    # only Expert-difficulty lanes so we compare apples to apples (predictions
    # are written at Expert and reduced for lower difficulties).
    ("drums",  "PART DRUMS",   lambda p: 96 <= p <= 100),
    ("guitar", "PART GUITAR",  lambda p: 96 <= p <= 100),
    ("bass",   "PART BASS",    lambda p: 96 <= p <= 100),
    # Vocals: pitched range, excluding phrase markers (105/106) + overdrive (116)
    ("vocals", "PART VOCALS",  lambda p: 36 <= p <= 84),
]


def evaluate_song(gt_path: Path, pred_path: Path, tol_s: float) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for label, track_name, pitch_filter in PARTS:
        gt = [(t, p) for t, p in parse_track_events(gt_path, track_name)["onsets"] if pitch_filter(p)]
        pred = [(t, p) for t, p in parse_track_events(pred_path, track_name)["onsets"] if pitch_filter(p)]
        if not gt:
            results[label] = {"skipped": "no GT track"}
            continue
        if not pred:
            results[label] = {"skipped": "no pred track", "gt_count": len(gt)}
            continue
        results[label] = onset_f1(gt, pred, tol_s, require_lane_match=False)
    return results


def aggregate(per_song: dict[str, dict[str, dict]]) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for part, *_ in PARTS:
        tp = fp = fn = lane_correct = 0
        for song_results in per_song.values():
            r = song_results.get(part, {})
            if "tp" not in r:
                continue
            tp += r["tp"]
            fp += r["fp"]
            fn += r["fn"]
            lane_correct += int(r["lane_accuracy"] * r["tp"])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        agg[part] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "lane_accuracy": lane_correct / tp if tp else 0.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--tolerance-ms", type=float, default=50.0)
    ap.add_argument("--out", type=Path, default=Path("benchmark_results.json"))
    args = ap.parse_args()

    tol_s = args.tolerance_ms / 1000.0
    per_song: dict[str, dict[str, dict]] = {}

    for gt_song in sorted(args.gt_dir.iterdir()):
        if not gt_song.is_dir():
            continue
        gt_mid = gt_song / "notes.mid"
        pred_mid = args.pred_dir / gt_song.name / "notes.mid"
        if not gt_mid.exists() or not pred_mid.exists():
            print(f"[skip] {gt_song.name} (missing pred or gt midi)")
            continue
        print(f"[eval] {gt_song.name}")
        per_song[gt_song.name] = evaluate_song(gt_mid, pred_mid, tol_s)

    summary = aggregate(per_song)

    print("\n=== Per-instrument summary (tolerance ±%dms) ===" % args.tolerance_ms)
    print(f"{'PART':<10} {'F1':>7} {'Prec':>7} {'Rec':>7} {'LaneAcc':>9} {'TP':>6} {'FP':>6} {'FN':>6}")
    for part, m in summary.items():
        print(
            f"{part:<10} {m['f1']*100:>6.1f}% {m['precision']*100:>6.1f}% "
            f"{m['recall']*100:>6.1f}% {m['lane_accuracy']*100:>8.1f}% "
            f"{m['tp']:>6} {m['fp']:>6} {m['fn']:>6}"
        )

    args.out.write_text(json.dumps(
        {"tolerance_ms": args.tolerance_ms, "per_song": per_song, "summary": summary},
        indent=2,
    ))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
