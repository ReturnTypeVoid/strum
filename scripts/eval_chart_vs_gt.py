#!/usr/bin/env python3
"""
End-to-end chart scorer.

Compares a predicted Clone Hero notes.mid against a ground-truth notes.mid
per-instrument:

  - Drums   (PART DRUMS)         onset-F1 by lane
  - Guitar  (PART GUITAR)        onset-F1 by fret
  - Bass    (PART BASS)          onset-F1 by fret
  - Keys    (PART KEYS)          onset-F1 (any-lane)
  - Vocals  (PART VOCALS)        onset-F1 of phrase starts (timing only)

Scoring: per-onset 50ms tolerance, greedy bisect matcher over time.
Produces a per-song summary and an aggregate table across many songs.

Usage:
    python scripts/eval_chart_vs_gt.py \\
        --pairs pairs.json \\
        --output outputs/chart_eval/summary.json

  pairs.json: [{"id": "...", "pred": "path/to/pred/notes.mid",
                "gt": "path/to/gt/notes.mid"}, ...]
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import mido

TIME_TOL_MS = 50.0

# ─── Track name aliases ────────────────────────────────────────────────────
DRUM_TRACKS   = {"PART DRUMS", "PART DRUMS_2X"}
GUITAR_TRACKS = {"PART GUITAR", "T1 GEMS"}
BASS_TRACKS   = {"PART BASS", "PART RHYTHM"}
KEYS_TRACKS   = {"PART KEYS"}
VOCAL_TRACKS  = {"PART VOCALS", "PART VOCAL", "VOCALS"}

# Expert-difficulty pitch ranges
DRUM_EXPERT_LO, DRUM_EXPERT_HI = 96, 100        # 96..100 = lanes 0..4
GUITAR_EXPERT_LO, GUITAR_EXPERT_HI = 96, 100    # same range
KEYS_EXPERT_LO, KEYS_EXPERT_HI = 96, 100


# ─── MIDI helpers ──────────────────────────────────────────────────────────
def _tempo_map(midi: mido.MidiFile) -> list[tuple[int, int]]:
    """Return list of (abs_tick, tempo_us_per_beat) sorted by tick."""
    tempo_events = [(0, 500000)]  # default 120 BPM
    for track in midi.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "set_tempo":
                tempo_events.append((t, msg.tempo))
    tempo_events.sort()
    # Dedup at same tick — keep last
    out = []
    for tick, tempo in tempo_events:
        if out and out[-1][0] == tick:
            out[-1] = (tick, tempo)
        else:
            out.append((tick, tempo))
    return out


def _tick_to_ms(tick: int, tempo_map: list[tuple[int, int]], tpb: int) -> float:
    """Convert absolute tick → ms, honoring tempo changes."""
    ms = 0.0
    last_tick = 0
    last_tempo = tempo_map[0][1]
    for evt_tick, evt_tempo in tempo_map:
        if evt_tick >= tick:
            break
        ms += (evt_tick - last_tick) * last_tempo / tpb / 1000.0
        last_tick = evt_tick
        last_tempo = evt_tempo
    ms += (tick - last_tick) * last_tempo / tpb / 1000.0
    return ms


def _extract_track_onsets(
    midi: mido.MidiFile,
    track_names: set[str],
    pitch_lo: int | None = None,
    pitch_hi: int | None = None,
) -> list[tuple[float, int]]:
    """Return [(onset_ms, pitch), ...] for note_on events with velocity > 0
    in the first matching track.  If pitch_lo/hi given, only keep notes in
    [lo, hi]."""
    tempo_map = _tempo_map(midi)
    tpb = midi.ticks_per_beat
    for track in midi.tracks:
        name = (track.name or "").strip().upper()
        if name in track_names:
            onsets = []
            t = 0
            for msg in track:
                t += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    p = msg.note
                    if pitch_lo is not None and (p < pitch_lo or p > pitch_hi):
                        continue
                    onsets.append((_tick_to_ms(t, tempo_map, tpb), p))
            return onsets
    return []


# ─── Matching / scoring ────────────────────────────────────────────────────
@dataclass
class InstrScore:
    pred_count: int = 0
    gt_count: int = 0
    matched: int = 0
    unmatched_gt: int = 0     # misses
    unmatched_pred: int = 0   # FPs

    @property
    def precision(self) -> float:
        return self.matched / max(self.pred_count, 1)

    @property
    def recall(self) -> float:
        return self.matched / max(self.gt_count, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-9)

    @property
    def density_ratio(self) -> float:
        return self.pred_count / max(self.gt_count, 1)


def _greedy_match(pred_times: list[float], gt_times: list[float],
                  tol_ms: float = TIME_TOL_MS) -> tuple[int, int, int]:
    """Greedy 1-to-1 match within tol_ms.  Returns (matched, unmatched_gt,
    unmatched_pred)."""
    pred = sorted(pred_times)
    gt = sorted(gt_times)
    used_pred = [False] * len(pred)
    matched = 0
    for g in gt:
        # bisect window
        lo = bisect.bisect_left(pred, g - tol_ms)
        hi = bisect.bisect_right(pred, g + tol_ms)
        best_j = -1
        best_dt = float("inf")
        for j in range(lo, hi):
            if used_pred[j]:
                continue
            dt = abs(pred[j] - g)
            if dt < best_dt:
                best_dt = dt
                best_j = j
        if best_j >= 0:
            used_pred[best_j] = True
            matched += 1
    return matched, len(gt) - matched, len(pred) - matched


def _score_instrument(pred_onsets, gt_onsets,
                      tol_ms: float = TIME_TOL_MS) -> InstrScore:
    pred_t = [t for t, _p in pred_onsets]
    gt_t = [t for t, _p in gt_onsets]
    m, mg, mp = _greedy_match(pred_t, gt_t, tol_ms)
    return InstrScore(
        pred_count=len(pred_t), gt_count=len(gt_t),
        matched=m, unmatched_gt=mg, unmatched_pred=mp,
    )


# ─── Per-song scorer ───────────────────────────────────────────────────────
INSTR_SPECS = [
    ("drums",  DRUM_TRACKS,   DRUM_EXPERT_LO,  DRUM_EXPERT_HI),
    ("guitar", GUITAR_TRACKS, GUITAR_EXPERT_LO, GUITAR_EXPERT_HI),
    ("bass",   BASS_TRACKS,   GUITAR_EXPERT_LO, GUITAR_EXPERT_HI),
    ("keys",   KEYS_TRACKS,   KEYS_EXPERT_LO,  KEYS_EXPERT_HI),
    ("vocals", VOCAL_TRACKS,  None, None),
]


def score_song(pred_mid: Path, gt_mid: Path) -> dict[str, InstrScore]:
    pred = mido.MidiFile(str(pred_mid))
    gt = mido.MidiFile(str(gt_mid))
    out: dict[str, InstrScore] = {}
    for name, tracks, lo, hi in INSTR_SPECS:
        p_on = _extract_track_onsets(pred, tracks, lo, hi)
        g_on = _extract_track_onsets(gt, tracks, lo, hi)
        out[name] = _score_instrument(p_on, g_on)
    return out


# ─── Aggregation + report ──────────────────────────────────────────────────
def aggregate(per_song: dict[str, dict[str, InstrScore]]) -> dict[str, InstrScore]:
    agg: dict[str, InstrScore] = defaultdict(InstrScore)
    for sid, scores in per_song.items():
        for instr, s in scores.items():
            a = agg[instr]
            a.pred_count    += s.pred_count
            a.gt_count      += s.gt_count
            a.matched       += s.matched
            a.unmatched_gt  += s.unmatched_gt
            a.unmatched_pred += s.unmatched_pred
    return dict(agg)


def print_report(per_song, agg):
    print("\n" + "=" * 92)
    print(f"{'song':<14} {'instr':<7} {'gt':>6} {'pred':>6} {'match':>6} "
          f"{'P':>6} {'R':>6} {'F1':>6} {'dens':>6}")
    print("-" * 92)
    for sid, scores in per_song.items():
        for instr, s in scores.items():
            if s.gt_count == 0 and s.pred_count == 0:
                continue
            print(f"{sid[:14]:<14} {instr:<7} {s.gt_count:>6} {s.pred_count:>6} "
                  f"{s.matched:>6} {s.precision*100:>5.1f}% {s.recall*100:>5.1f}% "
                  f"{s.f1*100:>5.1f}% {s.density_ratio:>5.2f}")
    print("=" * 92)
    print(f"{'AGGREGATE':<14} {'instr':<7} {'gt':>6} {'pred':>6} {'match':>6} "
          f"{'P':>6} {'R':>6} {'F1':>6} {'dens':>6}")
    print("-" * 92)
    for instr, s in agg.items():
        if s.gt_count == 0 and s.pred_count == 0:
            continue
        print(f"{'-':<14} {instr:<7} {s.gt_count:>6} {s.pred_count:>6} "
              f"{s.matched:>6} {s.precision*100:>5.1f}% {s.recall*100:>5.1f}% "
              f"{s.f1*100:>5.1f}% {s.density_ratio:>5.2f}")
    print("=" * 92)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="JSON file: [{id, pred, gt}, ...]")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    pairs = json.loads(Path(args.pairs).read_text())
    per_song: dict[str, dict[str, InstrScore]] = {}
    for entry in pairs:
        sid = entry["id"]
        pred = Path(entry["pred"])
        gt = Path(entry["gt"])
        if not pred.exists():
            print(f"  SKIP {sid}: pred missing ({pred})", file=sys.stderr)
            continue
        if not gt.exists():
            print(f"  SKIP {sid}: gt missing ({gt})", file=sys.stderr)
            continue
        try:
            per_song[sid] = score_song(pred, gt)
        except Exception as e:
            print(f"  ERR {sid}: {e}", file=sys.stderr)
    agg = aggregate(per_song)
    print_report(per_song, agg)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out = {
            "per_song": {sid: {k: asdict(v) for k, v in scores.items()}
                         for sid, scores in per_song.items()},
            "aggregate": {k: asdict(v) for k, v in agg.items()},
        }
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
