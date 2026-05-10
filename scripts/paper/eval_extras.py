"""Compute paper-grade evaluation extras from a paired GT / prediction set.

Three artefacts:
  1. Drum lane confusion matrix -> CSV + PNG heatmap.
     Rows = GT drum lane, cols = predicted drum lane (matched onsets only,
     ignoring lane on the matching pass).
  2. GT chart-vs-audio onset offset histogram -> CSV + PNG.
     Detects audio onsets on the (Demucs) drum stem with librosa, matches each
     GT chart hit to its nearest audio onset, and histograms the signed delta.
     Justifies the choice of tolerance window in the paper.
  3. Per-instrument F1 bar chart -> PNG (reads benchmark_results.json).

Usage:
    python scripts/paper/eval_extras.py \
        --gt-dir    /mnt/ml-data/benchmark-gt-charts \
        --pred-dir  /mnt/ml-data/benchmark-pred \
        --stem-dir  /mnt/ml-data/benchmark-stems   \
        --bench-json benchmark_results.json \
        --out-dir   paper/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mido
import numpy as np

DRUM_PITCH_TO_LANE = {96: "kick", 97: "red", 98: "yellow", 99: "blue", 100: "green"}
TOM_MARKERS = {110, 111, 112}
LANE_ORDER = ["kick", "red", "yellow", "blue", "green"]


def parse_drum_track(midi_path: Path) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """Return (onsets[(t,pitch)], tom_markers[(t,pitch)]) from PART DRUMS."""
    try:
        mid = mido.MidiFile(str(midi_path), clip=True)
    except Exception:
        return [], []
    onsets: list[tuple[float, int]] = []
    toms: list[tuple[float, int]] = []
    for track in mid.tracks:
        name = next((m.name for m in track if m.type == "track_name"), "")
        if name != "PART DRUMS":
            continue
        tempo = 500_000
        elapsed = 0.0
        for msg in track:
            elapsed += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == "set_tempo":
                tempo = msg.tempo
            elif msg.type == "note_on" and msg.velocity > 0:
                if msg.note in TOM_MARKERS:
                    toms.append((elapsed, msg.note))
                elif 96 <= msg.note <= 100:
                    onsets.append((elapsed, msg.note))
        break
    onsets.sort()
    toms.sort()
    return onsets, toms


def pitch_to_lane(pitch: int) -> str:
    return DRUM_PITCH_TO_LANE.get(pitch, "?")


def confusion_matrix(gt_dir: Path, pred_dir: Path, tol_s: float) -> np.ndarray:
    n = len(LANE_ORDER)
    cm = np.zeros((n, n), dtype=int)
    idx = {l: i for i, l in enumerate(LANE_ORDER)}
    for gt_song in sorted(gt_dir.iterdir()):
        if not gt_song.is_dir():
            continue
        gt_mid = gt_song / "notes.mid"
        pr_mid = pred_dir / gt_song.name / "notes.mid"
        if not gt_mid.exists() or not pr_mid.exists():
            continue
        gt, _ = parse_drum_track(gt_mid)
        pr, _ = parse_drum_track(pr_mid)
        used: set[int] = set()
        for g_t, g_p in gt:
            best_j = -1
            best_dt = float("inf")
            for j, (p_t, p_p) in enumerate(pr):
                if j in used:
                    continue
                dt = abs(p_t - g_t)
                if dt > tol_s:
                    if p_t > g_t + tol_s:
                        break
                    continue
                if dt < best_dt:
                    best_dt = dt
                    best_j = j
            if best_j >= 0:
                used.add(best_j)
                cm[idx[pitch_to_lane(g_p)], idx[pitch_to_lane(pr[best_j][1])]] += 1
    return cm


def write_confusion(cm: np.ndarray, out_csv: Path, out_png: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gt\\pred"] + LANE_ORDER)
        for i, lane in enumerate(LANE_ORDER):
            w.writerow([lane] + cm[i].tolist())
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.0, 4.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LANE_ORDER)))
    ax.set_yticks(range(len(LANE_ORDER)))
    ax.set_xticklabels(LANE_ORDER)
    ax.set_yticklabels(LANE_ORDER)
    ax.set_xlabel("Predicted lane")
    ax.set_ylabel("Ground-truth lane")
    ax.set_title("Drum lane confusion (row-normalised)")
    for i in range(len(LANE_ORDER)):
        for j in range(len(LANE_ORDER)):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def offset_histogram(gt_dir: Path, stem_dir: Path, out_csv: Path, out_png: Path,
                     window_ms: float = 250.0) -> None:
    """For each GT drum hit, find nearest librosa-detected audio onset on the
    drums stem and record signed delta (audio_t - chart_t)."""
    try:
        import librosa  # heavy import, only if needed
    except ImportError:
        print("  [warn] librosa not installed; skipping offset histogram")
        return
    deltas: list[float] = []
    win_s = window_ms / 1000.0
    for gt_song in sorted(gt_dir.iterdir()):
        if not gt_song.is_dir():
            continue
        gt_mid = gt_song / "notes.mid"
        if not gt_mid.exists():
            continue
        # find matching stem audio (drums.wav under stem_dir/<song>/)
        cand = list((stem_dir / gt_song.name).glob("**/drums.wav"))
        if not cand:
            cand = list((stem_dir / gt_song.name).glob("**/drums.mp3"))
        if not cand:
            continue
        try:
            y, sr = librosa.load(str(cand[0]), sr=22050, mono=True)
        except Exception:
            continue
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="time",
                                                  backtrack=True)
        gt, _ = parse_drum_track(gt_mid)
        for g_t, _ in gt:
            i = int(np.searchsorted(onset_frames, g_t))
            cands_t = []
            if i > 0:
                cands_t.append(onset_frames[i - 1])
            if i < len(onset_frames):
                cands_t.append(onset_frames[i])
            if not cands_t:
                continue
            best = min(cands_t, key=lambda x: abs(x - g_t))
            d = best - g_t
            if abs(d) <= win_s:
                deltas.append(d * 1000.0)  # ms

    if not deltas:
        print("  [warn] no offsets collected")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["delta_ms"])
        w.writerows([[round(d, 2)] for d in deltas])

    arr = np.array(deltas)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(arr, bins=80, range=(-window_ms, window_ms),
            color="#4477aa", edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(-100, color="red", linewidth=0.8, linestyle="--", label=r"$\pm$100 ms")
    ax.axvline(100, color="red", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Audio onset $-$ chart hit (ms)")
    ax.set_ylabel("Count")
    pct_in = float((np.abs(arr) <= 100).mean()) * 100
    ax.set_title(f"GT chart timing vs audio onsets  (n={len(arr)}, "
                 f"{pct_in:.1f}% within ±100 ms, "
                 f"median={np.median(arr):+.1f} ms)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def per_instrument_bar(bench_json: Path, out_png: Path) -> None:
    data = json.loads(bench_json.read_text())
    summary = data.get("summary", {})
    parts = list(summary.keys())
    if not parts:
        return
    f1  = [summary[p]["f1"]        * 100 for p in parts]
    pr  = [summary[p]["precision"] * 100 for p in parts]
    rc  = [summary[p]["recall"]    * 100 for p in parts]

    x = np.arange(len(parts))
    w = 0.27
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(x - w, pr, w, label="Precision", color="#bbbbbb")
    ax.bar(x,     rc, w, label="Recall",    color="#888888")
    ax.bar(x + w, f1, w, label="F1",        color="#4477aa")
    ax.set_xticks(x)
    ax.set_xticklabels(parts)
    ax.set_ylabel(f"Score (%) at $\\pm${data.get('tolerance_ms', 100):.0f} ms")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("STRUM end-to-end vs human chart authors")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path)
    ap.add_argument("--pred-dir", type=Path)
    ap.add_argument("--stem-dir", type=Path,
                    help="Directory containing per-song Demucs stems with drums.wav.")
    ap.add_argument("--bench-json", type=Path,
                    help="benchmark_results.json (for per-instrument bar chart).")
    ap.add_argument("--tolerance-ms", type=float, default=100.0)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tol_s = args.tolerance_ms / 1000.0

    if args.gt_dir and args.pred_dir:
        print("[1/3] confusion matrix")
        cm = confusion_matrix(args.gt_dir, args.pred_dir, tol_s)
        write_confusion(cm,
                        args.out_dir / "drum_confusion.csv",
                        args.out_dir / "drum_confusion.png")

    if args.gt_dir and args.stem_dir:
        print("[2/3] GT-vs-audio onset offset histogram")
        offset_histogram(args.gt_dir, args.stem_dir,
                         args.out_dir / "gt_audio_offsets.csv",
                         args.out_dir / "gt_audio_offsets.png")

    if args.bench_json:
        print("[3/3] per-instrument F1 bar chart")
        per_instrument_bar(args.bench_json,
                           args.out_dir / "per_instrument_f1.png")

    print(f"wrote artefacts to {args.out_dir}")


if __name__ == "__main__":
    main()
