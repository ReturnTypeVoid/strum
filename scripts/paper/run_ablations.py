"""Ablation harness for the paper's drums section.

For each entry in --ablations, sets the env-var overrides, runs the full
pipeline on a fixed input directory, evaluates against the GT directory,
and records the aggregate per-instrument F1.

Ablation config (JSON):
    {
      "ablations": [
        {"name": "full_system",            "env": {}},
        {"name": "no_drumsep",             "env": {"STRUM_USE_DRUMSEP": "0"}},
        {"name": "no_phase3",              "env": {"STRUM_PHASE3": "0", "STRUM_RESCUE": "0"}},
        {"name": "no_tom_refinement",      "env": {"STRUM_TOM_REFINEMENT": "0"}},
        {"name": "no_snare_hh_roll_veto",  "env": {"STRUM_SNARE_HH_ROLL_VETO": "0"}},
        {"name": "no_crash_ride_costack",  "env": {"STRUM_CR_COSTACK_VETO": "0"}},
        {"name": "no_fill_rescue",         "env": {"STRUM_FILL_RESCUE": "0"}},
        {"name": "no_mc_hybrid",           "env": {"STRUM_MC_HYBRID": "0"}}
      ]
    }

Usage:
    python scripts/paper/run_ablations.py \
        --songs-dir /mnt/ml-data/benchmark-audio \
        --gt-dir    /mnt/ml-data/benchmark-gt-charts \
        --work-dir  /mnt/ml-data/benchmark-ablations \
        --config    paper/ablations.json \
        --out       logs/ablations.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], env: dict[str, str]) -> int:
    print("  $", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs-dir", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path,
                    help="Predictions and per-ablation eval results land here.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tolerance-ms", type=float, default=100.0)
    ap.add_argument("--only", nargs="*",
                    help="Restrict to named ablation entries.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip ablations whose pred dir already has notes.mid in every song.")
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    ablations = cfg["ablations"]
    if args.only:
        ablations = [a for a in ablations if a["name"] in args.only]
    print(f"running {len(ablations)} ablations on {args.songs_dir}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for ab in ablations:
        name = ab["name"]
        env_overrides = ab.get("env", {})
        pred_dir = args.work_dir / name
        eval_out = args.work_dir / f"{name}_eval.json"
        print(f"\n=== ablation: {name} ===")
        print(f"  env: {env_overrides}")

        env = os.environ.copy()
        # Reset every flag we know about so a previous shell state doesn't bleed in.
        for k in list(env):
            if k.startswith("STRUM_"):
                env.pop(k)
        env.update({k: str(v) for k, v in env_overrides.items()})

        need_predict = True
        if args.skip_existing and pred_dir.exists():
            song_names = [p.name for p in args.songs_dir.iterdir() if p.is_file()]
            if song_names and all((pred_dir / Path(s).stem / "notes.mid").exists()
                                  for s in song_names):
                print("  predictions already present -> skipping inference")
                need_predict = False

        t0 = time.time()
        if need_predict:
            rc = run([
                sys.executable, str(ROOT / "scripts" / "batch_pipeline.py"),
                str(args.songs_dir), str(pred_dir),
            ], env)
            if rc != 0:
                print(f"  [warn] pipeline exited rc={rc}, continuing to eval anyway")
        infer_s = time.time() - t0

        rc = run([
            sys.executable, str(ROOT / "scripts" / "eval_benchmark.py"),
            "--gt-dir",   str(args.gt_dir),
            "--pred-dir", str(pred_dir),
            "--tolerance-ms", str(args.tolerance_ms),
            "--out",      str(eval_out),
        ], env)
        if rc != 0 or not eval_out.exists():
            print(f"  [warn] eval failed for {name}, skipping summary")
            continue

        eval_data = json.loads(eval_out.read_text())
        summary = eval_data.get("summary", {})
        row = {
            "name": name,
            "env": env_overrides,
            "inference_seconds": round(infer_s, 1),
            "metrics": {
                part: {
                    "f1":          round(m["f1"], 4),
                    "precision":   round(m["precision"], 4),
                    "recall":      round(m["recall"], 4),
                    "lane_acc":    round(m.get("lane_accuracy", 0.0), 4),
                    "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                }
                for part, m in summary.items() if "tp" in m
            },
        }
        results.append(row)
        print(f"  drums F1 = {row['metrics'].get('drums', {}).get('f1', 'n/a')}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "tolerance_ms": args.tolerance_ms,
        "results": results,
    }, indent=2))
    print(f"\nwrote {args.out}")

    print("\n=== Drums F1 by ablation ===")
    for r in results:
        f1 = r["metrics"].get("drums", {}).get("f1")
        print(f"  {r['name']:<28} drums F1 = {f1}")


if __name__ == "__main__":
    main()
