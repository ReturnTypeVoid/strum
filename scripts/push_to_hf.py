"""Push canonical STRUM checkpoints to Hugging Face Hub.

Usage (after `hf auth login` with a write token):

    python scripts/push_to_hf.py                      # dry-run, prints plan
    python scripts/push_to_hf.py --execute            # actually upload
    python scripts/push_to_hf.py --execute --only drums_v14

The repo layout on the Hub becomes::

    opria123/strum/
    ├── README.md                          (model card, this script writes it)
    ├── drums/
    │   ├── drums_v14/best.pt              # onset detector
    │   ├── drums_mc_onset/best.pt
    │   ├── drums_phase3/best.pt
    │   ├── drums_cymbal_onset/best_union_f1.pt
    │   └── tom_refinement_demucs/best.pt
    ├── drums_classifier_ensemble/
    │   ├── onset_classifier/best_f1.pt   (V2)
    │   ├── onset_classifier_v4/best_f1.pt
    │   ├── onset_classifier_v6/best_f1.pt
    │   ├── onset_classifier_v12_clean/best_f1.pt
    │   ├── onset_classifier_v12c_community/best_f1.pt
    │   ├── onset_classifier_v15/best_f1.pt
    │   ├── onset_classifier_v16/best_f1.pt
    │   └── onset_classifier_v17/best_f1.pt
    ├── guitar/
    │   ├── guitar_v2_onset/best.pt
    │   └── fret_mapper_v4.pt
    └── section_classifier/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "opria123/strum"
ROOT = Path(__file__).resolve().parent.parent

# (group, repo_subdir, local_path)
PLAN: list[tuple[str, str, Path]] = [
    # Drums backbone
    ("drums_v14",                  "drums/drums_v14/best.pt",                                   ROOT / "checkpoints/drums_v14/best.pt"),
    ("drums_mc_onset",             "drums/drums_mc_onset/best.pt",                              ROOT / "checkpoints/drums_mc_onset/best.pt"),
    ("drums_phase3",               "drums/drums_phase3/best.pt",                                ROOT / "checkpoints/drums_phase3/best.pt"),
    ("drums_cymbal_onset",         "drums/drums_cymbal_onset/best_union_f1.pt",                 ROOT / "checkpoints/drums_cymbal_onset/best_union_f1.pt"),
    ("tom_refinement_demucs",      "drums/tom_refinement_demucs/best.pt",                       ROOT / "checkpoints/tom_refinement_demucs/best.pt"),
    # Classifier ensemble
    ("onset_classifier",           "drums_classifier_ensemble/onset_classifier/best_f1.pt",     ROOT / "checkpoints/onset_classifier/best_f1.pt"),
    ("onset_classifier_v4",        "drums_classifier_ensemble/onset_classifier_v4/best_f1.pt",  ROOT / "checkpoints/onset_classifier_v4/best_f1.pt"),
    ("onset_classifier_v6",        "drums_classifier_ensemble/onset_classifier_v6/best_f1.pt",  ROOT / "checkpoints/onset_classifier_v6/best_f1.pt"),
    ("onset_classifier_v12_clean", "drums_classifier_ensemble/onset_classifier_v12_clean/best_f1.pt", ROOT / "checkpoints/onset_classifier_v12_clean/best_f1.pt"),
    ("onset_classifier_v12c_community", "drums_classifier_ensemble/onset_classifier_v12c_community/best_f1.pt", ROOT / "checkpoints/onset_classifier_v12c_community/best_f1.pt"),
    ("onset_classifier_v15",       "drums_classifier_ensemble/onset_classifier_v15/best_f1.pt", ROOT / "checkpoints/onset_classifier_v15/best_f1.pt"),
    ("onset_classifier_v16",       "drums_classifier_ensemble/onset_classifier_v16/best_f1.pt", ROOT / "checkpoints/onset_classifier_v16/best_f1.pt"),
    ("onset_classifier_v17",       "drums_classifier_ensemble/onset_classifier_v17/best_f1.pt", ROOT / "checkpoints/onset_classifier_v17/best_f1.pt"),
    # Guitar
    ("guitar_v2_onset",            "guitar/guitar_v2_onset/best.pt",                            ROOT / "checkpoints/guitar_v2/guitar_v2_onset/best.pt"),
    ("fret_mapper_v4",             "guitar/fret_mapper_v4.pt",                                  ROOT / "checkpoints/fret_mapper_v4.pt"),
    # Section
    ("section_classifier",         "section_classifier/best.pt",                                ROOT / "checkpoints/section_classifier/best.pt"),
    # Paper artifacts (benchmark methodology + results)
    ("paper_manifest_v4",          "paper/benchmark_manifest_v4.json",                          ROOT / "paper/benchmark_manifest_v4.json"),
    ("paper_candidates_strict",    "paper/benchmark_candidates_strict.csv",                     ROOT / "paper/benchmark_candidates_strict.csv"),
    ("paper_envelope_features",    "paper/audio_envelope_features.json",                        ROOT / "paper/audio_envelope_features.json"),
    ("benchmark_results",          "benchmark_results.json",                                    ROOT / "benchmark_results.json"),
]


MODEL_CARD = """---
license: mit
tags:
  - audio
  - music
  - midi
  - drum-transcription
  - guitar-transcription
  - clone-hero
  - yarg
library_name: pytorch
---

# STRUM — Spectral Transcription & Rhythm Understanding Model

End-to-end pipeline that turns a song (`.wav` / `.mp3` / `.ogg`) into a fully
playable Clone Hero / YARG chart package: drums, guitar, bass, vocals (with
lyrics), and keys.

Source: <https://github.com/opria123/strum>

## What's in this repo

| Folder | What it is | Used by |
|--------|------------|---------|
| `drums/drums_v14/`              | TwoStageDrumsCRNN onset detector (mel input, 22050 Hz) | `batch_infer_hybrid.py` Stage 1 |
| `drums/drums_mc_onset/`         | Multi-class onset head fine-tuned on V14 backbone | Stage-1 alt head |
| `drums/drums_phase3/`           | Phase-3 multi-class rescue model | Late-stage rescue / reclassify |
| `drums/drums_cymbal_onset/`     | Cymbal-specialist onset head | Cymbal-specific rescue |
| `drums/tom_refinement_demucs/`  | Tom vs. cymbal CNN running on Demucs drum stem | Tom/cymbal disambiguation |
| `drums_classifier_ensemble/`    | 6-model OnsetClassifier ensemble (V2, V4, V6, V12c, V15, V16) + V17 | Per-onset 8-lane classification |
| `guitar/guitar_v2_onset/`       | Guitar onset CRNN (Event F1 0.81) | Hybrid guitar pipeline |
| `guitar/fret_mapper_v4.pt`      | Pitch → 5-fret mapper (replaces librosa rule mapper) | Hybrid guitar pipeline |
| `section_classifier/`           | Verse/chorus/bridge section labeler | Chart sections |

## Performance

Held-out test set (from 3,299 human-authored Pro Drum charts):

| Component | Metric | Score |
|-----------|--------|-------|
| Drums onset detection (V14)            | Frame F1     | 93.9% |
| Drums lane classification (6-ensemble) | Per-onset F1 | 85.2% |

End-to-end vs ground-truth Clone Hero / YARG charts on an **in-envelope
benchmark** of 29 songs sampled from a 3,299-song held-out pool. Songs were
pre-screened with a single audio-feature gate (median Demucs `htdemucs_6s`
drum-stem RMS ≥ 0.018, 1 s windows at 22050 Hz mono). Eval is Expert
difficulty, ±100 ms tolerance, with a per-song global offset search
(±200 ms / 10 ms steps).

| Instrument | F1    | Precision | Recall |
|------------|-------|-----------|--------|
| Drums      | 83.8% | 82.4%     | 85.4%  |
| Guitar     | 65.1% | 74.5%     | 57.8%  |
| Bass       | 69.4% | 65.8%     | 73.4%  |
| Vocals     | 53.9% | 63.2%     | 47.0%  |

See the source repo's `benchmark_results.json` for per-song breakdown and
`scripts/eval_benchmark.py` for the harness.

## Usage

The checkpoints are loaded by the STRUM pipeline scripts. Clone the repo and
download the checkpoints into `checkpoints/` preserving the layout:

```bash
git clone https://github.com/opria123/strum
cd strum
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Pull weights from the Hub
huggingface-cli download opria123/strum --local-dir checkpoints/ \\
    --local-dir-use-symlinks False

# Run the full pipeline on a folder of audio files
python scripts/batch_pipeline.py /path/to/songs /path/to/charts
```

The pipeline expects this layout (mirrors the `drums/` and `guitar/`
subfolders here, just under `checkpoints/`):

```
checkpoints/
├── drums_v14/best.pt
├── drums_mc_onset/best.pt
├── drums_phase3/best.pt
├── drums_cymbal_onset/best_union_f1.pt
├── tom_refinement_demucs/best.pt
├── onset_classifier/best_f1.pt
├── onset_classifier_v4/best_f1.pt
├── onset_classifier_v6/best_f1.pt
├── onset_classifier_v12_clean/best_f1.pt
├── onset_classifier_v12c_community/best_f1.pt
├── onset_classifier_v15/best_f1.pt
├── onset_classifier_v16/best_f1.pt
├── onset_classifier_v17/best_f1.pt
├── guitar_v2/guitar_v2_onset/best.pt
├── fret_mapper_v4.pt
└── section_classifier/best.pt
```

A small reorganisation script `scripts/sync_from_hf.sh` in the source repo
handles the `drums/` → flat-checkpoints/ mapping.

## License

MIT. See the source repository for full attribution of the underlying
training data (Clone Hero / YARG community charters) and dependencies
(Demucs v4, librosa, OpenAI Whisper, Spotify Basic Pitch).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="actually upload (default = dry-run)")
    ap.add_argument("--only", nargs="*", help="restrict to named groups (see PLAN keys)")
    ap.add_argument("--readme-only", action="store_true", help="upload only the model card README, skip checkpoints/artifacts")
    ap.add_argument("--repo-id", default=REPO_ID)
    args = ap.parse_args()

    plan = [] if args.readme_only else (PLAN if not args.only else [p for p in PLAN if p[0] in args.only])
    if not plan and not args.readme_only:
        print(f"no entries matched --only {args.only!r}")
        sys.exit(1)

    total = 0
    print(f"target repo: {args.repo_id}")
    for group, sub, src in plan:
        if not src.exists():
            print(f"  MISS  {src}  (skip)")
            continue
        sz = src.stat().st_size
        total += sz
        print(f"  {sz / 1024 / 1024:8.1f} MB  {src}  ->  {sub}")
    print(f"total: {total / 1024 / 1024 / 1024:.2f} GB across {len(plan)} files")

    if not args.execute:
        print("\n(dry-run — re-run with --execute to actually upload)")
        return

    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", exist_ok=True)

    # Write model card to a temp file (don't pollute the repo root)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(MODEL_CARD)
        card_path = Path(fh.name)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="docs: model card",
    )
    print("uploaded README.md")

    for group, sub, src in plan:
        if not src.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=sub,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"upload {group}",
        )
        print(f"uploaded {sub}")

    print("\ndone.")


if __name__ == "__main__":
    main()
