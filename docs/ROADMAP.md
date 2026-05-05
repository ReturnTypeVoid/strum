# STRUM Roadmap

## Status

The core pipeline is **production**: drums + guitar + bass + vocals + keys all
generate playable Clone Hero / YARG charts with cross-instrument grid
alignment. Verified on a held-out test set of 9 paired audio + ground-truth
chart pairs (see `docs/ARCHITECTURE.md` and the Performance section in
[README.md](../README.md)).

## Shipped

- ✅ Two-stage drum onset detector + ensemble lane classifier (8 lanes)
- ✅ Six audio-coupled rescue passes (onset / cymbal / tom / drumsep)
- ✅ Hybrid guitar & bass (V2 onset CRNN + Spotify Basic Pitch + fret mapping)
- ✅ Whisper + pYIN vocal pipeline with LRCLIB lyrics
- ✅ Spectral keyboard detection with Pro Keys output
- ✅ Cross-instrument BPM refinement + phase shift + 32nd-note snap with
  per-lane roll detection
- ✅ Difficulty reduction (Expert / Hard / Medium / Easy)
- ✅ Clone Hero / YARG packaging (`notes.mid`, `song.ini`, album art)
- ✅ Backend selection via env vars (`STRUM_GUITAR_BACKEND`,
  `STRUM_BASS_BACKEND`, `STRUM_FRET_MAPPER`, `STRUM_V12C_VARIANT`)
- ✅ Training pipelines for every model with W&B logging

## In Flight

- 🔄 Per-instrument benchmark harness (drums F1 verified; guitar / bass /
  vocals / keys numbers being collected on the GT test set)
- 🔄 Hugging Face Hub model release (`opria123/strum`)
- 🔄 OCTAVE chart-editor integration

## Planned

- ☐ Whitepaper writeup (problem framing, alignment story, results,
  limitations)
- ☐ Streaming inference mode (chunked Demucs + incremental rescue passes)
- ☐ Pro Guitar (string + fret inference, not just 5-fret)
- ☐ Web demo (Hugging Face Space) — drag-and-drop a song, get a chart
- ☐ Genre-specific drum classifier variants (metal, jazz, electronic)
- ☐ Multi-take training data from charter community

## Hardware

- **Training**: NVIDIA DGX Spark (GB10, 12 GB)
- **Inference**: any CUDA GPU; CPU works but ~10× slower

## Tracking

W&B project: `strum`. Runs are tagged by model family
(`drums-v14-*`, `onset-classifier-v15-*`, `guitar-v2-*`, etc.).
