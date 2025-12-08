# STRUM Roadmap

## Project Status: Active Development

**Start Date**: January 2026  
**Current Phase**: Milestone 1 - Pro Drums Pipeline

---

## Milestone 1: Pro Drums Pipeline ✅ IN PROGRESS

**Goal**: End-to-end pipeline for generating pro-drums charts from audio.

### Tasks

- [x] Project scaffolding (repo structure, pyproject.toml, Docker)
- [x] Context documentation (.github/copilot-instructions.md, ARCHITECTURE.md)
- [ ] Chart parsers (.mid and .chart formats)
- [ ] Demucs integration for stem separation
- [ ] Audio-chart alignment via cross-correlation
- [ ] Preprocessing CLI (`strum preprocess`)
- [ ] Dataset manifest generation with 85/15 split
- [ ] CRNN model architecture for pro drums
- [ ] Training loop with W&B integration
- [ ] Checkpoint saving (every 5 epochs + best)
- [ ] Drums inference pipeline
- [ ] .mid export with pro drums mapping
- [ ] Validation on held-out test set

### Success Criteria
- [ ] Model achieves >80% lane-wise F1 on test set
- [ ] Output charts load correctly in YARG
- [ ] Full pipeline runs via CLI commands
- [ ] Training reproducible with Docker

---

## Milestone 2: Guitar & Bass Pipeline 🔜 PLANNED

**Goal**: Rule-based reduction from Basic Pitch AMT to 5-fret game format.

### Tasks

- [ ] Basic Pitch integration on bass stem
- [ ] Basic Pitch integration on "other" stem
- [ ] Guitar vs keys classification heuristics
- [ ] Guitar-priority logic (keys only when guitar silent)
- [ ] Pitch-to-fret mapping rules
- [ ] HOPO detection (170ms threshold)
- [ ] Chord simplification rules
- [ ] Open-note assignment logic
- [ ] Validation against existing guitar/bass charts
- [ ] Rule threshold tuning

### Success Criteria
- [ ] >75% agreement with human-charted guitar/bass
- [ ] HOPO placement matches charting conventions
- [ ] No unplayable note patterns generated

---

## Milestone 3: Vocals Pipeline 🔜 PLANNED

**Goal**: Lead melody + conditional harmony extraction.

### Tasks

- [ ] Basic Pitch on vocal stem
- [ ] Lead melody extraction (loudest/most consistent)
- [ ] Harmony line clustering
- [ ] Presence ratio calculation
- [ ] Configurable harmony threshold (default 30%)
- [ ] Phrase detection for lyric alignment (future)
- [ ] Validation against existing vocal charts

### Success Criteria
- [ ] Lead melody accuracy >80% on charted songs
- [ ] Harmonies included only when meaningfully present
- [ ] Smooth pitch contours (no jitter)

---

## Milestone 4: Pro Keys Pipeline 🔜 PLANNED

**Goal**: Direct AMT-to-chart mapping for piano/keys.

### Tasks

- [ ] MT3 or Onsets & Frames integration
- [ ] Keys extraction from "other" stem
- [ ] Velocity preservation
- [ ] Full-range pro keys MIDI output
- [ ] Validation against pro keys charts

### Success Criteria
- [ ] Reasonable transcription on clear piano sections
- [ ] No major timing drift
- [ ] Velocity dynamics preserved

---

## Milestone 5: Difficulty Generation 🔜 PLANNED

**Goal**: Generate Expert/Hard/Medium/Easy from canonical charts.

### Tasks

- [ ] Density caps per difficulty level
- [ ] Note thinning algorithms
- [ ] Chord simplification per difficulty
- [ ] Ghost note removal for lower difficulties
- [ ] Drum roll simplification
- [ ] HOPO preservation rules
- [ ] Star Power phrase placement
- [ ] Per-instrument difficulty configs

### Success Criteria
- [ ] All difficulties playable and reasonable
- [ ] Smooth progression from Easy to Expert
- [ ] No illegal note patterns

---

## Milestone 6: Polish & Whitepaper 🔜 PLANNED

**Goal**: Production-ready release with documentation.

### Tasks

- [ ] Batch processing CLI (`strum batch`)
- [ ] Progress bars and error handling
- [ ] HTML/Markdown evaluation reports
- [ ] Docker multi-service compose (preprocess/train/infer)
- [ ] README with quickstart guide
- [ ] WHITEPAPER.md (problem, methodology, results, limitations)
- [ ] Architecture diagrams for whitepaper
- [ ] Sample outputs and audio/MIDI comparisons
- [ ] Performance benchmarks (GPU utilization, throughput)
- [ ] GitHub release with versioned models

### Success Criteria
- [ ] Full pipeline usable by someone with just the README
- [ ] Whitepaper suitable for blog post or portfolio
- [ ] Docker images published to registry
- [ ] Clean git history with meaningful commits

---

## Future Ideas (Post v1.0)

- [ ] Web UI for drag-and-drop charting
- [ ] Real-time preview in browser
- [ ] Custom model fine-tuning interface
- [ ] Genre-specific model variants
- [ ] Lyrics alignment for vocals
- [ ] Section detection (intro/verse/chorus)
- [ ] Automatic BPM detection improvements
- [ ] Pro guitar (string/fret inference)
- [ ] Community model sharing

---

## Metrics & Tracking

### Weights & Biases Project
- **Project**: `strum`
- **Runs**: Tagged by milestone (drums-v1, guitar-v1, etc.)

### Key Metrics to Track
| Instrument | Primary Metric | Target |
|------------|---------------|--------|
| Drums | Lane-wise F1 | >80% |
| Guitar | Chart agreement | >75% |
| Bass | Chart agreement | >75% |
| Vocals | Pitch accuracy | >80% |
| Keys | Note F1 | >70% |

### Hardware Requirements
- **Training**: NVIDIA RTX 4080 (16GB VRAM)
- **Inference**: CUDA GPU or CPU (slower)
- **Storage**: ~500GB for full dataset + stems
