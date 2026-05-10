# Paper tooling

Scripts that produce the numbers and figures the STRUM paper depends on.
Nothing here is used by the production pipeline.

## Order of operations

1. **Dataset stats** — emit train/val/test counts, hours of audio, per-class onsets.
   ```bash
   python scripts/paper/dataset_stats.py \
       --manifest /mnt/ml-data/dataset_drums/manifest.json \
       --out      logs/dataset_stats.json
   ```

2. **Build a candidate CSV by hand** — `paper/benchmark_candidates.csv` with columns:
   `title, artist, genre, audio_path, midi_path[, duration_s, source]`.
   Aim for ~50 candidates spread across the genre tags listed in
   `build_eval_benchmark.py` (rock, metal, punk, pop, hiphop, jazz,
   country, prog, acoustic).

3. **Sample the held-out benchmark with verified holdout against train**.
   ```bash
   python scripts/paper/build_eval_benchmark.py \
       --candidates  paper/benchmark_candidates.csv \
       --train-manifest /mnt/ml-data/dataset_drums/manifest.json \
       --out         paper/benchmark_manifest.json \
       --target-per-genre 3
   ```

4. **Run inference + eval on the new benchmark**.
   Stage the audio under one dir and the GT charts under another with matching
   per-song folder names, then:
   ```bash
   python scripts/batch_pipeline.py \
       /mnt/ml-data/benchmark-audio /mnt/ml-data/benchmark-pred
   python scripts/eval_benchmark.py \
       --gt-dir    /mnt/ml-data/benchmark-gt \
       --pred-dir  /mnt/ml-data/benchmark-pred \
       --tolerance-ms 100 \
       --out       benchmark_results_v2.json
   ```

5. **Confusion matrix, offset histogram, F1 bar chart**.
   ```bash
   python scripts/paper/eval_extras.py \
       --gt-dir    /mnt/ml-data/benchmark-gt \
       --pred-dir  /mnt/ml-data/benchmark-pred \
       --stem-dir  /mnt/ml-data/benchmark-stems \
       --bench-json benchmark_results_v2.json \
       --out-dir   paper/figures
   ```

6. **Ablation grid (overnight on a GPU)**.
   ```bash
   python scripts/paper/run_ablations.py \
       --songs-dir /mnt/ml-data/benchmark-audio \
       --gt-dir    /mnt/ml-data/benchmark-gt \
       --work-dir  /mnt/ml-data/benchmark-ablations \
       --config    scripts/paper/ablations.json \
       --out       logs/ablations.json
   ```
