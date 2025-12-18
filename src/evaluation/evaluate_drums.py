"""
Drums model evaluation script.

Runs inference on test set and computes F1/precision/recall metrics.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from src.inference.drums import infer_drums
from src.evaluation.metrics import (
    evaluate_drums,
    DrumMetrics,
    format_metrics_report,
)
from src.preprocessing.parsers.midi_parser import DrumChart, DrumHit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ground_truth(labels_path: Path) -> DrumChart:
    """Load ground truth labels as a DrumChart."""
    with open(labels_path) as f:
        data = json.load(f)
    
    hits = [
        DrumHit(
            time_ms=h["time_ms"],
            tick=h.get("tick", 0),  # tick may not be in labels, use 0
            lane=h["lane"],
            velocity=h["velocity"],
            is_cymbal=h.get("is_cymbal", False),
        )
        for h in data["hits"]
    ]
    
    return DrumChart(hits=hits)


def evaluate_test_set(
    data_dir: Path,
    model_path: Path,
    config_path: Path,
    tolerance_ms: float = 50.0,
    max_songs: Optional[int] = None,
    auto_align: bool = True,
) -> dict:
    """
    Evaluate model on test set songs.
    
    Args:
        data_dir: Path to processed dataset (with manifest.json)
        model_path: Path to model checkpoint
        config_path: Path to inference config
        tolerance_ms: Time tolerance for matching hits
        max_songs: Limit number of songs to evaluate (for quick testing)
        auto_align: Auto-detect and correct chart start offset
        
    Returns:
        Dict with aggregate and per-song metrics
    """
    data_dir = Path(data_dir)
    
    # Load manifest
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Filter to test songs only
    test_songs = [s for s in manifest["songs"] if s["split"] == "test"]
    
    if max_songs:
        test_songs = test_songs[:max_songs]
    
    logger.info(f"Evaluating on {len(test_songs)} test songs...")
    
    # Accumulate metrics
    all_results = []
    lane_stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(5)}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_timing_errors = []
    cymbal_correct = 0
    cymbal_total = 0
    
    for song in tqdm(test_songs, desc="Evaluating"):
        song_id = song["id"]
        drums_stem = data_dir / song["stems"]["drums"]
        labels_path = data_dir / song_id / "drums_labels.json"
        
        if not labels_path.exists():
            logger.warning(f"Skipping {song_id}: no labels found")
            continue
        
        # Run inference
        try:
            predicted = infer_drums(
                audio_path=drums_stem,
                model_path=model_path,
                config_path=config_path,
            )
        except Exception as e:
            logger.warning(f"Skipping {song_id}: inference failed - {e}")
            continue
        
        # Load ground truth
        ground_truth = load_ground_truth(labels_path)
        
        # Auto-align: detect chart start offset and filter predictions
        if auto_align and predicted.hits and ground_truth.hits:
            gt_start = min(h.time_ms for h in ground_truth.hits)
            # Filter predictions to only include those after chart start - 1 second
            # This removes false positives from audio before the chart begins
            filter_start = gt_start - 1000  # Allow 1 second before first GT hit
            predicted_hits_filtered = [h for h in predicted.hits if h.time_ms >= filter_start]
            predicted = DrumChart(hits=predicted_hits_filtered)
        
        # Evaluate
        metrics = evaluate_drums(predicted, ground_truth, tolerance_ms)
        
        all_results.append({
            "song_id": song_id,
            "metrics": metrics,
        })
        
        # Accumulate
        total_tp += metrics.true_positives
        total_fp += metrics.false_positives
        total_fn += metrics.false_negatives
        
        for lane in range(5):
            # Recalculate per-lane from the full metrics
            lane_tp = int(metrics.recall[lane] * (metrics.true_positives + metrics.false_negatives) / 5) if metrics.recall[lane] > 0 else 0
            lane_fn = int((1 - metrics.recall[lane]) * (metrics.true_positives + metrics.false_negatives) / 5) if metrics.recall[lane] < 1 else 0
        
        if metrics.true_positives > 0:
            all_timing_errors.extend([metrics.mean_timing_error_ms] * metrics.true_positives)
        
        # Cymbal accuracy accumulation
        gt_cymbal_hits = sum(1 for h in ground_truth.hits if h.lane >= 2)
        cymbal_total += gt_cymbal_hits
        cymbal_correct += int(metrics.cymbal_accuracy * gt_cymbal_hits)
    
    # Compute aggregate metrics
    agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    agg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0.0
    agg_cymbal_acc = cymbal_correct / cymbal_total if cymbal_total > 0 else 0.0
    
    import numpy as np
    agg_timing = np.mean(all_timing_errors) if all_timing_errors else 0.0
    
    return {
        "config": {
            "model_path": str(model_path),
            "tolerance_ms": tolerance_ms,
            "num_test_songs": len(test_songs),
            "num_evaluated": len(all_results),
        },
        "aggregate": {
            "precision": agg_precision,
            "recall": agg_recall,
            "f1": agg_f1,
            "cymbal_accuracy": agg_cymbal_acc,
            "mean_timing_error_ms": float(agg_timing),
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
        },
        "per_song": [
            {
                "song_id": r["song_id"],
                "precision": r["metrics"].overall_precision,
                "recall": r["metrics"].overall_recall,
                "f1": r["metrics"].overall_f1,
                "tp": r["metrics"].true_positives,
                "fp": r["metrics"].false_positives,
                "fn": r["metrics"].false_negatives,
            }
            for r in all_results
        ],
    }


def print_evaluation_report(results: dict) -> None:
    """Print a formatted evaluation report."""
    agg = results["aggregate"]
    cfg = results["config"]
    
    print("\n" + "=" * 60)
    print("       DRUMS MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"\nModel: {cfg['model_path']}")
    print(f"Test songs: {cfg['num_evaluated']}/{cfg['num_test_songs']}")
    print(f"Tolerance: {cfg['tolerance_ms']} ms")
    
    print("\n" + "-" * 40)
    print("AGGREGATE METRICS")
    print("-" * 40)
    print(f"  Precision:  {agg['precision']:.3f}  ({agg['true_positives']}/{agg['true_positives'] + agg['false_positives']} hits correct)")
    print(f"  Recall:     {agg['recall']:.3f}  ({agg['true_positives']}/{agg['true_positives'] + agg['false_negatives']} hits found)")
    print(f"  F1 Score:   {agg['f1']:.3f}")
    print(f"\n  Cymbal Acc: {agg['cymbal_accuracy']:.3f}")
    print(f"  Timing Err: {agg['mean_timing_error_ms']:.1f} ms (mean)")
    
    print("\n" + "-" * 40)
    print("COUNTS")
    print("-" * 40)
    print(f"  True Positives:  {agg['true_positives']:,}")
    print(f"  False Positives: {agg['false_positives']:,}")
    print(f"  False Negatives: {agg['false_negatives']:,}")
    
    # Per-song summary (best and worst)
    per_song = sorted(results["per_song"], key=lambda x: x["f1"], reverse=True)
    
    if len(per_song) >= 5:
        print("\n" + "-" * 40)
        print("BEST 5 SONGS (by F1)")
        print("-" * 40)
        for s in per_song[:5]:
            print(f"  {s['song_id']}: F1={s['f1']:.3f} (P={s['precision']:.3f}, R={s['recall']:.3f})")
        
        print("\n" + "-" * 40)
        print("WORST 5 SONGS (by F1)")
        print("-" * 40)
        for s in per_song[-5:]:
            print(f"  {s['song_id']}: F1={s['f1']:.3f} (P={s['precision']:.3f}, R={s['recall']:.3f})")
    
    print("\n" + "=" * 60 + "\n")


def run_evaluation(
    data_dir: str = "./dataset_v2",
    model_path: str = "./checkpoints/drums/best.pt",
    config_path: str = "./configs/inference.yaml",
    tolerance_ms: float = 50.0,
    output_json: Optional[str] = None,
    max_songs: Optional[int] = None,
) -> dict:
    """
    Run full evaluation and print report.
    
    Args:
        data_dir: Path to dataset
        model_path: Path to model checkpoint
        config_path: Path to inference config
        tolerance_ms: Matching tolerance in ms
        output_json: Optional path to save results as JSON
        max_songs: Limit songs for quick testing
        
    Returns:
        Results dict
    """
    results = evaluate_test_set(
        data_dir=Path(data_dir),
        model_path=Path(model_path),
        config_path=Path(config_path),
        tolerance_ms=tolerance_ms,
        max_songs=max_songs,
    )
    
    print_evaluation_report(results)
    
    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_json}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate drums model on test set")
    parser.add_argument("--data-dir", default="./dataset_v2", help="Dataset directory")
    parser.add_argument("--model", default="./checkpoints/drums/best.pt", help="Model checkpoint")
    parser.add_argument("--config", default="./configs/inference.yaml", help="Inference config")
    parser.add_argument("--tolerance", type=float, default=50.0, help="Matching tolerance (ms)")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("--max-songs", type=int, help="Limit number of songs (for testing)")
    
    args = parser.parse_args()
    
    run_evaluation(
        data_dir=args.data_dir,
        model_path=args.model,
        config_path=args.config,
        tolerance_ms=args.tolerance,
        output_json=args.output,
        max_songs=args.max_songs,
    )
