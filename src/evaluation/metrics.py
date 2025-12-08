"""
Evaluation metrics for STRUM.

Computes F1, precision, recall, and other metrics for drum transcription.
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np

from src.preprocessing.parsers.midi_parser import DrumChart, DrumHit

logger = logging.getLogger(__name__)


@dataclass
class DrumMetrics:
    """Evaluation metrics for drum transcription."""
    
    # Per-lane metrics
    precision: dict[int, float]
    recall: dict[int, float]
    f1: dict[int, float]
    
    # Aggregate metrics
    overall_precision: float
    overall_recall: float
    overall_f1: float
    
    # Cymbal accuracy (for lanes with cymbals)
    cymbal_accuracy: float
    
    # Timing metrics
    mean_timing_error_ms: float
    std_timing_error_ms: float
    
    # Counts
    true_positives: int
    false_positives: int
    false_negatives: int


def evaluate_drums(
    predicted: DrumChart,
    ground_truth: DrumChart,
    tolerance_ms: float = 50.0,
) -> DrumMetrics:
    """
    Evaluate drum predictions against ground truth.
    
    Args:
        predicted: Predicted drum chart
        ground_truth: Ground truth drum chart
        tolerance_ms: Time tolerance for matching hits (default 50ms)
        
    Returns:
        DrumMetrics with detailed evaluation results
    """
    # Match predictions to ground truth
    matches, unmatched_pred, unmatched_gt = _match_hits(
        predicted.hits,
        ground_truth.hits,
        tolerance_ms,
    )
    
    # Compute per-lane metrics
    lane_metrics = {}
    for lane in range(5):
        lane_matches = [m for m in matches if m["pred"].lane == lane]
        lane_fp = [h for h in unmatched_pred if h.lane == lane]
        lane_fn = [h for h in unmatched_gt if h.lane == lane]
        
        tp = len(lane_matches)
        fp = len(lane_fp)
        fn = len(lane_fn)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        lane_metrics[lane] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    # Aggregate metrics
    total_tp = len(matches)
    total_fp = len(unmatched_pred)
    total_fn = len(unmatched_gt)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0 else 0.0
    )
    
    # Cymbal accuracy (only for matched hits on lanes 2-4)
    cymbal_matches = [m for m in matches if m["pred"].lane >= 2]
    if cymbal_matches:
        cymbal_correct = sum(
            1 for m in cymbal_matches
            if m["pred"].is_cymbal == m["gt"].is_cymbal
        )
        cymbal_accuracy = cymbal_correct / len(cymbal_matches)
    else:
        cymbal_accuracy = 0.0
    
    # Timing error
    timing_errors = [m["timing_error_ms"] for m in matches]
    if timing_errors:
        mean_timing_error = np.mean(timing_errors)
        std_timing_error = np.std(timing_errors)
    else:
        mean_timing_error = 0.0
        std_timing_error = 0.0
    
    return DrumMetrics(
        precision={lane: lane_metrics[lane]["precision"] for lane in range(5)},
        recall={lane: lane_metrics[lane]["recall"] for lane in range(5)},
        f1={lane: lane_metrics[lane]["f1"] for lane in range(5)},
        overall_precision=overall_precision,
        overall_recall=overall_recall,
        overall_f1=overall_f1,
        cymbal_accuracy=cymbal_accuracy,
        mean_timing_error_ms=mean_timing_error,
        std_timing_error_ms=std_timing_error,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
    )


def _match_hits(
    predicted: list[DrumHit],
    ground_truth: list[DrumHit],
    tolerance_ms: float,
) -> tuple[list[dict], list[DrumHit], list[DrumHit]]:
    """
    Match predicted hits to ground truth using greedy matching.
    
    Returns:
        - matches: List of dicts with 'pred', 'gt', 'timing_error_ms'
        - unmatched_pred: False positives
        - unmatched_gt: False negatives
    """
    matches = []
    unmatched_pred = list(predicted)
    unmatched_gt = list(ground_truth)
    
    # Sort by time
    pred_sorted = sorted(enumerate(predicted), key=lambda x: x[1].time_ms)
    gt_sorted = sorted(enumerate(ground_truth), key=lambda x: x[1].time_ms)
    
    matched_pred_indices = set()
    matched_gt_indices = set()
    
    # Greedy matching: for each GT hit, find closest prediction within tolerance
    for gt_idx, gt_hit in gt_sorted:
        best_match = None
        best_error = float("inf")
        
        for pred_idx, pred_hit in pred_sorted:
            if pred_idx in matched_pred_indices:
                continue
            
            # Must be same lane
            if pred_hit.lane != gt_hit.lane:
                continue
            
            timing_error = abs(pred_hit.time_ms - gt_hit.time_ms)
            
            if timing_error <= tolerance_ms and timing_error < best_error:
                best_match = (pred_idx, pred_hit)
                best_error = timing_error
            
            # Early exit if we've passed the tolerance window
            if pred_hit.time_ms > gt_hit.time_ms + tolerance_ms:
                break
        
        if best_match is not None:
            pred_idx, pred_hit = best_match
            matched_pred_indices.add(pred_idx)
            matched_gt_indices.add(gt_idx)
            
            matches.append({
                "pred": pred_hit,
                "gt": gt_hit,
                "timing_error_ms": best_error,
            })
    
    # Compute unmatched
    unmatched_pred = [
        pred_hit for idx, pred_hit in enumerate(predicted)
        if idx not in matched_pred_indices
    ]
    unmatched_gt = [
        gt_hit for idx, gt_hit in enumerate(ground_truth)
        if idx not in matched_gt_indices
    ]
    
    return matches, unmatched_pred, unmatched_gt


def evaluate_batch(
    predictions: list[DrumChart],
    ground_truths: list[DrumChart],
    song_names: Optional[list[str]] = None,
    tolerance_ms: float = 50.0,
) -> dict:
    """
    Evaluate a batch of predictions.
    
    Args:
        predictions: List of predicted charts
        ground_truths: List of ground truth charts
        song_names: Optional song names for logging
        tolerance_ms: Matching tolerance
        
    Returns:
        Dict with aggregate metrics and per-song breakdown
    """
    if song_names is None:
        song_names = [f"song_{i}" for i in range(len(predictions))]
    
    per_song = {}
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_timing_errors = []
    all_cymbal_correct = 0
    all_cymbal_total = 0
    
    for name, pred, gt in zip(song_names, predictions, ground_truths):
        metrics = evaluate_drums(pred, gt, tolerance_ms)
        per_song[name] = metrics
        
        all_tp += metrics.true_positives
        all_fp += metrics.false_positives
        all_fn += metrics.false_negatives
        
        # Accumulate timing errors (approximation)
        if metrics.true_positives > 0:
            all_timing_errors.extend([metrics.mean_timing_error_ms] * metrics.true_positives)
        
        # Accumulate cymbal stats
        cymbal_hits = sum(
            1 for h in gt.hits if h.lane >= 2
        )
        all_cymbal_total += cymbal_hits
        all_cymbal_correct += int(metrics.cymbal_accuracy * cymbal_hits)
    
    # Aggregate
    agg_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    agg_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    agg_f1 = (
        2 * agg_precision * agg_recall / (agg_precision + agg_recall)
        if (agg_precision + agg_recall) > 0 else 0.0
    )
    agg_cymbal_acc = (
        all_cymbal_correct / all_cymbal_total
        if all_cymbal_total > 0 else 0.0
    )
    agg_timing_error = np.mean(all_timing_errors) if all_timing_errors else 0.0
    
    return {
        "aggregate": {
            "precision": agg_precision,
            "recall": agg_recall,
            "f1": agg_f1,
            "cymbal_accuracy": agg_cymbal_acc,
            "mean_timing_error_ms": agg_timing_error,
            "total_true_positives": all_tp,
            "total_false_positives": all_fp,
            "total_false_negatives": all_fn,
        },
        "per_song": per_song,
    }


def format_metrics_report(metrics: DrumMetrics) -> str:
    """Format metrics as a human-readable report."""
    lines = [
        "=== Drum Transcription Evaluation ===",
        "",
        "Overall Metrics:",
        f"  Precision: {metrics.overall_precision:.3f}",
        f"  Recall:    {metrics.overall_recall:.3f}",
        f"  F1 Score:  {metrics.overall_f1:.3f}",
        "",
        "Per-Lane F1:",
        f"  Kick:   {metrics.f1[0]:.3f}",
        f"  Snare:  {metrics.f1[1]:.3f}",
        f"  Yellow: {metrics.f1[2]:.3f}",
        f"  Blue:   {metrics.f1[3]:.3f}",
        f"  Green:  {metrics.f1[4]:.3f}",
        "",
        f"Cymbal Accuracy: {metrics.cymbal_accuracy:.3f}",
        "",
        "Timing:",
        f"  Mean Error: {metrics.mean_timing_error_ms:.1f} ms",
        f"  Std Error:  {metrics.std_timing_error_ms:.1f} ms",
        "",
        "Counts:",
        f"  True Positives:  {metrics.true_positives}",
        f"  False Positives: {metrics.false_positives}",
        f"  False Negatives: {metrics.false_negatives}",
    ]
    
    return "\n".join(lines)
