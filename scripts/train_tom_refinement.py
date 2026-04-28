#!/usr/bin/env python3
"""
Train the Tom Refinement CNN.

Classifies context windows as {NotTom, HighTom, LowTom, FloorTom}.
Uses mel + CQT dual-channel input with ~500ms context.
"""

import sys
import os
import time
import gc
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tom_refinement import (
    TomRefinementCNN,
    TOM_CLASS_NAMES,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


def evaluate(
    model: TomRefinementCNN,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float]:
    """Evaluate per-class precision, recall, F1."""
    tp = np.zeros(4)
    fp = np.zeros(4)
    fn = np.zeros(4)

    model.eval()
    with torch.no_grad():
        for batch_idx, (mel, cqt, labels) in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break

            mel = mel.to(device)
            cqt = cqt.to(device)
            labels = labels.to(device)

            logits = model(mel, cqt)
            preds = logits.argmax(dim=1)

            for c in range(4):
                tp[c] += ((preds == c) & (labels == c)).sum().item()
                fp[c] += ((preds == c) & (labels != c)).sum().item()
                fn[c] += ((preds != c) & (labels == c)).sum().item()

    results = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for c in range(4):
        p = tp[c] / max(tp[c] + fp[c], 1)
        r = tp[c] / max(tp[c] + fn[c], 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        results[f'{TOM_CLASS_NAMES[c]}_P'] = round(p * 100, 1)
        results[f'{TOM_CLASS_NAMES[c]}_R'] = round(r * 100, 1)
        results[f'{TOM_CLASS_NAMES[c]}_F1'] = round(f1 * 100, 1)
        if c > 0:  # skip NotTom for overall
            total_tp += tp[c]
            total_fp += fp[c]
            total_fn += fn[c]

    overall_p = total_tp / max(total_tp + total_fp, 1)
    overall_r = total_tp / max(total_tp + total_fn, 1)
    overall_f1 = 2 * overall_p * overall_r / max(overall_p + overall_r, 1e-8)
    results['Tom_Overall_P'] = round(overall_p * 100, 1)
    results['Tom_Overall_R'] = round(overall_r * 100, 1)
    results['Tom_Overall_F1'] = round(overall_f1 * 100, 1)
    results['Accuracy'] = round((tp.sum()) / max(tp.sum() + fp.sum(), 1) * 100, 1)

    return results


def train_tom_refinement():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/mnt/ml-data/tom_refinement_windows')
    parser.add_argument('--output-dir', type=str, default='outputs/tom_refinement')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/tom_refinement')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--context-frames', type=int, default=43)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--max-val-batches', type=int, default=0)
    parser.add_argument('--init-from', type=str, default='',
                        help='Path to existing checkpoint to warm-start from')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TOM REFINEMENT TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # ─── Dataset ─────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.pt"
    val_path = data_dir / "test.pt"
    print(f"Loading precomputed windows from: {data_dir}")

    train_data = torch.load(str(train_path), weights_only=False)
    val_data = torch.load(str(val_path), weights_only=False)

    # Add channel dim if missing: (N, freq, time) -> (N, 1, freq, time)
    for data in [train_data, val_data]:
        if data['mel_windows'].ndim == 3:
            data['mel_windows'] = data['mel_windows'].unsqueeze(1)
            data['cqt_windows'] = data['cqt_windows'].unsqueeze(1)

    train_dataset = torch.utils.data.TensorDataset(
        train_data['mel_windows'],
        train_data['cqt_windows'],
        train_data['labels'],
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_data['mel_windows'],
        val_data['cqt_windows'],
        val_data['labels'],
    )

    print(f"  Train: {len(train_dataset)} windows")
    print(f"  Val:   {len(val_dataset)} windows")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # ─── Model ────────────────────────────────────────────────────────
    model = TomRefinementCNN(
        n_mel=128,
        n_cqt=84,
        context_frames=args.context_frames,
        hidden_dim=128,
        dropout=0.3,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTomRefinementCNN: {n_params/1e3:.1f}K params")

    if args.init_from:
        print(f"  Warm-starting from {args.init_from}")
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        sd = ck.get('model_state_dict', ck)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"    missing={len(missing)} unexpected={len(unexpected)}")

    # ─── Class weights ────────────────────────────────────────────────
    # Count per class for weighting
    class_counts = np.zeros(4)
    for label in train_data['labels'].tolist():
        class_counts[label] += 1

    # Inverse frequency weighting, capped
    total = class_counts.sum()
    class_weights = total / (4 * class_counts + 1e-8)
    class_weights = np.clip(class_weights, 0.5, 10.0)
    class_weights = torch.FloatTensor(class_weights).to(device)

    print(f"  Class counts: {dict(zip(TOM_CLASS_NAMES, class_counts.astype(int).tolist()))}")
    print(f"  Class weights: {class_weights.cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )

    # ─── Training loop ────────────────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0
    patience = 10

    print(f"\nTraining plan:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr} → {args.min_lr}")
    print(f"  Patience: {patience}")
    print()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            file=sys.stderr,
        )

        for batch_idx, (mel, cqt, labels) in enumerate(pbar):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break

            mel = mel.to(device)
            cqt = cqt.to(device)
            labels = labels.to(device)

            logits = model(mel, cqt)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{train_correct/max(train_total,1)*100:.1f}%",
            )

        scheduler.step()

        avg_loss = train_loss / max(batch_idx + 1, 1)
        train_acc = train_correct / max(train_total, 1) * 100
        elapsed = time.time() - epoch_start

        # Evaluate
        val_results = evaluate(
            model, val_loader, device,
            max_batches=args.max_val_batches,
        )

        f1 = val_results['Tom_Overall_F1']
        print(
            f"Epoch {epoch+1}/{args.epochs} "
            f"[{elapsed:.0f}s] "
            f"loss={avg_loss:.4f} "
            f"train_acc={train_acc:.1f}% "
            f"| Tom F1={f1:.1f}% "
            f"P={val_results['Tom_Overall_P']:.1f}% "
            f"R={val_results['Tom_Overall_R']:.1f}%"
        )
        print(
            f"  HighTom: F1={val_results['HighTom_F1']}% "
            f"P={val_results['HighTom_P']}% R={val_results['HighTom_R']}%"
        )
        print(
            f"  LowTom:  F1={val_results['LowTom_F1']}% "
            f"P={val_results['LowTom_P']}% R={val_results['LowTom_R']}%"
        )
        print(
            f"  FloorTom: F1={val_results['FloorTom_F1']}% "
            f"P={val_results['FloorTom_P']}% R={val_results['FloorTom_R']}%"
        )
        print(
            f"  NotTom:  F1={val_results['NotTom_F1']}% "
            f"P={val_results['NotTom_P']}% R={val_results['NotTom_R']}%"
        )
        sys.stdout.flush()

        # Checkpoint
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'val_results': val_results,
            }, checkpoint_dir / 'best.pt')
            print(f"  ★ New best Tom F1: {best_f1:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_results': val_results,
            }, checkpoint_dir / f'epoch_{epoch+1}.pt')

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTraining complete. Best Tom F1: {best_f1:.1f}%")
    print(f"Checkpoint: {checkpoint_dir / 'best.pt'}")


if __name__ == '__main__':
    train_tom_refinement()
