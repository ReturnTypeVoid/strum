"""Train a 3-way HiHat / Ride / Crash specialist on cymbal-positive onsets.

Reuses the existing onset-window cache at
  outputs/onset_classifier/cache_community/cache
and the existing OnsetClassifier model with num_classes=3.

Usage:
    python scripts/train_cymbal_specialist.py \
        --config configs/cymbal_specialist.yaml

Outputs:
    checkpoints/cymbal_specialist/best.pt    (best test macro-F1)
    checkpoints/cymbal_specialist/last.pt    (last epoch)
    outputs/cymbal_specialist/train_log.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.onset_classifier import OnsetClassifier  # noqa: E402

CYMBAL_NAMES = ["HiHat", "Ride", "Crash"]
# 4-way mode (V2): adds an explicit NotCymbal abstention class as index 3.
NOT_CYMBAL_NAME = "NotCymbal"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CymbalSubsetDataset(Dataset):
    """Iterates only the cymbal-positive subset of the cached onset windows.

    Cache layout (mmap'd, zero-copy):
        {split}_mel_fine.npy     (N, 128, 87)  float16
        {split}_mel_coarse.npy   (N, 128, 44)  float16
        {split}_mel_lowfreq.npy  (N, 128, 44)  float16
        {split}_labels.npy       (N, 8)        uint8 multi-hot
        {split}_contexts.npy     (N, 64)       float32
    """

    def __init__(
        self,
        cache_dir: Path,
        split: str,
        cymbal_indices: tuple[int, ...] = (2, 4, 6),
        max_onsets: int | None = None,
        seed: int = 0,
        augment: bool = False,
        aug_cfg: dict | None = None,
        max_neg_onsets: int | None = None,
        with_not_cymbal: bool = False,
    ):
        cache_dir = Path(cache_dir)
        self.split = split
        self.cymbal_indices = tuple(cymbal_indices)
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        self.with_not_cymbal = with_not_cymbal
        self.num_classes = 4 if with_not_cymbal else 3

        self.mel_fine = np.load(cache_dir / f"{split}_mel_fine.npy", mmap_mode="r")
        self.mel_coarse = np.load(cache_dir / f"{split}_mel_coarse.npy", mmap_mode="r")
        self.mel_lowfreq = np.load(cache_dir / f"{split}_mel_lowfreq.npy", mmap_mode="r")
        self.labels = np.load(cache_dir / f"{split}_labels.npy", mmap_mode="r")
        self.contexts = np.load(cache_dir / f"{split}_contexts.npy", mmap_mode="r")

        # Identify cymbal-positive onsets
        cym_mask = np.zeros(self.labels.shape[0], dtype=bool)
        for ci in self.cymbal_indices:
            cym_mask |= (self.labels[:, ci] > 0)
        cym_indices = np.flatnonzero(cym_mask)
        print(f"[{split}] cymbal-positive onsets: {len(cym_indices):,} "
              f"of {self.labels.shape[0]:,} total ({100*len(cym_indices)/self.labels.shape[0]:.1f}%)")

        # Per-class breakdown
        for cls_idx, name in zip(self.cymbal_indices, CYMBAL_NAMES):
            n = int((self.labels[cym_indices, cls_idx] > 0).sum())
            print(f"  {name:6s}: {n:,}")

        rng = np.random.default_rng(seed)
        if max_onsets is not None and len(cym_indices) > max_onsets:
            cym_indices = rng.choice(cym_indices, size=max_onsets, replace=False)
            cym_indices.sort()
            print(f"[{split}] cymbal-positive subsampled to {len(cym_indices):,}")

        self.is_cym = np.ones(len(cym_indices), dtype=bool)

        if with_not_cymbal:
            neg_indices = np.flatnonzero(~cym_mask)
            print(f"[{split}] non-cymbal onsets: {len(neg_indices):,} "
                  f"({100*len(neg_indices)/self.labels.shape[0]:.1f}%)")
            if max_neg_onsets is not None and len(neg_indices) > max_neg_onsets:
                neg_indices = rng.choice(neg_indices, size=max_neg_onsets, replace=False)
                neg_indices.sort()
                print(f"[{split}] non-cymbal subsampled to {len(neg_indices):,}")
            self.indices = np.concatenate([cym_indices, neg_indices])
            self.is_cym = np.concatenate(
                [np.ones(len(cym_indices), dtype=bool),
                 np.zeros(len(neg_indices), dtype=bool)]
            )
            # Per-class non-cymbal breakdown for sanity
            from collections import Counter
            neg_label_breakdown = Counter()
            sample_n = min(50000, len(neg_indices))
            sample = rng.choice(neg_indices, size=sample_n, replace=False)
            for c in range(8):
                if c in self.cymbal_indices:
                    continue
                cnt = int((self.labels[sample, c] > 0).sum())
                neg_label_breakdown[c] = cnt
            print(f"[{split}] non-cymbal label dist (sample={sample_n}): "
                  + ", ".join(f"cls{c}={n}" for c, n in neg_label_breakdown.items()))
        else:
            self.indices = cym_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        mf = self.mel_fine[idx].astype(np.float32)        # (128, 87)
        mc = self.mel_coarse[idx].astype(np.float32)      # (128, 44)
        ml = self.mel_lowfreq[idx].astype(np.float32)     # (128, 44)
        ctx = self.contexts[idx].astype(np.float32)       # (64,)
        lbl = self.labels[idx].astype(np.float32)         # (8,)

        # Cymbal slice
        cym_target = np.array([lbl[ci] for ci in self.cymbal_indices], dtype=np.float32)
        if self.with_not_cymbal:
            # 4-way: append NotCymbal bit = 1 iff no cymbal label is set
            not_cym = 1.0 if cym_target.sum() == 0 else 0.0
            target = np.concatenate([cym_target, np.array([not_cym], dtype=np.float32)])
        else:
            target = cym_target

        if self.augment:
            mf = self._augment_mel(mf)
            mc = self._augment_mel(mc)
            ml = self._augment_mel(ml)

        return {
            "mel_fine": torch.from_numpy(mf).unsqueeze(0),     # (1, 128, 87)
            "mel_coarse": torch.from_numpy(mc).unsqueeze(0),   # (1, 128, 44)
            "mel_lowfreq": torch.from_numpy(ml).unsqueeze(0),  # (1, 128, 44)
            "context": torch.from_numpy(ctx),                  # (64,)
            "target": torch.from_numpy(target),                # (3,) or (4,)
        }

    def _augment_mel(self, m: np.ndarray) -> np.ndarray:
        cfg = self.aug_cfg
        # Additive noise on log-mel scale
        noise_std = float(cfg.get("noise_std", 0.0))
        if noise_std > 0:
            m = m + np.random.randn(*m.shape).astype(np.float32) * noise_std
        # Gain (dB) over whole window
        gain_range = cfg.get("gain_db_range")
        if gain_range:
            gain_db = float(np.random.uniform(gain_range[0], gain_range[1]))
            m = m + (gain_db / 20.0)  # log-mel ~ log magnitude -> add gain in log
        # SpecAugment masks
        if cfg.get("spec_augment", False):
            fw = int(cfg.get("freq_mask_width", 0))
            tw = int(cfg.get("time_mask_width", 0))
            if fw > 0:
                f0 = np.random.randint(0, max(1, m.shape[0] - fw))
                m[f0:f0 + fw, :] = m.mean()
            if tw > 0:
                t0 = np.random.randint(0, max(1, m.shape[1] - tw))
                m[:, t0:t0 + tw] = m.mean()
        return m


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    p_neg = (1 - p).clamp(min=clip) if clip > 0 else (1 - p)
    log_pos = torch.log(p.clamp(min=1e-8))
    log_neg = torch.log(p_neg.clamp(min=1e-8))
    loss_pos = targets * ((1 - p) ** gamma_pos) * log_pos
    loss_neg = (1 - targets) * (p ** gamma_neg) * log_neg
    loss = -(loss_pos + loss_neg) * class_weights.unsqueeze(0)
    return loss.mean()


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             num_classes: int = 3, class_names: list[str] | None = None) -> dict:
    model.eval()
    n_classes = num_classes
    names = class_names or (CYMBAL_NAMES if n_classes == 3 else CYMBAL_NAMES + [NOT_CYMBAL_NAME])
    tp = torch.zeros(n_classes, device=device)
    fp = torch.zeros(n_classes, device=device)
    fn = torch.zeros(n_classes, device=device)
    confusion = torch.zeros(n_classes, n_classes, device=device)  # arg-true vs arg-pred
    n_total = 0
    for batch in loader:
        mf = batch["mel_fine"].to(device, non_blocking=True)
        mc = batch["mel_coarse"].to(device, non_blocking=True)
        ml = batch["mel_lowfreq"].to(device, non_blocking=True)
        ctx = batch["context"].to(device, non_blocking=True)
        tgt = batch["target"].to(device, non_blocking=True)
        logits = model(mf, mc, ctx, mel_lowfreq=ml)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        tp += (pred * tgt).sum(0)
        fp += (pred * (1 - tgt)).sum(0)
        fn += ((1 - pred) * tgt).sum(0)
        # Confusion on the dominant active class per onset
        true_arg = tgt.argmax(dim=1)
        pred_arg = probs.argmax(dim=1)
        for t, p in zip(true_arg.tolist(), pred_arg.tolist()):
            confusion[t, p] += 1
        n_total += tgt.shape[0]

    metrics = {"n": n_total}
    f1s = []
    for c, name in enumerate(names):
        precision = tp[c] / (tp[c] + fp[c] + 1e-9)
        recall = tp[c] / (tp[c] + fn[c] + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        metrics[f"{name}_precision"] = float(precision.item())
        metrics[f"{name}_recall"] = float(recall.item())
        metrics[f"{name}_f1"] = float(f1.item())
        f1s.append(float(f1.item()))
    metrics["macro_f1"] = float(sum(f1s) / len(f1s))
    metrics["confusion"] = confusion.long().tolist()
    # argmax accuracy across all classes
    diag = sum(confusion[i, i].item() for i in range(n_classes))
    metrics["argmax_accuracy"] = float(diag / max(confusion.sum().item(), 1))
    # Cymbal-only macro-F1 (excludes NotCymbal) for V1<->V2 comparability
    if n_classes == 4:
        metrics["cymbal_macro_f1"] = float(sum(f1s[:3]) / 3)
    return metrics


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def build_model(cfg: dict) -> OnsetClassifier:
    m = cfg["model"]
    return OnsetClassifier(
        num_classes=m["num_classes"],
        branch_channels=list(m["branch_channels"]),
        context_size=m["context_size"],
        context_hidden=m["context_hidden"],
        classifier_hidden=m["classifier_hidden"],
        spectral_dim=m.get("spectral_dim", 32),
        dropout=m.get("dropout", 0.3),
        use_freq_attn=m.get("use_freq_attn", False),
        use_hpss=m.get("use_hpss", False),
        enhanced_spectral=m.get("enhanced_spectral", False),
        use_dual_head=False,  # 3-way doesn't need dual head
        use_lowfreq_branch=m.get("use_lowfreq_branch", False),
        use_lowfreq_spectral=m.get("use_lowfreq_spectral", False),
        context_classes=m.get("context_classes", m["num_classes"]),
    )


def cosine_schedule(step: int, total_steps: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/cymbal_specialist.yaml")
    ap.add_argument("--smoke-test", action="store_true",
                    help="Run on tiny subset for sanity check (2 epochs, 50k onsets, 1k val)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    aug_cfg = cfg.get("augmentation", {})

    out_dir = Path(paths["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(paths["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Datasets ----
    cymbal_indices = tuple(ds_cfg["cymbal_indices"])
    num_classes = int(cfg["model"]["num_classes"])
    with_not_cymbal = num_classes == 4
    class_names = CYMBAL_NAMES + ([NOT_CYMBAL_NAME] if with_not_cymbal else [])
    max_train_pos = ds_cfg.get("max_train_pos_onsets", ds_cfg.get("max_train_onsets"))
    max_train_neg = ds_cfg.get("max_train_neg_onsets")
    max_test_neg = ds_cfg.get("max_test_neg_onsets")
    if args.smoke_test:
        max_train_pos = 30_000
        max_train_neg = 20_000 if with_not_cymbal else None
        train_cfg["epochs"] = 2

    train_ds = CymbalSubsetDataset(
        cache_dir=paths["cache_dir"], split="train",
        cymbal_indices=cymbal_indices,
        max_onsets=max_train_pos,
        max_neg_onsets=max_train_neg,
        with_not_cymbal=with_not_cymbal,
        seed=0, augment=True, aug_cfg=aug_cfg,
    )
    test_ds = CymbalSubsetDataset(
        cache_dir=paths["cache_dir"], split="test",
        cymbal_indices=cymbal_indices,
        max_onsets=10_000 if args.smoke_test else None,
        max_neg_onsets=5_000 if args.smoke_test else max_test_neg,
        with_not_cymbal=with_not_cymbal,
        seed=0, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True,
        persistent_workers=train_cfg.get("num_workers", 4) > 0, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # ---- Model ----
    model = build_model(cfg).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    class_weights = torch.tensor(loss_cfg["class_weights"], dtype=torch.float32, device=device)
    use_asl = loss_cfg.get("use_asl", True)
    gamma_neg = float(loss_cfg.get("asl_gamma_neg", 4.0))
    gamma_pos = float(loss_cfg.get("asl_gamma_pos", 0.0))
    asl_clip = float(loss_cfg.get("asl_clip", 0.05))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )
    total_steps = train_cfg["epochs"] * len(train_loader)
    warmup_steps = int(train_cfg.get("warmup_epochs", 0)) * len(train_loader)
    base_lr = train_cfg["learning_rate"]
    min_lr = float(train_cfg.get("min_lr", 1e-5))
    grad_clip = float(train_cfg.get("gradient_clip", 1.0))

    log = {"config_path": args.config, "epochs": []}
    best_f1 = -1.0
    epochs_no_improve = 0
    patience = int(train_cfg.get("early_stop_patience", 8))

    global_step = 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            lr = cosine_schedule(global_step, total_steps, warmup_steps, base_lr, min_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr

            mf = batch["mel_fine"].to(device, non_blocking=True)
            mc = batch["mel_coarse"].to(device, non_blocking=True)
            ml = batch["mel_lowfreq"].to(device, non_blocking=True)
            ctx = batch["context"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            logits = model(mf, mc, ctx, mel_lowfreq=ml)
            if use_asl:
                loss = asymmetric_loss(logits, tgt, class_weights,
                                       gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=asl_clip)
            else:
                bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
                loss = (bce * class_weights.unsqueeze(0)).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            global_step += 1

            if global_step % 100 == 0:
                print(f"  step {global_step}/{total_steps}  lr={lr:.2e}  loss={loss.item():.4f}")

        train_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}  ({elapsed:.0f}s)")

        if epoch % int(train_cfg.get("eval_interval", 1)) == 0:
            metrics = evaluate(model, test_loader, device,
                               num_classes=num_classes, class_names=class_names)
            cym_f1_str = f"  cym_macro_f1={metrics.get('cymbal_macro_f1', metrics['macro_f1'])*100:.2f}%" if with_not_cymbal else ""
            print(f"  test argmax_acc={metrics['argmax_accuracy']*100:.2f}%  "
                  f"macro_f1={metrics['macro_f1']*100:.2f}%{cym_f1_str}")
            for name in class_names:
                print(f"    {name:10s}: P={metrics[f'{name}_precision']*100:5.1f}%  "
                      f"R={metrics[f'{name}_recall']*100:5.1f}%  "
                      f"F1={metrics[f'{name}_f1']*100:5.1f}%")
            cm = metrics["confusion"]
            print(f"    confusion (rows=true, cols=pred): {class_names}")
            for r, name in enumerate(class_names):
                print(f"      {name:10s}: {cm[r]}")

            log["epochs"].append({
                "epoch": epoch, "train_loss": train_loss, "lr": lr,
                "elapsed_s": elapsed, **{k: v for k, v in metrics.items() if k != "confusion"},
                "confusion": metrics["confusion"],
            })

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                epochs_no_improve = 0
                ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "metrics": {k: v for k, v in metrics.items() if k != "confusion"},
                    "best_macro_f1": best_f1,
                    "class_names": class_names,
                    "cymbal_indices": list(cymbal_indices),
                    "with_not_cymbal": with_not_cymbal,
                }
                torch.save(ckpt, ckpt_dir / "best.pt")
                print(f"  ✓ saved new best to {ckpt_dir/'best.pt'}  (macro_f1={best_f1*100:.2f}%)")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stop: {patience} epochs without macro_f1 improvement.")
                    break

        # Always keep last
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "config": cfg, "best_macro_f1": best_f1},
                   ckpt_dir / "last.pt")

        (out_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    print(f"\nDone. Best macro_f1 = {best_f1*100:.2f}%")


if __name__ == "__main__":
    main()
