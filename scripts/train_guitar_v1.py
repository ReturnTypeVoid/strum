#!/usr/bin/env python3
"""Train Guitar V1 — two-stage neural pipeline.

Subcommands:
    onset    train Stage 1 — GuitarOnsetCRNN (frame-level binary onset)
    fret     train Stage 2 — GuitarFretClassifier (5-bit multi-label)
    both     run onset → fret sequentially

Inputs (produced by ``scripts/preprocess_guitar_windows.py``):
    {cache}/onset/{split}_segments_mel.npy     (N, n_mels, T)  fp16
    {cache}/onset/{split}_segments_onset.npy   (N, T)          fp16  (smeared)
    {cache}/fret/{split}_window_mel.npy        (N, n_mels, W)  fp16
    {cache}/fret/{split}_window_fret.npy       (N, 5)          uint8
    {cache}/fret/{split}_window_chord.npy      (N,)            uint8

Outputs:
    {ckpt}/guitar_v1_onset/best.pt
    {ckpt}/guitar_v1_fret/best.pt
    {ckpt}/guitar_v1_{onset|fret}/last.pt
    {ckpt}/guitar_v1_{onset|fret}/history.json

Usage:
    python scripts/train_guitar_v1.py both --config configs/guitar_v1.yaml
    python scripts/train_guitar_v1.py onset --epochs 5 --batch-size 16     (override)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.guitar_v1 import (    # noqa: E402
    GuitarOnsetCRNN,
    GuitarFretClassifier,
    OnsetCRNNConfig,
    FretClassifierConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_guitar_v1")

try:
    import wandb
    _WANDB = True
except ImportError:                 # pragma: no cover
    _WANDB = False


# ─────────────────────────── Datasets ───────────────────────────────────────
class OnsetSegmentDataset(Dataset):
    """Memory-mapped onset segments. Returns (mel, target) pairs."""

    def __init__(self, cache_dir: Path, split: str,
                 specaug_freq: int = 0, specaug_time: int = 0):
        self.mel = np.load(cache_dir / "onset" / f"{split}_segments_mel.npy",
                           mmap_mode="r")
        self.tgt = np.load(cache_dir / "onset" / f"{split}_segments_onset.npy",
                           mmap_mode="r")
        assert self.mel.shape[0] == self.tgt.shape[0]
        self.specaug_freq = specaug_freq
        self.specaug_time = specaug_time

    def __len__(self) -> int:
        return self.mel.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.mel[idx].astype(np.float32))   # (n_mels, T)
        y = torch.from_numpy(self.tgt[idx].astype(np.float32))   # (T,)
        if self.specaug_freq > 0:
            f = torch.randint(1, self.specaug_freq + 1, (1,)).item()
            f0 = torch.randint(0, max(1, x.shape[0] - f), (1,)).item()
            x[f0:f0 + f, :] = x.min()
        if self.specaug_time > 0:
            t = torch.randint(1, self.specaug_time + 1, (1,)).item()
            t0 = torch.randint(0, max(1, x.shape[1] - t), (1,)).item()
            x[:, t0:t0 + t] = x.min()
        return x.unsqueeze(0), y                                  # (1, n_mels, T), (T,)


class FretWindowDataset(Dataset):
    """Memory-mapped per-onset windows. Returns (mel, fret_label, chord_flag)."""

    def __init__(self, cache_dir: Path, split: str,
                 specaug_freq: int = 0, specaug_time: int = 0):
        self.mel = np.load(cache_dir / "fret" / f"{split}_window_mel.npy",
                           mmap_mode="r")
        self.lab = np.load(cache_dir / "fret" / f"{split}_window_fret.npy",
                           mmap_mode="r")
        self.chord = np.load(cache_dir / "fret" / f"{split}_window_chord.npy",
                             mmap_mode="r")
        assert self.mel.shape[0] == self.lab.shape[0] == self.chord.shape[0]
        self.specaug_freq = specaug_freq
        self.specaug_time = specaug_time

    def __len__(self) -> int:
        return self.mel.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.mel[idx].astype(np.float32))
        y = torch.from_numpy(self.lab[idx].astype(np.float32))
        c = torch.tensor(float(self.chord[idx]))
        if self.specaug_freq > 0:
            f = torch.randint(1, self.specaug_freq + 1, (1,)).item()
            f0 = torch.randint(0, max(1, x.shape[0] - f), (1,)).item()
            x[f0:f0 + f, :] = x.min()
        if self.specaug_time > 0:
            t = torch.randint(1, self.specaug_time + 1, (1,)).item()
            t0 = torch.randint(0, max(1, x.shape[1] - t), (1,)).item()
            x[:, t0:t0 + t] = x.min()
        return x.unsqueeze(0), y, c


# ─────────────────────────── Loss helpers ───────────────────────────────────
def focal_bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                          alpha: float = 0.25, gamma: float = 2.0,
                          pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Sigmoid focal loss (binary). target same shape as logits."""
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none",
    )
    p = torch.sigmoid(logits)
    pt = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    loss = a_t * (1 - pt) ** gamma * bce
    return loss.mean()


# ─────────────────────────── Schedulers / utils ─────────────────────────────
def make_cosine_warmup(optim: torch.optim.Optimizer, warmup: int, total: int,
                       lr0: float, min_lr: float):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return (min_lr / lr0) + (1 - min_lr / lr0) * cos
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def save_ckpt(path: Path, model: nn.Module, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


# ─────────────────────────── Stage 1 trainer ────────────────────────────────
@dataclass
class StageMetrics:
    loss: float = 0.0
    f1: float = 0.0


def evaluate_onset(model: nn.Module, loader: DataLoader, device: str,
                   threshold: float = 0.4) -> StageMetrics:
    model.eval()
    tp = fp = fn = 0
    total_loss = 0.0
    n_batch = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            total_loss += loss.item()
            n_batch += 1
            pred = (torch.sigmoid(logits) >= threshold).float()
            # Frame-level F1 (smeared targets count as positives)
            tp += int(((pred > 0.5) & (y > 0.5)).sum().item())
            fp += int(((pred > 0.5) & (y <= 0.5)).sum().item())
            fn += int(((pred <= 0.5) & (y > 0.5)).sum().item())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return StageMetrics(loss=total_loss / max(n_batch, 1), f1=f1)


def train_onset(cfg: dict, args: argparse.Namespace) -> Path:
    cache_dir = Path(cfg["paths"]["cache_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "guitar_v1_onset"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    o = cfg["onset"]
    tr = o["train"]

    if args.epochs is not None:
        tr["epochs"] = args.epochs
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size

    log.info("=== Stage 1: Onset CRNN ===")
    log.info("config: epochs=%d batch=%d lr=%g", tr["epochs"],
             tr["batch_size"], tr["learning_rate"])

    train_ds = OnsetSegmentDataset(
        cache_dir, "train",
        specaug_freq=tr["spec_augment_freq"],
        specaug_time=tr["spec_augment_time"],
    )
    val_ds = OnsetSegmentDataset(cache_dir, "val")
    log.info("dataset: train=%d val=%d segments", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=tr["batch_size"], shuffle=True,
        num_workers=tr["num_workers"], pin_memory=tr["pin_memory"],
        persistent_workers=tr["num_workers"] > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr["batch_size"], shuffle=False,
        num_workers=tr["num_workers"], pin_memory=tr["pin_memory"],
        persistent_workers=tr["num_workers"] > 0,
    )

    device = args.device
    model = GuitarOnsetCRNN(OnsetCRNNConfig(**o["model"])).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model: %.2fM params", n_params / 1e6)

    pos_weight = torch.tensor([o["data"]["pos_weight"]], device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"],
                              weight_decay=tr["weight_decay"])
    total_steps = max(1, len(train_loader) * tr["epochs"])
    warmup_steps = max(1, len(train_loader) * tr["warmup_epochs"])
    sched = make_cosine_warmup(optim, warmup_steps, total_steps,
                               tr["learning_rate"], tr["min_lr"])

    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    best_f1 = -1.0
    patience = tr["early_stop_patience"]
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batch = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                logits = model(x)
                loss = focal_bce_with_logits(
                    logits, y,
                    alpha=o["data"]["pos_focal_alpha"],
                    gamma=o["data"]["pos_focal_gamma"],
                    pos_weight=pos_weight,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"])
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()
            n_batch += 1
        train_loss = running / max(n_batch, 1)
        val_m = evaluate_onset(model, val_loader, device,
                               threshold=cfg["onset"]["inference"]["peak_threshold"])
        epoch_time = time.time() - t0
        lr_now = optim.param_groups[0]["lr"]
        log.info("epoch %2d  train_loss=%.4f  val_loss=%.4f  val_f1=%.3f  "
                 "lr=%.2e  (%.1fs)",
                 epoch, train_loss, val_m.loss, val_m.f1, lr_now, epoch_time)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_m.loss, "val_f1": val_m.f1,
            "lr": lr_now, "wall_sec": epoch_time,
        })
        if args.wandb and _WANDB:
            wandb.log({"onset/train_loss": train_loss,
                       "onset/val_loss": val_m.loss,
                       "onset/val_f1": val_m.f1,
                       "onset/lr": lr_now,
                       "epoch": epoch})

        save_ckpt(ckpt_dir / "last.pt", model,
                  {"epoch": epoch, "val_f1": val_m.f1, "config": o})
        if val_m.f1 > best_f1:
            best_f1 = val_m.f1
            bad_epochs = 0
            save_ckpt(ckpt_dir / "best.pt", model,
                      {"epoch": epoch, "val_f1": val_m.f1, "config": o})
            log.info("  ↑ new best f1=%.3f → best.pt", best_f1)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("early stop after %d non-improving epochs", patience)
                break

        with (ckpt_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    return ckpt_dir / "best.pt"


# ─────────────────────────── Stage 2 trainer ────────────────────────────────
def evaluate_fret(model: nn.Module, loader: DataLoader, device: str,
                  threshold: float = 0.5) -> StageMetrics:
    model.eval()
    tp = fp = fn = 0
    total_loss = 0.0
    n_batch = 0
    with torch.no_grad():
        for x, y, _c in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out["fret_logits"], y)
            total_loss += loss.item()
            n_batch += 1
            pred = (torch.sigmoid(out["fret_logits"]) >= threshold).float()
            tp += int(((pred > 0.5) & (y > 0.5)).sum().item())
            fp += int(((pred > 0.5) & (y <= 0.5)).sum().item())
            fn += int(((pred <= 0.5) & (y > 0.5)).sum().item())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return StageMetrics(loss=total_loss / max(n_batch, 1), f1=f1)


def train_fret(cfg: dict, args: argparse.Namespace) -> Path:
    cache_dir = Path(cfg["paths"]["cache_dir"])
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"]) / "guitar_v1_fret"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fcfg = cfg["fret"]
    tr = fcfg["train"]
    if args.epochs is not None:
        tr["epochs"] = args.epochs
    if args.batch_size is not None:
        tr["batch_size"] = args.batch_size

    log.info("=== Stage 2: Fret Classifier ===")
    log.info("config: epochs=%d batch=%d lr=%g", tr["epochs"],
             tr["batch_size"], tr["learning_rate"])

    train_ds = FretWindowDataset(
        cache_dir, "train",
        specaug_freq=tr["spec_augment_freq"],
        specaug_time=tr["spec_augment_time"],
    )
    val_ds = FretWindowDataset(cache_dir, "val")
    log.info("dataset: train=%d val=%d windows", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=tr["batch_size"], shuffle=True,
        num_workers=tr["num_workers"], pin_memory=tr["pin_memory"],
        persistent_workers=tr["num_workers"] > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr["batch_size"], shuffle=False,
        num_workers=tr["num_workers"], pin_memory=tr["pin_memory"],
        persistent_workers=tr["num_workers"] > 0,
    )

    device = args.device
    model = GuitarFretClassifier(FretClassifierConfig(**fcfg["model"])).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model: %.2fM params", n_params / 1e6)

    fret_pos_weight = torch.tensor(fcfg["data"]["fret_pos_weights"], device=device)
    chord_pos_weight = torch.tensor([fcfg["data"]["chord_pos_weight"]], device=device)
    aux_w = tr["aux_chord_loss_weight"]

    optim = torch.optim.AdamW(model.parameters(), lr=tr["learning_rate"],
                              weight_decay=tr["weight_decay"])
    total_steps = max(1, len(train_loader) * tr["epochs"])
    warmup_steps = max(1, len(train_loader) * tr["warmup_epochs"])
    sched = make_cosine_warmup(optim, warmup_steps, total_steps,
                               tr["learning_rate"], tr["min_lr"])

    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))

    best_f1 = -1.0
    patience = tr["early_stop_patience"]
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batch = 0
        for x, y, c in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True).unsqueeze(-1)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                out = model(x)
                fret_loss = F.binary_cross_entropy_with_logits(
                    out["fret_logits"], y, pos_weight=fret_pos_weight,
                )
                loss = fret_loss
                if "chord_logits" in out and aux_w > 0:
                    chord_loss = F.binary_cross_entropy_with_logits(
                        out["chord_logits"], c, pos_weight=chord_pos_weight,
                    )
                    loss = loss + aux_w * chord_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), tr["gradient_clip"])
            scaler.step(optim)
            scaler.update()
            sched.step()
            running += loss.item()
            n_batch += 1
        train_loss = running / max(n_batch, 1)
        val_m = evaluate_fret(model, val_loader, device)
        epoch_time = time.time() - t0
        lr_now = optim.param_groups[0]["lr"]
        log.info("epoch %2d  train_loss=%.4f  val_loss=%.4f  val_f1=%.3f  "
                 "lr=%.2e  (%.1fs)",
                 epoch, train_loss, val_m.loss, val_m.f1, lr_now, epoch_time)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_m.loss, "val_f1": val_m.f1,
            "lr": lr_now, "wall_sec": epoch_time,
        })
        if args.wandb and _WANDB:
            wandb.log({"fret/train_loss": train_loss,
                       "fret/val_loss": val_m.loss,
                       "fret/val_f1": val_m.f1,
                       "fret/lr": lr_now,
                       "epoch": epoch})

        save_ckpt(ckpt_dir / "last.pt", model,
                  {"epoch": epoch, "val_f1": val_m.f1, "config": fcfg})
        if val_m.f1 > best_f1:
            best_f1 = val_m.f1
            bad_epochs = 0
            save_ckpt(ckpt_dir / "best.pt", model,
                      {"epoch": epoch, "val_f1": val_m.f1, "config": fcfg})
            log.info("  ↑ new best f1=%.3f → best.pt", best_f1)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("early stop after %d non-improving epochs", patience)
                break

        with (ckpt_dir / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

    return ckpt_dir / "best.pt"


# ─────────────────────────── CLI ────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["onset", "fret", "both"])
    ap.add_argument("--config", default="configs/guitar_v1.yaml")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    log.info("device=%s  config=%s", args.device, args.config)

    if args.wandb:
        if not _WANDB:
            log.warning("wandb not installed; ignoring --wandb")
        else:
            w = cfg["wandb"]
            wandb.init(project=w["project"], entity=w["entity"],
                       name=w["run_name"], tags=w["tags"], config=cfg)

    ck = Path(cfg["paths"]["checkpoint_dir"])
    ck.mkdir(parents=True, exist_ok=True)

    if args.stage in ("onset", "both"):
        train_onset(cfg, args)
    if args.stage in ("fret", "both"):
        train_fret(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
