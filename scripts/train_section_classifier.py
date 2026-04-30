"""
Train SectionClassifier on cached 2-s mel patches.

Reads /mnt/ml-data/guitar_section_cache/{split}_section_{mel,label}.npy
Outputs checkpoints/section_classifier/best.pt

Run: python scripts/train_section_classifier.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from src.models.section_classifier import SectionClassifier, count_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_section")

LABELS = ["silence", "constant_strum", "chord_stab", "lead_line", "single_notes", "mixed"]


class SectionDataset(Dataset):
    def __init__(self, cache_dir: Path, split: str):
        self.mel = np.load(cache_dir / f"{split}_section_mel.npy", mmap_mode="r")
        self.lab = np.load(cache_dir / f"{split}_section_label.npy")
        assert len(self.mel) == len(self.lab), f"size mismatch {len(self.mel)} vs {len(self.lab)}"

    def __len__(self) -> int:
        return len(self.lab)

    def __getitem__(self, i: int):
        mel = np.array(self.mel[i], dtype=np.float32)  # (n_mels, T)
        # Z-norm per sample
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        return torch.from_numpy(mel).unsqueeze(0), int(self.lab[i])


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, dict]:
    model.eval()
    n_total = 0
    n_correct = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    with torch.no_grad():
        for mel, lab in loader:
            mel = mel.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            logits = model(mel)
            pred = logits.argmax(dim=1)
            n_total += lab.size(0)
            n_correct += (pred == lab).sum().item()
            for p, l in zip(pred.tolist(), lab.tolist()):
                per_class_total[l] += 1
                if p == l:
                    per_class_correct[l] += 1
    acc = n_correct / max(n_total, 1)
    per_class = {LABELS[c]: per_class_correct[c] / max(per_class_total[c], 1) for c in per_class_total}
    return acc, per_class


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/ml-data/guitar_section_cache")
    ap.add_argument("--ckpt-dir", default="checkpoints/section_classifier")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device=%s", device)

    train_ds = SectionDataset(cache_dir, "train")
    val_ds = SectionDataset(cache_dir, "val")
    log.info("train=%d val=%d", len(train_ds), len(val_ds))

    # Class weights (inverse frequency on train)
    counts = Counter(int(l) for l in train_ds.lab)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(LABELS) * max(counts.get(c, 1), 1)) for c in range(len(LABELS))],
        dtype=torch.float32,
    ).to(device)
    log.info("class weights: %s", {LABELS[i]: round(w.item(), 3) for i, w in enumerate(weights)})

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = SectionClassifier().to(device)
    log.info("params=%d", count_params(model))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    scaler = torch.amp.GradScaler("cuda")

    best_acc = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for mel, lab in train_loader:
            mel = mel.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            optim.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(mel)
                loss = loss_fn(logits, lab)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        train_loss = sum(losses) / max(len(losses), 1)

        val_acc, per_class = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        log.info(
            "ep=%d train_loss=%.4f val_acc=%.4f per_class=%s (%.1fs)",
            ep, train_loss, val_acc,
            {k: round(v, 3) for k, v in per_class.items()}, elapsed,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": ep,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "per_class": per_class,
            }, ckpt_dir / "best.pt")
            log.info("  ↳ saved best (val_acc=%.4f)", val_acc)

    log.info("done. best val_acc=%.4f -> %s/best.pt", best_acc, ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
