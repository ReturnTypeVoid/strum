"""
Train Guitar Onset Detection Model - Stage 1 of Hybrid System

Two-stage approach:
1. ONSET DETECTION (this model): Learn WHEN notes occur from audio
2. FRET ASSIGNMENT (rule-based): Assign frets using musical heuristics

Why this works better than end-to-end:
- Onset detection is LEARNABLE: clear acoustic events in audio
- Fret assignment is SUBJECTIVE: same pitch maps to different frets based on
  charter preference, difficulty target, playability, musical context
- The v4 model peaked at 11.8% F1 because fret choices are inconsistent across charts

This model outputs:
- Binary onset probability per frame (note start vs silence)
- No fret prediction - that's handled by rule-based stage 2

Usage:
    # Preprocess (uses existing guitar_features data)
    python train_guitar_onset.py preprocess --features-dir data/guitar_features --output-dir data/onset_features

    # Train
    python train_guitar_onset.py train --features-dir data/onset_features --checkpoint-dir checkpoints/guitar_onset
    
    # Inference
    python train_guitar_onset.py infer --checkpoint checkpoints/guitar_onset/best.pt --audio song.wav --output onsets.json
"""

import argparse
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class OnsetConfig:
    # Audio
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    
    # Segments
    segment_duration_sec: float = 5.0
    segment_overlap: float = 0.5
    
    # Model - smaller than v4 since only binary output
    input_channels: int = 2  # other + bass
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    
    # Training  
    batch_size: int = 32  # Can be larger with binary output
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    
    # Loss - binary focal loss
    focal_gamma: float = 2.0
    pos_weight: float = 5.0  # Onset frames are sparse
    
    # Augmentation
    augment: bool = True
    spec_augment_freq: int = 15
    spec_augment_time: int = 40
    
    # Data split
    train_ratio: float = 0.85
    
    @property
    def frames_per_segment(self) -> int:
        return int(self.segment_duration_sec * self.sample_rate / self.hop_length)


# =============================================================================
# Model: Lightweight CRNN for Onset Detection
# =============================================================================

class OnsetCRNN(nn.Module):
    """
    Lightweight CRNN for onset detection.
    
    Input: (batch, 2, n_mels, time) - other + bass spectrograms
    Output: (batch, time) - onset probability per frame
    """
    
    def __init__(self, config: OnsetConfig):
        super().__init__()
        self.config = config
        
        # CNN encoder - extract local features
        self.cnn = nn.Sequential(
            # Block 1: (2, 128, T) -> (32, 64, T)
            nn.Conv2d(config.input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.1),
            
            # Block 2: (32, 64, T) -> (64, 32, T)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.1),
            
            # Block 3: (64, 32, T) -> (128, 16, T)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),
            
            # Block 4: (128, 16, T) -> (256, 8, T)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),
        )
        
        # After CNN: (256, 8, T) -> flatten to (256 * 8, T) = (2048, T)
        cnn_out_dim = 256 * (config.n_mels // 16)  # 256 * 8 = 2048
        
        # RNN for temporal modeling
        self.rnn = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        
        # Output head - binary onset prediction
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),  # Single output: onset probability
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, n_mels, time) input spectrograms
        Returns:
            (batch, time) onset logits
        """
        batch_size = x.size(0)
        
        # CNN: (B, 2, 128, T) -> (B, 256, 8, T)
        x = self.cnn(x)
        
        # Reshape for RNN: (B, 256, 8, T) -> (B, T, 2048)
        x = x.permute(0, 3, 1, 2)  # (B, T, 256, 8)
        x = x.reshape(batch_size, x.size(1), -1)  # (B, T, 2048)
        
        # RNN: (B, T, 2048) -> (B, T, 512)
        x, _ = self.rnn(x)
        
        # Head: (B, T, 512) -> (B, T, 1) -> (B, T)
        x = self.head(x).squeeze(-1)
        
        return x


# =============================================================================
# Dataset
# =============================================================================

class OnsetDataset(Dataset):
    """Dataset for onset detection training with full memory loading."""
    
    def __init__(
        self,
        features_dir: Path,
        song_ids: List[str],
        config: OnsetConfig,
        augment: bool = False,
        preload: bool = True,  # Load all data into memory
    ):
        self.features_dir = Path(features_dir)
        self.config = config
        self.augment = augment
        self.preload = preload
        
        # Pre-load all song data into memory
        self.song_data = {}
        
        # Build index of all segments
        self.segments = []
        
        for song_id in tqdm(song_ids, desc="Loading songs", disable=not preload):
            song_dir = self.features_dir / song_id
            onset_path = song_dir / "onsets.npy"
            
            if not onset_path.exists():
                continue
            
            # Load data
            mel_other = np.load(song_dir / "mel_other.npy")
            mel_bass = np.load(song_dir / "mel_bass.npy")
            onsets = np.load(song_dir / "onsets.npy")
            total_frames = len(onsets)
            
            # Store in memory if preload enabled
            if preload:
                self.song_data[song_id] = (mel_other, mel_bass, onsets)
            
            # Generate segment indices
            frames_per_seg = config.frames_per_segment
            hop_frames = int(frames_per_seg * (1 - config.segment_overlap))
            
            for start_frame in range(0, max(1, total_frames - frames_per_seg + 1), hop_frames):
                self.segments.append({
                    'song_id': song_id,
                    'start_frame': start_frame,
                    'total_frames': total_frames,
                })
        
        logger.info(f"Built dataset with {len(self.segments)} segments from {len(song_ids)} songs")
        if preload:
            # Calculate memory usage
            mem_mb = sum(m.nbytes + b.nbytes + o.nbytes for m, b, o in self.song_data.values()) / 1024 / 1024
            logger.info(f"Preloaded {len(self.song_data)} songs ({mem_mb:.1f} MB)")
    
    def _load_song(self, song_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load song data (from memory if preloaded)."""
        if self.preload and song_id in self.song_data:
            return self.song_data[song_id]
        
        song_dir = self.features_dir / song_id
        mel_other = np.load(song_dir / "mel_other.npy")
        mel_bass = np.load(song_dir / "mel_bass.npy") 
        onsets = np.load(song_dir / "onsets.npy")
        return mel_other, mel_bass, onsets
    
    def __len__(self) -> int:
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seg = self.segments[idx]
        
        # Load from memory
        mel_other, mel_bass, onsets = self._load_song(seg['song_id'])
        
        # Extract segment
        start = seg['start_frame']
        end = start + self.config.frames_per_segment
        
        # Pad if needed
        if end > mel_other.shape[1]:
            pad_frames = end - mel_other.shape[1]
            mel_other = np.pad(mel_other, ((0, 0), (0, pad_frames)), mode='constant')
            mel_bass = np.pad(mel_bass, ((0, 0), (0, pad_frames)), mode='constant')
            onsets = np.pad(onsets, (0, pad_frames), mode='constant')
        
        mel_other = mel_other[:, start:end]
        mel_bass = mel_bass[:, start:end]
        onsets = onsets[start:end]
        
        # Stack channels: (2, n_mels, time)
        spec = np.stack([mel_other, mel_bass], axis=0)
        
        # To tensor
        spec = torch.from_numpy(spec).float()
        onsets = torch.from_numpy(onsets).float()
        
        # Augmentation
        if self.augment:
            spec = self._augment(spec)
        
        return spec, onsets
    
    def _augment(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment."""
        # Frequency masking
        if random.random() < 0.5:
            f = random.randint(0, self.config.spec_augment_freq)
            f0 = random.randint(0, spec.size(1) - f)
            spec[:, f0:f0+f, :] = 0
        
        # Time masking
        if random.random() < 0.5:
            t = random.randint(0, self.config.spec_augment_time)
            t0 = random.randint(0, max(1, spec.size(2) - t))
            spec[:, :, t0:t0+t] = 0
        
        return spec


# =============================================================================
# Preprocessing
# =============================================================================

def preprocess_onset_labels(
    guitar_features_dir: Path,
    output_dir: Path,
    config: OnsetConfig,
):
    """
    Convert guitar labels to binary onset labels.
    
    Reads: data/guitar_features/{song_id}/labels.npy (T, 5) multi-fret labels
    Writes: data/onset_features/{song_id}/onsets.npy (T,) binary onset labels
    
    Also copies spectrograms from guitar_features.
    """
    guitar_features_dir = Path(guitar_features_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all songs
    song_dirs = [d for d in guitar_features_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(song_dirs)} songs in guitar_features")
    
    manifest = []
    total_onsets = 0
    total_frames = 0
    
    for song_dir in tqdm(song_dirs, desc="Converting labels"):
        song_id = song_dir.name
        labels_path = song_dir / "labels.npy"
        
        if not labels_path.exists():
            continue
        
        # Load multi-fret labels
        labels = np.load(labels_path)  # (T, 5)
        
        # Convert to binary onset: any fret active = onset
        onsets = (labels.sum(axis=1) > 0).astype(np.float32)  # (T,)
        
        # Create output directory
        out_song_dir = output_dir / song_id
        out_song_dir.mkdir(exist_ok=True)
        
        # Save onset labels
        np.save(out_song_dir / "onsets.npy", onsets)
        
        # Copy spectrograms (symlink to save space)
        for spec_name in ["mel_other.npy", "mel_bass.npy"]:
            src = song_dir / spec_name
            dst = out_song_dir / spec_name
            if src.exists() and not dst.exists():
                # Copy instead of symlink for Windows compatibility
                import shutil
                shutil.copy2(src, dst)
        
        # Stats
        total_onsets += onsets.sum()
        total_frames += len(onsets)
        
        manifest.append({
            'song_id': song_id,
            'num_frames': len(onsets),
            'num_onsets': int(onsets.sum()),
            'onset_density': float(onsets.mean()),
        })
    
    # Save manifest
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"Processed {len(manifest)} songs")
    logger.info(f"Total frames: {total_frames:,}")
    logger.info(f"Total onsets: {total_onsets:,.0f}")
    logger.info(f"Onset density: {total_onsets / total_frames * 100:.1f}%")


# =============================================================================
# Training
# =============================================================================

class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification."""
    
    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, time) raw logits
            targets: (batch, time) binary targets
        """
        probs = torch.sigmoid(logits)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, 
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction='none'
        )
        
        # Focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        return (focal_weight * bce).mean()


def get_split(song_ids: List[str], train_ratio: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split based on hash."""
    train_ids, test_ids = [], []
    
    for song_id in song_ids:
        h = int(hashlib.md5(song_id.encode()).hexdigest(), 16)
        if (h % 100) < train_ratio * 100:
            train_ids.append(song_id)
        else:
            test_ids.append(song_id)
    
    return train_ids, test_ids


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Compute onset detection metrics."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'threshold': threshold,
        }


def train(
    features_dir: Path,
    checkpoint_dir: Path,
    config: OnsetConfig,
    use_wandb: bool = True,
    preload: bool = True,
    resume_from: Optional[Path] = None,
):
    """Train onset detection model."""
    features_dir = Path(features_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Preload data: {preload}")
    
    # Load manifest
    with open(features_dir / "manifest.json") as f:
        manifest = json.load(f)
    
    song_ids = [m['song_id'] for m in manifest]
    train_ids, test_ids = get_split(song_ids, config.train_ratio)
    logger.info(f"Train: {len(train_ids)} songs, Test: {len(test_ids)} songs")
    
    # Datasets
    train_dataset = OnsetDataset(features_dir, train_ids, config, augment=config.augment, preload=preload)
    test_dataset = OnsetDataset(features_dir, test_ids, config, augment=False, preload=preload)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid hanging
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Model
    model = OnsetCRNN(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")
    
    # Loss, optimizer, scheduler
    criterion = BinaryFocalLoss(gamma=config.focal_gamma, pos_weight=config.pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs - config.warmup_epochs
    )
    
    scaler = GradScaler()
    
    best_f1 = 0.0
    start_epoch = 0
    
    # Resume from checkpoint
    if resume_from and resume_from.exists():
        logger.info(f"Resuming from {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_f1 = ckpt.get('best_f1', 0.0)
        logger.info(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")
    
    # W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="strum-guitar-onset",
            config={
                **config.__dict__,
                'num_params': num_params,
                'train_songs': len(train_ids),
                'test_songs': len(test_ids),
                'train_segments': len(train_dataset),
                'test_segments': len(test_dataset),
            }
        )
        wandb.watch(model, log_freq=100)
    
    for epoch in range(start_epoch, config.epochs):
        # Warmup
        if epoch < config.warmup_epochs:
            warmup_lr = config.learning_rate * (epoch + 1) / config.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        
        # Train
        model.train()
        train_loss = 0.0
        train_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (specs, targets) in enumerate(pbar):
            specs = specs.to(device)
            targets = targets.to(device)
            
            with autocast():
                logits = model(specs)
                loss = criterion(logits, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            batch_metrics = compute_metrics(logits, targets)
            for k in train_metrics:
                train_metrics[k] += batch_metrics[k]
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'f1': f"{batch_metrics['f1']:.3f}",
            })
        
        train_loss /= len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
        
        with torch.no_grad():
            for specs, targets in test_loader:
                specs = specs.to(device)
                targets = targets.to(device)
                
                with autocast():
                    logits = model(specs)
                    loss = criterion(logits, targets)
                
                val_loss += loss.item()
                batch_metrics = compute_metrics(logits, targets)
                for k in val_metrics:
                    val_metrics[k] += batch_metrics[k]
        
        val_loss /= len(test_loader)
        for k in val_metrics:
            val_metrics[k] /= len(test_loader)
        
        # Update scheduler
        if epoch >= config.warmup_epochs:
            scheduler.step()
        
        # Log
        logger.info(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, train_f1={train_metrics['f1']:.3f}, "
            f"val_loss={val_loss:.4f}, val_f1={val_metrics['f1']:.3f}"
        )
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'train/precision': train_metrics['precision'],
                'train/recall': train_metrics['recall'],
                'train/f1': train_metrics['f1'],
                'val/loss': val_loss,
                'val/precision': val_metrics['precision'],
                'val/recall': val_metrics['recall'],
                'val/f1': val_metrics['f1'],
                'lr': optimizer.param_groups[0]['lr'],
            })
        
        # Save best
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'metrics': val_metrics,
                'config': config.__dict__,
            }, checkpoint_dir / "best.pt")
            logger.info(f"Saved best model with F1={best_f1:.4f}")
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
            }, checkpoint_dir / f"epoch_{epoch+1}.pt")
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    logger.info(f"Training complete. Best F1: {best_f1:.4f}")
    return best_f1


# =============================================================================
# Inference
# =============================================================================

def load_model(checkpoint_path: Path, device: torch.device) -> Tuple[OnsetCRNN, OnsetConfig]:
    """Load trained model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = OnsetConfig(**ckpt['config'])
    model = OnsetCRNN(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def detect_onsets(
    model: OnsetCRNN,
    config: OnsetConfig,
    audio_path: Path,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Detect note onsets from audio.
    
    Returns:
        onset_times: array of onset times in seconds
    """
    # Load and separate audio (requires demucs)
    # For now, assume we have separated stems
    import librosa
    
    # Load audio
    y_other, sr = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    
    # Create mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    )
    
    y_tensor = torch.from_numpy(y_other).unsqueeze(0)
    mel = mel_transform(y_tensor)
    mel = torch.log(mel + 1e-8)
    
    # Use same for both channels (simplified - in practice use separated stems)
    spec = torch.stack([mel, mel], dim=1).squeeze(2)  # (1, 2, n_mels, T)
    
    # Predict in segments
    frames_per_seg = config.frames_per_segment
    total_frames = spec.size(-1)
    all_probs = torch.zeros(total_frames)
    counts = torch.zeros(total_frames)
    
    with torch.no_grad():
        for start in range(0, total_frames, frames_per_seg // 2):
            end = min(start + frames_per_seg, total_frames)
            segment = spec[:, :, :, start:end].to(device)
            
            # Pad if needed
            if segment.size(-1) < frames_per_seg:
                pad = frames_per_seg - segment.size(-1)
                segment = F.pad(segment, (0, pad))
            
            logits = model(segment)
            probs = torch.sigmoid(logits).cpu()
            
            seg_len = min(end - start, probs.size(-1))
            all_probs[start:start+seg_len] += probs[0, :seg_len]
            counts[start:start+seg_len] += 1
    
    # Average overlapping predictions
    all_probs /= counts.clamp(min=1)
    
    # Convert to onset times
    onset_frames = (all_probs > threshold).nonzero().squeeze(-1).numpy()
    onset_times = onset_frames * config.hop_length / config.sample_rate
    
    return onset_times


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train guitar onset detection model")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Preprocess
    preprocess_parser = subparsers.add_parser('preprocess', help='Convert labels to onset format')
    preprocess_parser.add_argument('--features-dir', type=Path, default='data/guitar_features',
                                   help='Input guitar features directory')
    preprocess_parser.add_argument('--output-dir', type=Path, default='data/onset_features',
                                   help='Output onset features directory')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train onset model')
    train_parser.add_argument('--features-dir', type=Path, default='data/onset_features',
                              help='Onset features directory')
    train_parser.add_argument('--checkpoint-dir', type=Path, default='checkpoints/guitar_onset',
                              help='Checkpoint directory')
    train_parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    train_parser.add_argument('--no-preload', action='store_true', help='Disable preloading data into RAM (slower but uses less memory)')
    train_parser.add_argument('--resume', type=Path, default=None, help='Resume from checkpoint')
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--lr', type=float, default=1e-3)
    
    # Infer
    infer_parser = subparsers.add_parser('infer', help='Detect onsets from audio')
    infer_parser.add_argument('--checkpoint', type=Path, required=True)
    infer_parser.add_argument('--audio', type=Path, required=True)
    infer_parser.add_argument('--output', type=Path, required=True)
    infer_parser.add_argument('--threshold', type=float, default=0.5)
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        config = OnsetConfig()
        preprocess_onset_labels(args.features_dir, args.output_dir, config)
    
    elif args.command == 'train':
        config = OnsetConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
        )
        train(args.features_dir, args.checkpoint_dir, config, 
              use_wandb=not args.no_wandb, 
              preload=not args.no_preload,
              resume_from=args.resume)
    
    elif args.command == 'infer':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model, config = load_model(args.checkpoint, device)
        onset_times = detect_onsets(model, config, args.audio, device, args.threshold)
        
        with open(args.output, 'w') as f:
            json.dump({'onset_times': onset_times.tolist()}, f, indent=2)
        
        logger.info(f"Detected {len(onset_times)} onsets, saved to {args.output}")
    
    import sys
    sys.exit(0)


if __name__ == '__main__':
    main()
