"""
V13 Two-Stage Drums Model — Onset Detection + Classification.

Architecture overview:
  ┌─────────────────────────────────────────────────────────┐
  │                   Mel Spectrogram                       │
  │                  (B, 1, 128, 431)                       │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │            Shared CNN Encoder (4 blocks)                │
  │  Conv2D [64→128→256→512] + BN + ReLU + MaxPool(freq)   │
  │  Output: (B, 512, 8, 431)                              │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │         Frequency Sub-Band Pooling                      │
  │  Split freq dim into 4 perceptual sub-bands             │
  │  Pool each → project to 128-D → concat = 512-D         │
  │  Output: (B, 431, 512)                                  │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │     3-Layer BiLSTM (hidden=512) + Self-Attention        │
  │  Output: (B, 431, 1024)                                 │
  └────────────────────────┬────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
  ┌───────────▼──────────┐  ┌───────────▼──────────────────┐
  │  Stage 1: Onset Det  │  │  Stage 2: Classification      │
  │  FC(1024→256)→ReLU   │  │  FC(1024→512)→ReLU            │
  │  FC(256→1) → sigmoid │  │  FC(512→8) → per-class sigmoid│
  │  Output: (B,T,1)     │  │  Output: (B,T,8)              │
  └──────────────────────┘  └──────────────────────────────┘

During training:
  - Stage 1 loss: Focal BCE on ALL frames (binary onset detection)
  - Stage 2 loss: Focal BCE ONLY at ground-truth onset frames
    (model learns to classify at every frame, but loss only backprops at onsets)

During inference:
  - Stage 1: Peak-detect onsets (threshold + NMS)
  - Stage 2: Classify each detected onset into 1 of 8 drum sounds
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowAttention(nn.Module):
    """
    Sliding window self-attention — O(n × window) instead of O(n²).
    
    Each frame attends only to frames within ±window_size//2.
    This supports long sequences (30s = 2580 frames, full songs = 15-35K)
    without the memory explosion of full attention.
    
    For drum transcription, local context (~2-5 seconds) is sufficient
    for onset detection. Musical patterns requiring longer context
    are captured by the BiLSTM's hidden state.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        window_size: int = 256,  # ~3s at 11.6ms/frame
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        assert self.head_dim * num_heads == embed_dim
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)
        Returns:
            (B, T, embed_dim)
        """
        B, T, _ = x.shape
        
        # If sequence is short enough, use full attention (faster)
        if T <= self.window_size:
            return self._full_attention(x)
        
        # Sliding window: pad, unfold, attend locally
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, heads, T, head_dim)
        
        half_w = self.window_size // 2
        
        # Pad k and v along time dimension: pad half_w on each side
        # After padding, dim2 has T + 2*half_w elements
        # unfold(size=window_size, step=1) -> (T + 2*half_w - window_size + 1) windows
        # With window_size = 2*half_w, that gives T + 1 windows — one too many
        # So pad (half_w, half_w-1) to get exactly T windows
        pad_left = half_w
        pad_right = half_w - 1  # total padded len = T + window_size - 1
        k_padded = F.pad(k, (0, 0, pad_left, pad_right), value=0)  # (B,H,T+W-1,D)
        v_padded = F.pad(v, (0, 0, pad_left, pad_right), value=0)  # (B,H,T+W-1,D)
        
        # Create a mask for padded positions (1 = valid, 0 = padded)
        padded_len = T + pad_left + pad_right  # T + window_size - 1
        mask = torch.ones(padded_len, device=x.device, dtype=torch.bool)
        mask[:pad_left] = False
        mask[T + pad_left:] = False  # last pad_right positions
        
        # Unfold into windows: (B, H, T, D, W)
        # padded_len - window_size + 1 = (T + W - 1) - W + 1 = T windows ✓
        k_unfolded = k_padded.unfold(2, self.window_size, 1)  # (B, H, T, D, W)
        v_unfolded = v_padded.unfold(2, self.window_size, 1)  # (B, H, T, D, W)
        
        # Window mask: (T, W)
        mask_windows = mask.unfold(0, self.window_size, 1)  # (T, W)
        
        # Attention scores via matmul — avoids materializing (B,H,T,W,D) intermediate
        # q (B,H,T,1,D) @ k (B,H,T,D,W) → (B,H,T,1,W) → squeeze → (B,H,T,W)
        attn = torch.matmul(q.unsqueeze(3), k_unfolded).squeeze(3) / self.scale
        
        # Mask out padded positions
        attn = attn.masked_fill(~mask_windows.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply to values via matmul: attn (B,H,T,1,W) @ v (B,H,T,W,D) → (B,H,T,1,D)
        v_windows = v_unfolded.permute(0, 1, 2, 4, 3)  # (B, H, T, W, D)
        out = torch.matmul(attn.unsqueeze(3), v_windows).squeeze(3)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(out)
    
    def _full_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Standard full attention for short sequences."""
        B, T, _ = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(out)


class FlashAttention(nn.Module):
    """
    Full self-attention using PyTorch SDPA (Flash Attention kernel).

    Casts Q/K/V to fp16 for the Flash kernel (requires head_dim <= 128),
    keeps all other computation in fp32. This enables full-song attention
    (T=20K+) in ~17 GB for a 4-min song vs 68 GB for naive full attention.

    Drop-in replacement for SlidingWindowAttention with identical
    Linear layer shapes (embed_dim x embed_dim), enabling warm-start
    from V13 checkpoints.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        assert self.head_dim <= 128, (
            f"head_dim={self.head_dim} > 128 — Flash Attention requires head_dim <= 128"
        )

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim) in fp32
        Returns:
            (B, T, embed_dim) in fp32
        """
        B, T, _ = x.shape

        # Project in fp32
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (B, heads, T, head_dim) fp32

        # Cast to fp16 for SDPA Flash kernel
        q_half = q.half()
        k_half = k.half()
        v_half = v.half()

        dp = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q_half, k_half, v_half,
            dropout_p=dp,
            is_causal=False,
        )  # (B, heads, T, head_dim) fp16

        # Back to fp32
        out = out.float()
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        return self.out_proj(out)


class FrequencySubBandPooling(nn.Module):
    """
    Split CNN output along frequency dimension into perceptual sub-bands,
    pool each sub-band, project to a fixed dimension, then concatenate.
    
    This gives the model explicit frequency-range awareness — critical for
    distinguishing toms (low-mid freq) from cymbals (high freq) that share
    the same MIDI lane.
    
    Args:
        in_channels: Number of CNN output channels (e.g., 512)
        freq_dim: Total frequency dimension after CNN pooling (e.g., 8)
        subband_boundaries: Mel bin boundaries for sub-bands (before CNN pooling)
        proj_dim: Projection dimension per sub-band
    """
    
    def __init__(
        self,
        in_channels: int,
        freq_dim: int,
        num_subbands: int = 4,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.num_subbands = num_subbands
        self.proj_dim = proj_dim
        
        # After 4x MaxPool(2,1) on freq, 128 mel → 8 freq bins
        # Split 8 bins into 4 sub-bands of 2 bins each
        self.subband_size = freq_dim // num_subbands
        assert self.subband_size * num_subbands == freq_dim, (
            f"freq_dim={freq_dim} not divisible by num_subbands={num_subbands}"
        )
        
        # Each sub-band: pool channels*subband_size → proj_dim
        subband_input_dim = in_channels * self.subband_size
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(subband_input_dim, proj_dim),
                nn.ReLU(inplace=True),
                nn.LayerNorm(proj_dim),
            )
            for _ in range(num_subbands)
        ])
        
        self.output_dim = proj_dim * num_subbands
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: CNN output (B, C, F, T) where F is the pooled freq dim
        Returns:
            (B, T, num_subbands * proj_dim) — ready for LSTM
        """
        B, C, F, T = x.shape
        
        # Permute to (B, T, C, F) for easier slicing
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        
        subband_outputs = []
        for i, proj in enumerate(self.projections):
            start = i * self.subband_size
            end = start + self.subband_size
            sb = x[:, :, :, start:end]  # (B, T, C, subband_size)
            sb = sb.reshape(B, T, -1)   # (B, T, C * subband_size)
            sb = proj(sb)               # (B, T, proj_dim)
            subband_outputs.append(sb)
        
        return torch.cat(subband_outputs, dim=-1)  # (B, T, num_subbands * proj_dim)


class TwoStageDrumsCRNN(nn.Module):
    """
    Two-stage CRNN for drum transcription.
    
    Stage 1 (Onset Detection): Class-agnostic binary onset detector.
    Stage 2 (Classification): 8-class drum sound classifier.
    
    Key insight: By separating detection from classification, the onset
    detector treats all drum hits equally (no rare-class problem for toms),
    and the classifier only needs to distinguish between drum sounds at
    known onset locations (a much more balanced problem).
    """
    
    # 8-class names
    CLASS_NAMES = [
        'Kick', 'Snare', 'HiHat', 'HighTom',
        'Ride', 'LowTom', 'Crash', 'FloorTom',
    ]
    
    def __init__(
        self,
        n_mels: int = 128,
        conv_channels: list[int] = [64, 128, 256, 512],
        freq_subbands: list[int] = [32, 64, 96, 128],
        subband_proj_dim: int = 128,
        lstm_hidden: int = 512,
        lstm_layers: int = 3,
        attention_heads: int = 8,
        attention_window: int = 256,  # Sliding window size (~3s context)
        attention_type: str = "sliding",  # "sliding" or "flash"
        dropout: float = 0.3,
        onset_detector_hidden: int = 256,
        classifier_hidden: int = 512,
        num_classes: int = 8,
        predict_velocity: bool = True,
        cymbal_onset_head: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.predict_velocity = predict_velocity
        
        # ── CNN Encoder (shared) ──
        self.conv_layers = nn.ModuleList()
        in_ch = 1
        for out_ch in conv_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 1)),  # Pool only in frequency
            ))
            in_ch = out_ch
        
        # After CNN: freq_dim = n_mels / 2^num_conv_blocks
        freq_dim = n_mels // (2 ** len(conv_channels))
        num_subbands = len(freq_subbands)
        
        # ── Frequency Sub-Band Pooling ──
        self.subband_pool = FrequencySubBandPooling(
            in_channels=conv_channels[-1],
            freq_dim=freq_dim,
            num_subbands=num_subbands,
            proj_dim=subband_proj_dim,
        )
        lstm_input_dim = self.subband_pool.output_dim  # num_subbands * proj_dim
        
        # ── BiLSTM ──
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        lstm_out_dim = lstm_hidden * 2  # Bidirectional
        
        # ── Self-Attention ──
        self.attn_norm = nn.LayerNorm(lstm_out_dim)
        if attention_type == "flash":
            self.self_attention = FlashAttention(
                embed_dim=lstm_out_dim,
                num_heads=attention_heads,
                dropout=dropout,
            )
        else:
            self.self_attention = SlidingWindowAttention(
                embed_dim=lstm_out_dim,
                num_heads=attention_heads,
                window_size=attention_window,
                dropout=dropout,
            )
        
        # ── Shared Feature Projection ──
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        feat_dim = lstm_out_dim // 2  # 512
        
        # ── Stage 1: Onset Detector ──
        self.onset_detector = nn.Sequential(
            nn.Linear(feat_dim, onset_detector_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(onset_detector_hidden, 1),
        )
        
        # ── Stage 2: 8-Class Classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )
        
        # ── Velocity Head ──
        if predict_velocity:
            self.velocity_head = nn.Sequential(
                nn.Linear(feat_dim, classifier_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, num_classes),
            )
        
        # ── Cymbal/Tom Onset Head (optional, for multi-head onset) ──
        self.has_cymbal_onset = cymbal_onset_head
        if cymbal_onset_head:
            self.cymbal_onset_detector = nn.Sequential(
                nn.Linear(feat_dim, onset_detector_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(onset_detector_hidden, 1),
            )
        
        # ── Multi-Class Onset Head (optional, direct per-class onset) ──
        self.has_multiclass_onset = False  # set via add_multiclass_onset_head()
        self.multiclass_onset_detector = None
        self._deep_mc_head = False  # set via add_deep_multiclass_head()
        
        self.dropout_layer = nn.Dropout(dropout)
        self._count_params()
    
    def add_multiclass_onset_head(self) -> None:
        """Add a multi-class onset detector head (8 per-class onset outputs).
        
        Uses the same hidden dim as the original onset detector but outputs
        8 channels instead of 1. Each channel independently predicts onset
        probability for one drum class.
        """
        # Infer dimensions from existing onset detector
        feat_dim = self.onset_detector[0].in_features
        hidden_dim = self.onset_detector[0].out_features
        dropout_p = self.dropout_layer.p
        
        self.multiclass_onset_detector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.has_multiclass_onset = True
        self._count_params()

    def add_deep_multiclass_head(self, hidden_dim: int = 256, n_conv_layers: int = 3, kernel_size: int = 5) -> None:
        """Add a deeper multi-class onset head with temporal 1D convolutions.
        
        Architecture:
          features (B, T, feat_dim)
          → transpose to (B, feat_dim, T)
          → 1D Conv blocks: Conv1D + BN + ReLU + Dropout (×n_conv_layers, residual)
          → transpose back to (B, T, hidden)
          → MLP: Linear → ReLU → Dropout → Linear → 8
        
        The Conv1D layers capture local temporal patterns (onset shape, 
        neighboring frame relationships) that the frame-independent linear head misses.
        """
        feat_dim = self.onset_detector[0].in_features  # 512
        dropout_p = self.dropout_layer.p
        pad = (kernel_size - 1) // 2  # same padding

        conv_layers = []
        in_ch = feat_dim
        for i in range(n_conv_layers):
            out_ch = hidden_dim if i > 0 else hidden_dim
            conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout_p),
            ))
            in_ch = out_ch

        self._deep_mc_convs = nn.ModuleList(conv_layers)
        # Projection for residual when dimensions change
        self._deep_mc_proj = nn.Conv1d(feat_dim, hidden_dim, 1) if feat_dim != hidden_dim else nn.Identity()
        
        self._deep_mc_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.has_multiclass_onset = True
        self._deep_mc_head = True  # flag for forward()
        self._count_params()

    def _deep_mc_forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward through the deep multi-class head.
        
        Args:
            features: (B, T, feat_dim) from feature_proj
        Returns:
            logits: (B, T, 8)
        """
        x = features.transpose(1, 2)  # (B, feat_dim, T)
        residual = self._deep_mc_proj(x)  # (B, hidden, T)
        
        for i, conv_block in enumerate(self._deep_mc_convs):
            if i == 0:
                x = conv_block(x)
            else:
                x = conv_block(x) + x  # residual within conv stack
        
        x = x + residual  # skip connection from input
        x = x.transpose(1, 2)  # (B, T, hidden)
        return self._deep_mc_mlp(x)  # (B, T, 8)
    
    def _count_params(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"TwoStageDrumsCRNN: {total/1e6:.1f}M params ({trainable/1e6:.1f}M trainable)")
    
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Mel spectrogram (B, 1, n_mels, T)
        
        Returns:
            dict with:
                'onset_logits': (B, T, 1) — binary onset logits
                'onset_probs': (B, T, 1) — binary onset probabilities
                'class_logits': (B, T, 8) — per-class logits (at every frame)
                'class_probs': (B, T, 8) — per-class probabilities
                'velocities': (B, T, 8) — velocity estimates (if enabled)
        """
        B = x.shape[0]
        
        # CNN encoder
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        # x: (B, 512, freq_dim, T)
        
        # Frequency sub-band pooling
        x = self.subband_pool(x)  # (B, T, subband_features)
        
        # BiLSTM
        x, _ = self.lstm(x)  # (B, T, 1024)
        x = self.dropout_layer(x)
        
        # Self-attention with residual
        attn_out = self.self_attention(self.attn_norm(x))
        x = x + attn_out
        
        # Shared feature projection
        features = self.feature_proj(x)  # (B, T, 512)
        
        # Stage 1: Onset detection
        onset_logits = self.onset_detector(features)  # (B, T, 1)
        onset_probs = torch.sigmoid(onset_logits)
        
        # Stage 2: Classification (runs at every frame during training)
        class_logits = self.classifier(features)  # (B, T, 8)
        class_probs = torch.sigmoid(class_logits)
        
        outputs = {
            'onset_logits': onset_logits,
            'onset_probs': onset_probs,
            'class_logits': class_logits,
            'class_probs': class_probs,
        }
        
        if self.predict_velocity:
            outputs['velocities'] = torch.sigmoid(self.velocity_head(features))
        
        if self.has_cymbal_onset:
            cymbal_logits = self.cymbal_onset_detector(features)  # (B, T, 1)
            outputs['cymbal_onset_logits'] = cymbal_logits
            outputs['cymbal_onset_probs'] = torch.sigmoid(cymbal_logits)
        
        if self.has_multiclass_onset:
            if self._deep_mc_head:
                mc_logits = self._deep_mc_forward(features)  # (B, T, 8)
            else:
                mc_logits = self.multiclass_onset_detector(features)  # (B, T, 8)
            outputs['mc_onset_logits'] = mc_logits
            outputs['mc_onset_probs'] = torch.sigmoid(mc_logits)
        
        return outputs
    
    def inference(
        self,
        x: torch.Tensor,
        onset_threshold: float = 0.5,
        class_thresholds: Optional[dict[int, float]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Inference mode: detect onsets, then classify only at onset frames.
        
        Returns combined 8-class output matching V8 format for compatibility.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
        
        B, T, _ = outputs['onset_probs'].shape
        
        if class_thresholds is None:
            class_thresholds = {i: 0.5 for i in range(self.num_classes)}
        
        # For each batch item, combine onset detection + classification
        combined = torch.zeros(B, T, self.num_classes, device=x.device)
        
        for b in range(B):
            # Find onset frames
            onset_mask = (outputs['onset_probs'][b, :, 0] > onset_threshold)
            onset_frames = onset_mask.nonzero(as_tuple=True)[0]
            
            if len(onset_frames) > 0:
                # Get classification at onset frames
                for f in onset_frames:
                    for cls in range(self.num_classes):
                        if outputs['class_probs'][b, f, cls] > class_thresholds.get(cls, 0.5):
                            combined[b, f, cls] = outputs['class_probs'][b, f, cls]
        
        return {
            'onsets': combined,
            'onset_probs': outputs['onset_probs'],
            'class_probs': outputs['class_probs'],
        }


class TwoStageLoss(nn.Module):
    """
    Combined loss for two-stage drum detection.
    
    Stage 1 Loss (onset detection):
        Focal BCE on ALL frames. Binary target: is there ANY onset?
        Uses pos_weight to handle ~2% onset density.
    
    Stage 2 Loss (classification):
        Focal BCE at ground-truth onset frames plus optional temporal
        margin. When ``classify_margin_frames > 0``, frames within
        ±margin of each onset are supervised with class_targets=0.0
        (teaching the model that sustain ≠ onset). This addresses
        crash bias where the model learns to predict crash everywhere
        because sustain frames were never supervised.
    
    Velocity Loss:
        MSE at onset frames only.
    """
    
    def __init__(
        self,
        onset_pos_weight: float = 20.0,
        onset_focal_gamma: float = 2.0,
        classify_class_weight: list[float] = None,
        classify_focal_gamma: float = 2.0,
        classify_label_smoothing: float = 0.0,
        onset_loss_weight: float = 1.0,
        classify_loss_weight: float = 2.0,
        velocity_loss_weight: float = 0.3,
        classify_margin_frames: int = 0,
        onset_class_pos_weights: list[float] = None,
    ):
        super().__init__()
        self.onset_pos_weight = onset_pos_weight
        self.onset_focal_gamma = onset_focal_gamma
        self.classify_focal_gamma = classify_focal_gamma
        self.label_smoothing = classify_label_smoothing
        self.onset_loss_weight = onset_loss_weight
        self.classify_loss_weight = classify_loss_weight
        self.velocity_loss_weight = velocity_loss_weight
        self.classify_margin_frames = classify_margin_frames
        self.onset_class_pos_weights = onset_class_pos_weights
        
        if classify_class_weight is None:
            self.classify_class_weight = [1.0] * 8
        else:
            self.classify_class_weight = classify_class_weight

    @staticmethod
    def _expand_onset_mask(onset_mask: torch.Tensor, margin: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand onset mask by ±margin frames for temporal supervision.

        Returns:
            expanded_mask: (B, T) bool — True at onset frames AND margin frames
            is_onset: (B, T) bool — True ONLY at actual onset frames
        """
        if margin <= 0:
            return onset_mask, onset_mask
        # Use max_pool1d to dilate the boolean mask
        B, T = onset_mask.shape
        mask_f = onset_mask.float().unsqueeze(1)  # (B, 1, T)
        kernel = 2 * margin + 1
        dilated = torch.nn.functional.max_pool1d(
            mask_f, kernel_size=kernel, stride=1, padding=margin,
        ).squeeze(1) > 0.5  # (B, T)
        return dilated, onset_mask
    
    def _focal_bce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        pos_weight: float,
        gamma: float,
    ) -> torch.Tensor:
        """Focal BCE with scalar pos_weight. Computed in fp32 for stability."""
        # Force fp32 to avoid fp16 underflow in focal term
        logits_f = logits.float()
        targets_f = targets.float()
        pw = torch.tensor([pos_weight], device=logits.device, dtype=torch.float32)
        bce = F.binary_cross_entropy_with_logits(
            logits_f, targets_f, pos_weight=pw, reduction='none'
        )
        probs = torch.sigmoid(logits_f)
        p_t = torch.where(targets_f > 0.5, probs, 1 - probs)
        p_t = p_t.clamp(min=1e-6, max=1 - 1e-6)
        focal = (1 - p_t) ** gamma
        return (focal * bce).mean()
    
    def _focal_bce_class_aware(
        self,
        logits: torch.Tensor,           # (B, T)
        targets: torch.Tensor,           # (B, T)
        class_targets: torch.Tensor,     # (B, T, 8)
        base_pos_weight: float,
        class_pos_weights: torch.Tensor, # (8,)
        gamma: float,
    ) -> torch.Tensor:
        """Focal BCE with per-frame pos_weight derived from which drum class is active.

        For each onset frame, the pos_weight = base_pos_weight * max(class_pos_weights[active_classes]).
        For negative frames, pos_weight doesn't matter (target=0, so pos_weight is irrelevant
        in BCE formula — it only scales the positive term).

        This makes the detector penalize missing a Ride (class 4) onset more heavily
        than missing a Kick (class 0) onset, teaching it to be more sensitive to
        high-frequency cymbal transients.
        """
        logits_f = logits.float()
        targets_f = targets.float()

        # Compute per-frame pos_weight: for onset frames, use max weight of active classes
        # class_targets: (B, T, 8), class_pos_weights: (8,)
        # For each frame, take max weight among active (>0.5) classes, default 1.0
        active_weights = class_targets.float() * class_pos_weights.unsqueeze(0).unsqueeze(0)  # (B, T, 8)
        frame_class_weight = active_weights.max(dim=-1).values  # (B, T)
        # Non-onset frames get weight 1.0 (doesn't matter, but be safe)
        frame_class_weight = torch.where(targets_f > 0.5, frame_class_weight.clamp(min=1.0), torch.ones_like(frame_class_weight))
        
        # Per-frame pos_weight = base * class_weight
        frame_pw = base_pos_weight * frame_class_weight  # (B, T)
        
        # Manual focal BCE with per-frame pos_weight
        # BCE = -[pw * t * log(σ(x)) + (1-t) * log(σ(-x))]
        log_sigmoid_pos = F.logsigmoid(logits_f)       # log σ(x)
        log_sigmoid_neg = F.logsigmoid(-logits_f)      # log σ(-x) = log(1-σ(x))
        bce = -(frame_pw * targets_f * log_sigmoid_pos + (1 - targets_f) * log_sigmoid_neg)
        
        # Focal term
        probs = torch.sigmoid(logits_f)
        p_t = torch.where(targets_f > 0.5, probs, 1 - probs)
        p_t = p_t.clamp(min=1e-6, max=1 - 1e-6)
        focal = (1 - p_t) ** gamma
        
        return (focal * bce).mean()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        onset_targets: torch.Tensor,     # (B, T, 1) binary: any onset at this frame?
        class_targets: torch.Tensor,      # (B, T, 8) per-class binary targets
        velocity_targets: torch.Tensor = None,  # (B, T, 8)
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Compute combined two-stage loss.
        
        Args:
            predictions: Model outputs dict
            onset_targets: (B, T, 1) binary onset presence
            class_targets: (B, T, 8) per-class binary
            velocity_targets: (B, T, 8) velocity values
        
        Returns:
            (total_loss, loss_dict)
        """
        onset_logits = predictions['onset_logits']  # (B, T, 1)
        class_logits = predictions['class_logits']  # (B, T, 8)
        
        # Handle frame mismatch
        min_T = min(onset_logits.shape[1], onset_targets.shape[1])
        onset_logits = onset_logits[:, :min_T]
        onset_targets = onset_targets[:, :min_T]
        class_logits = class_logits[:, :min_T]
        class_targets = class_targets[:, :min_T]
        
        losses = {}
        
        # ── Stage 1: Onset detection loss (all frames) ──
        if self.onset_class_pos_weights is not None:
            # Class-aware onset loss: weight positive frames by which
            # drum class is present, so the detector learns that missing
            # a ride is worse than missing a kick (which it already nails).
            onset_loss = self._focal_bce_class_aware(
                onset_logits.squeeze(-1),       # (B, T)
                onset_targets.squeeze(-1),       # (B, T)
                class_targets,                   # (B, T, 8)
                base_pos_weight=self.onset_pos_weight,
                class_pos_weights=torch.tensor(
                    self.onset_class_pos_weights,
                    device=onset_logits.device,
                    dtype=torch.float32,
                ),
                gamma=self.onset_focal_gamma,
            )
        else:
            onset_loss = self._focal_bce(
                onset_logits.squeeze(-1),
                onset_targets.squeeze(-1),
                pos_weight=self.onset_pos_weight,
                gamma=self.onset_focal_gamma,
            )
        losses['onset_det'] = onset_loss.item()
        
        # ── Stage 2: Classification loss (onset frames + optional margin) ──
        # Mask: frames where ANY class has a ground-truth onset
        onset_mask = onset_targets.squeeze(-1) > 0.5  # (B, T)

        # Temporal margin: expand mask to ±N frames around onsets.
        # margin frames get class_targets=0 (teaching sustain≠onset).
        if self.classify_margin_frames > 0:
            expanded_mask, is_onset = self._expand_onset_mask(
                onset_mask, self.classify_margin_frames
            )
        else:
            expanded_mask = onset_mask
            is_onset = onset_mask

        if expanded_mask.any():
            # Extract logits at expanded region
            masked_logits = class_logits[expanded_mask].float()     # (N, 8) — fp32

            # Targets: real class labels at onset frames, zeros at margin frames
            if self.classify_margin_frames > 0:
                # Start with all zeros for the expanded region
                masked_targets = torch.zeros_like(masked_logits)
                # Fill in real targets at actual onset positions
                onset_within_expanded = is_onset[expanded_mask]  # bool mask within expanded
                masked_targets[onset_within_expanded] = class_targets[onset_mask].float()
            else:
                masked_targets = class_targets[onset_mask].float()   # (N_onsets, 8) — fp32
            
            # Apply label smoothing
            if self.label_smoothing > 0:
                masked_targets = masked_targets * (1 - self.label_smoothing) + self.label_smoothing / 2
            
            # Per-class weighted focal BCE — all in fp32 for stability
            pw = torch.tensor(
                self.classify_class_weight,
                device=class_logits.device,
                dtype=torch.float32,
            ).unsqueeze(0).expand_as(masked_logits)
            
            log_sigmoid_pos = F.logsigmoid(masked_logits)
            log_sigmoid_neg = F.logsigmoid(-masked_logits)
            bce = -(pw * masked_targets * log_sigmoid_pos + (1 - masked_targets) * log_sigmoid_neg)
            
            probs = torch.sigmoid(masked_logits)
            p_t = torch.where(masked_targets > 0.5, probs, 1 - probs)
            p_t = p_t.clamp(min=1e-6, max=1 - 1e-6)
            focal = (1 - p_t) ** self.classify_focal_gamma
            
            classify_loss = (focal * bce).mean()
            
            # Per-class loss for monitoring
            with torch.no_grad():
                per_class = (focal * bce).mean(dim=0)  # (8,)
                losses['per_class'] = per_class.cpu().tolist()
        else:
            classify_loss = torch.tensor(0.0, device=class_logits.device)
            losses['per_class'] = [0.0] * 8
        
        losses['classify'] = classify_loss.item()
        
        # ── Velocity loss (onset frames only) ──
        vel_loss = torch.tensor(0.0, device=class_logits.device)
        if velocity_targets is not None and 'velocities' in predictions:
            velocity_targets = velocity_targets[:, :min_T]
            vel_preds = predictions['velocities'][:, :min_T]
            
            if onset_mask.any():
                # Expand onset_mask to match 8-class dimension
                class_mask = class_targets[:, :min_T] > 0.5  # (B, T, 8)
                if class_mask.any():
                    vel_loss = F.mse_loss(vel_preds[class_mask], velocity_targets[class_mask])
        
        losses['velocity'] = vel_loss.item()
        
        # ── Combined loss ──
        total = (
            self.onset_loss_weight * onset_loss
            + self.classify_loss_weight * classify_loss
            + self.velocity_loss_weight * vel_loss
        )
        
        return total, losses


# Quick test
if __name__ == "__main__":
    print("Testing TwoStageDrumsCRNN...")
    
    # Test with short segment (5s)
    B, n_mels, T = 4, 128, 431
    x = torch.randn(B, 1, n_mels, T)
    
    model = TwoStageDrumsCRNN()
    out = model(x)
    
    print(f"Input:        {x.shape}")
    print(f"Onset logits: {out['onset_logits'].shape}")
    print(f"Onset probs:  {out['onset_probs'].shape}")
    print(f"Class logits: {out['class_logits'].shape}")
    print(f"Class probs:  {out['class_probs'].shape}")
    print(f"Velocities:   {out['velocities'].shape}")
    
    # Test loss
    onset_tgt = (torch.rand(B, T, 1) > 0.98).float()
    class_tgt = torch.zeros(B, T, 8)
    # At onset frames, randomly assign a class
    for b in range(B):
        for t in range(T):
            if onset_tgt[b, t, 0] > 0.5:
                cls = torch.randint(0, 8, (1,)).item()
                class_tgt[b, t, cls] = 1.0
    vel_tgt = class_tgt * 0.8
    
    loss_fn = TwoStageLoss()
    total, losses = loss_fn(out, onset_tgt, class_tgt, vel_tgt)
    print(f"\nTotal loss: {total.item():.4f}")
    for k, v in losses.items():
        print(f"  {k}: {v}")
    
    # Test with long segment (30s)
    print("\nTesting 30s segment (sliding window attention)...")
    T_long = 2580  # 30s at hop=512/sr=44100
    x_long = torch.randn(2, 1, n_mels, T_long)
    out_long = model(x_long)
    print(f"  Input: {x_long.shape}")
    print(f"  onset_logits: {out_long['onset_logits'].shape}")
    print(f"  class_logits: {out_long['class_logits'].shape}")
    
    print("\n✓ All tests passed!")
