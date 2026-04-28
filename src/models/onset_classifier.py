"""
Onset Window Classifier V2 — Purpose-built 8-class drum hit classifier.

Improvements over V1:
  - SE (Squeeze-and-Excitation) channel attention in ResNet blocks
  - Spectral feature extractor: centroid, bandwidth, band energies computed
    directly from mel input — provides explicit frequency discrimination
    for confused pairs (HighTom↔Snare, FloorTom↔Kick, Crash↔Ride)
  - Deeper classifier head (3 hidden layers, 512 dim)
  - Longer onset windows supported (100ms+400ms)

Architecture:
  - Dual-resolution mel input: fine (n_fft=1024) + coarse (n_fft=4096)
  - ResNet-style CNN with SE attention per resolution
  - Spectral feature extractor (from mel)
  - Onset context encoder (neighboring hit pattern)
  - Deep classification head with 8-class sigmoid output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        w = x.mean(dim=(-2, -1))  # (B, C)
        w = self.fc(w).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class ResBlock(nn.Module):
    """Residual block with SE attention and optional downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + self.shortcut(x))
        return out


class FrequencyAttention(nn.Module):
    """Learnable attention over frequency bands in the mel spectrogram.
    
    Helps the model focus on discriminative frequency regions:
    - Low bands for Kick vs LowTom/FloorTom discrimination  
    - Mid bands for Snare vs HighTom discrimination
    - High bands for HiHat vs Ride vs Crash discrimination
    """

    def __init__(self, channels: int, freq_bins: int):
        super().__init__()
        self.freq_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((freq_bins, 1)),  # pool time, keep freq
            nn.Flatten(1),  # (B, C * freq_bins)
            nn.Linear(channels * freq_bins, freq_bins),
            nn.ReLU(),
            nn.Linear(freq_bins, freq_bins),
            nn.Sigmoid(),
        )
        self.freq_bins = freq_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        B, C, F, T = x.shape
        # Pool to small freq resolution for efficiency
        attn = self.freq_attn(x)  # (B, freq_bins)
        # Interpolate back to original F
        attn = attn.unsqueeze(1).unsqueeze(-1)  # (B, 1, freq_bins, 1)
        attn = nn.functional.interpolate(attn, size=(F, 1), mode='bilinear', align_corners=False)
        return x * attn  # (B, C, F, T)


class HPSSLayer(nn.Module):
    """Approximate Harmonic-Percussive Source Separation via average pooling.

    Splits a mel spectrogram into harmonic (temporally stable) and percussive
    (spectrally broad) components using 1D average pooling. This provides
    the CNN with explicit tonal-vs-percussive discrimination:
      - Toms → strong harmonic channel (sustained pitch)
      - Snare/Kick → strong percussive channel (broadband transient)
    """

    def __init__(self, time_kernel: int = 11, freq_kernel: int = 11):
        super().__init__()
        self.time_kernel = time_kernel
        self.freq_kernel = freq_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, F, T)
        # Harmonic: smooth along time → preserves sustained-frequency content
        pad_t = self.time_kernel // 2
        harmonic = F.avg_pool2d(x, kernel_size=(1, self.time_kernel),
                                stride=1, padding=(0, pad_t))
        # Percussive: smooth along frequency → preserves broadband transients
        pad_f = self.freq_kernel // 2
        percussive = F.avg_pool2d(x, kernel_size=(self.freq_kernel, 1),
                                  stride=1, padding=(pad_f, 0))
        # Stack: [original, harmonic, percussive]
        return torch.cat([x, harmonic, percussive], dim=1)  # (B, 3, F, T)


class ProjectionHead(nn.Module):
    """MLP projection head for supervised contrastive learning.

    Projects fused embeddings to a lower-dimensional L2-normalized space
    where contrastive loss encourages same-class clustering and pushes
    apart different-class embeddings — critical for disentangling rare
    classes (HighTom, LowTom, FloorTom) from dominant ones.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)  # L2-normalized for cosine similarity


class SpectralBranch(nn.Module):
    """CNN feature extractor for one mel resolution with SE + frequency attention."""

    def __init__(self, channels: list[int] = [1, 32, 64, 128, 256],
                 use_freq_attn: bool = True, in_channels: int | None = None):
        super().__init__()
        channels = list(channels)
        if in_channels is not None:
            channels[0] = in_channels
        layers = []
        for i in range(len(channels) - 1):
            stride = 2 if i < len(channels) - 2 else 1
            layers.append(ResBlock(channels[i], channels[i + 1], stride=stride, use_se=True))
        self.blocks = nn.Sequential(*layers)
        self.out_channels = channels[-1]
        # Frequency attention after CNN blocks, before pooling
        # After 3 stride-2 layers, freq is roughly n_mels/8
        self.freq_attn = FrequencyAttention(channels[-1], freq_bins=8) if use_freq_attn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, C_in, n_mels, T) → Output: (B, C) global features."""
        x = self.blocks(x)
        if self.freq_attn is not None:
            x = self.freq_attn(x)
        x = x.mean(dim=(-2, -1))  # (B, C)
        return x


class SpectralFeatureExtractor(nn.Module):
    """
    Compute explicit spectral features from mel spectrograms.

    These features directly encode the frequency information that helps
    distinguish confused pairs:
      - Spectral centroid: kick is low, snare is mid, hat is high
      - Spectral bandwidth: toms are narrow, cymbals are wide
      - Band energies (4 bands): explicit low/mid/high energy ratios
      - Attack slope: how fast energy rises in each band
      - Temporal centroid: where in time the energy peaks (quick vs sustained)

    All computed from log-mel, no extra preprocessing needed.
    """

    def __init__(self, n_mels: int = 128, out_dim: int = 32, enhanced: bool = False):
        super().__init__()
        self.n_mels = n_mels
        self.enhanced = enhanced
        # Base: 4 band energies + centroid + bandwidth + temporal centroid + 4 attack slopes = 11
        # Enhanced adds: spectral flatness + low/high ratio + temporal decay = 14
        raw_dim = 14 if enhanced else 11
        self.proj = nn.Sequential(
            nn.Linear(raw_dim * 2, out_dim),  # *2 for fine + coarse
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

        # Band boundaries (in mel bin indices)
        # Low: bass (0-31), Low-mid: body (32-63), Mid: ring (64-95), High: treble (96-127)
        self.bands = [(0, 32), (32, 64), (64, 96), (96, 128)]

    def _extract(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract spectral features from (B, 1, n_mels, T) log-mel."""
        x = mel.squeeze(1)  # (B, n_mels, T)
        # Convert from log to linear for energy computations
        energy = torch.exp(x)  # (B, n_mels, T)

        B, F, T = energy.shape
        features = []

        # Time-averaged spectrum
        spec = energy.mean(dim=-1)  # (B, F)
        spec_sum = spec.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Spectral centroid (weighted mean frequency bin)
        bins = torch.arange(F, device=mel.device, dtype=mel.dtype).unsqueeze(0) / F  # (1, F)
        centroid = (spec * bins).sum(dim=-1) / spec_sum.squeeze(-1)  # (B,)
        features.append(centroid.unsqueeze(-1))

        # Spectral bandwidth (weighted std of frequency bin)
        bandwidth = ((spec * (bins - centroid.unsqueeze(-1)) ** 2).sum(dim=-1)
                     / spec_sum.squeeze(-1)).sqrt()
        features.append(bandwidth.unsqueeze(-1))

        # Band energies (4 bands)
        for lo, hi in self.bands:
            band_e = spec[:, lo:hi].mean(dim=-1)  # (B,)
            features.append(band_e.unsqueeze(-1))

        # Temporal centroid (where in time does energy peak?)
        time_energy = energy.sum(dim=1)  # (B, T)
        time_sum = time_energy.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        time_bins = torch.arange(T, device=mel.device, dtype=mel.dtype).unsqueeze(0) / max(T, 1)
        temporal_centroid = (time_energy * time_bins).sum(dim=-1) / time_sum.squeeze(-1)
        features.append(temporal_centroid.unsqueeze(-1))

        # Attack slope per band (energy at frame 2 - energy at frame 0, for onset sharpness)
        for lo, hi in self.bands:
            if T >= 3:
                early = energy[:, lo:hi, :2].mean(dim=(1, 2))
                mid = energy[:, lo:hi, 2:4].mean(dim=(1, 2))
                slope = mid - early
            else:
                slope = torch.zeros(B, device=mel.device, dtype=mel.dtype)
            features.append(slope.unsqueeze(-1))

        # === Enhanced features (V4+) ===
        if self.enhanced:
            # Spectral flatness: geometric_mean / arithmetic_mean of linear spectrum
            # A low flatness → tonal (toms), high flatness → noise-like (snare, kick)
            log_spec = x.mean(dim=-1)  # (B, F) — log-mel, time-averaged
            log_mean = log_spec.mean(dim=-1)  # (B,) geometric mean in log domain
            arith_mean = spec.mean(dim=-1).clamp(min=1e-8)  # (B,) arithmetic mean of linear
            flatness = torch.exp(log_mean) / arith_mean
            features.append(flatness.unsqueeze(-1))

            # Low/high energy ratio: low band energy / high band energy
            # Kick has very high ratio, toms moderate, cymbals low
            low_e = spec[:, :64].mean(dim=-1).clamp(min=1e-8)
            high_e = spec[:, 64:].mean(dim=-1).clamp(min=1e-8)
            ratio = torch.log(low_e / high_e)  # log ratio for stability
            features.append(ratio.unsqueeze(-1))

            # Temporal decay rate: energy in second half / first half
            # Toms sustain longer → higher ratio; kick/snare decay fast → lower ratio
            half = max(T // 2, 1)
            first_half_e = time_energy[:, :half].mean(dim=-1).clamp(min=1e-8)
            second_half_e = time_energy[:, half:].mean(dim=-1).clamp(min=1e-8)
            decay = second_half_e / first_half_e
            features.append(decay.unsqueeze(-1))

        return torch.cat(features, dim=-1)  # (B, 11 or 14)

    def forward(self, mel_fine: torch.Tensor, mel_coarse: torch.Tensor) -> torch.Tensor:
        fine_feat = self._extract(mel_fine)    # (B, 11)
        coarse_feat = self._extract(mel_coarse)  # (B, 11)
        combined = torch.cat([fine_feat, coarse_feat], dim=-1)  # (B, 22)
        return self.proj(combined)  # (B, out_dim)


class OnsetContextEncoder(nn.Module):
    """Encode the pattern of surrounding onsets as auxiliary features."""

    def __init__(self, num_classes: int = 8, context_size: int = 4, hidden: int = 64):
        super().__init__()
        input_dim = 2 * context_size * num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out_dim = hidden

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class CrashFluxEncoder(nn.Module):
    """Encode crash-band spectral flux time series into a compact embedding.

    Input: (B, 2, T) where:
      - Channel 0: Crash-band energy profile (mean of mel bins 80-127)
      - Channel 1: Crash-band onset flux (positive half-wave rectified diff)

    A true crash onset produces a large spike in channel 1 at the onset
    position (~frame 17). Crash sustain, ride shimmer, and Demucs bleed
    show flat/zero flux. This distinction is invisible to the mel CNN
    which compresses temporal dynamics via global average pooling.
    """

    def __init__(self, in_channels: int = 2, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, 32, 1)
            nn.Flatten(),             # (B, 32)
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, 2, T) → Output: (B, out_dim)."""
        return self.net(x)


class OnsetClassifier(nn.Module):
    """
    Dual-resolution onset classifier with SE attention, spectral features,
    and deeper classification head.

    V4 additions: HPSS preprocessing (harmonic/percussive channels),
    enhanced spectral features (flatness, energy ratio, temporal decay).
    """

    # Class index mapping for dual-head architecture
    COMMON_INDICES = [0, 1, 2, 4, 6]   # Kick, Snare, HiHat, Ride, Crash
    TOM_INDICES = [3, 5, 7]             # HighTom, LowTom, FloorTom

    def __init__(
        self,
        num_classes: int = 8,
        branch_channels: list[int] = [1, 32, 64, 128, 256],
        context_size: int = 4,
        context_hidden: int = 64,
        classifier_hidden: int = 512,
        spectral_dim: int = 32,
        dropout: float = 0.3,
        use_freq_attn: bool = True,
        use_hpss: bool = False,
        enhanced_spectral: bool = False,
        use_contrastive: bool = False,
        use_aux_head: bool = False,
        projection_dim: int = 128,
        use_dual_head: bool = False,
        tom_head_hidden: int = 256,
        use_lowfreq_branch: bool = False,
        use_lowfreq_spectral: bool = False,
        context_classes: int | None = None,
        use_crash_flux: bool = False,
        crash_flux_dim: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_hpss = use_hpss
        self.use_dual_head = use_dual_head
        self.use_lowfreq_branch = use_lowfreq_branch
        self.use_lowfreq_spectral = use_lowfreq_spectral
        self.use_crash_flux = use_crash_flux

        # Optional HPSS layer: splits mel into 3 channels [original, harmonic, percussive]
        self.hpss = HPSSLayer() if use_hpss else None
        in_ch = 3 if use_hpss else None  # None = use channels[0] as-is

        # Dual-resolution spectral branches with SE
        self.fine_branch = SpectralBranch(branch_channels, use_freq_attn=use_freq_attn, in_channels=in_ch)
        self.coarse_branch = SpectralBranch(branch_channels, use_freq_attn=use_freq_attn, in_channels=in_ch)

        # Optional low-frequency focused branch (30-2000 Hz, 128 mel bins)
        # Provides ~6x more frequency resolution in the 40-500 Hz range where
        # Kick, Snare, HighTom, LowTom, FloorTom fundamentals overlap
        self.lowfreq_branch = None
        if use_lowfreq_branch:
            self.lowfreq_branch = SpectralBranch(
                branch_channels, use_freq_attn=use_freq_attn, in_channels=in_ch)

        # Explicit spectral features
        self.spectral_extractor = SpectralFeatureExtractor(
            out_dim=spectral_dim, enhanced=enhanced_spectral)

        # Context encoder
        self.context_encoder = OnsetContextEncoder(
            num_classes=context_classes if context_classes is not None else num_classes,
            context_size=context_size,
            hidden=context_hidden,
        )

        # V16: Dedicated spectral features from 30-2000 Hz mel_lowfreq.
        # The CNN lowfreq_branch learns generic features, but compresses away
        # explicit centroid/bandwidth — the key discriminant for tom vs cymbal.
        # This path extracts raw spectral stats and projects them separately.
        self.lowfreq_spectral_proj = None
        if use_lowfreq_spectral:
            raw_lf_dim = 14 if enhanced_spectral else 11
            self.lowfreq_spectral_proj = nn.Sequential(
                nn.Linear(raw_lf_dim, spectral_dim),
                nn.ReLU(),
                nn.Linear(spectral_dim, spectral_dim),
                nn.ReLU(),
            )

        # V19: Crash-band spectral flux encoder.
        # Encodes the temporal onset-novelty profile of 5-22 kHz energy.
        # True crash onsets show a sharp positive spike; sustain/bleed is flat.
        self.crash_flux_encoder = None
        if use_crash_flux:
            self.crash_flux_encoder = CrashFluxEncoder(
                in_channels=2, out_dim=crash_flux_dim)

        # Fusion dimension
        fused_dim = (
            self.fine_branch.out_channels
            + self.coarse_branch.out_channels
            + self.spectral_extractor.out_dim
            + self.context_encoder.out_dim
        )
        if use_lowfreq_branch:
            fused_dim += self.lowfreq_branch.out_channels
        if use_lowfreq_spectral:
            fused_dim += spectral_dim
        if use_crash_flux:
            fused_dim += crash_flux_dim

        if use_dual_head:
            # V8: Separate classification heads for common vs tom classes.
            # Prevents tom gradients from warping non-tom pathways and vice versa.
            # Common head: larger capacity for 5 well-represented classes
            self.common_classifier = nn.Sequential(
                nn.Linear(fused_dim, classifier_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, classifier_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden // 2, len(self.COMMON_INDICES)),
            )
            # Tom head: dedicated pathway for rare tom classes
            self.tom_classifier = nn.Sequential(
                nn.Linear(fused_dim, tom_head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(tom_head_hidden, tom_head_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(tom_head_hidden // 2, len(self.TOM_INDICES)),
            )
            # Dummy classifier attribute for backward compat with freeze_backbone checks
            self.classifier = nn.Identity()
        else:
            # Standard single-head classifier
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, classifier_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, classifier_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden, classifier_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(classifier_hidden // 2, num_classes),
            )

        # V5: Supervised contrastive projection head
        self.projection_head = None
        if use_contrastive:
            self.projection_head = ProjectionHead(
                fused_dim, hidden_dim=256, out_dim=projection_dim
            )

        # V5: Auxiliary "is tom?" classification head
        # Provides explicit gradient signal for tom vs non-tom discrimination
        self.aux_head = None
        if use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Linear(fused_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        self._count_params()

    def _count_params(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"OnsetClassifier: {total / 1e6:.2f}M params ({trainable / 1e6:.2f}M trainable)")

    def forward(
        self,
        mel_fine: torch.Tensor,
        mel_coarse: torch.Tensor,
        context: torch.Tensor,
        return_features: bool = False,
        mel_lowfreq: torch.Tensor | None = None,
        crash_flux: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Optional HPSS: (B, 1, F, T) → (B, 3, F, T)
        if self.hpss is not None:
            mel_fine_hpss = self.hpss(mel_fine)
            mel_coarse_hpss = self.hpss(mel_coarse)
        else:
            mel_fine_hpss = mel_fine
            mel_coarse_hpss = mel_coarse

        fine_feat = self.fine_branch(mel_fine_hpss)       # (B, 256)
        coarse_feat = self.coarse_branch(mel_coarse_hpss) # (B, 256)
        # Spectral features always computed from original 1-channel mel
        spectral_feat = self.spectral_extractor(mel_fine, mel_coarse)  # (B, spectral_dim)
        ctx_feat = self.context_encoder(context)           # (B, 64)

        parts = [fine_feat, coarse_feat, spectral_feat, ctx_feat]

        # Optional low-frequency focused branch
        if self.lowfreq_branch is not None and mel_lowfreq is not None:
            if self.hpss is not None:
                mel_lowfreq_hpss = self.hpss(mel_lowfreq)
            else:
                mel_lowfreq_hpss = mel_lowfreq
            lowfreq_feat = self.lowfreq_branch(mel_lowfreq_hpss)  # (B, 256)
            parts.append(lowfreq_feat)

        # V16: Explicit spectral centroid/bandwidth from 30-2000 Hz mel
        if self.lowfreq_spectral_proj is not None and mel_lowfreq is not None:
            lf_raw = self.spectral_extractor._extract(mel_lowfreq)  # (B, 11 or 14)
            lf_spectral_feat = self.lowfreq_spectral_proj(lf_raw)   # (B, spectral_dim)
            parts.append(lf_spectral_feat)

        # V19: Crash-band spectral flux
        if self.crash_flux_encoder is not None:
            if crash_flux is not None:
                crash_flux_feat = self.crash_flux_encoder(crash_flux)  # (B, crash_flux_dim)
            else:
                # Zero-pad so fused_dim stays consistent when flux unavailable
                crash_flux_feat = torch.zeros(
                    mel_fine.shape[0], self.crash_flux_encoder.out_dim,
                    device=mel_fine.device, dtype=mel_fine.dtype)
            parts.append(crash_flux_feat)

        fused = torch.cat(parts, dim=1)

        if self.use_dual_head:
            # V8: Separate heads for common and tom classes
            common_logits = self.common_classifier(fused)  # (B, 5)
            tom_logits = self.tom_classifier(fused)        # (B, 3)
            # Reassemble into standard 8-class ordering
            B = fused.shape[0]
            logits = torch.zeros(B, self.num_classes, device=fused.device, dtype=fused.dtype)
            logits[:, self.COMMON_INDICES] = common_logits
            logits[:, self.TOM_INDICES] = tom_logits
        else:
            logits = self.classifier(fused)

        if not return_features:
            return logits

        extras = {}
        if self.projection_head is not None:
            extras['projection'] = self.projection_head(fused)
        if self.aux_head is not None:
            extras['aux_logits'] = self.aux_head(fused).squeeze(-1)
        return logits, extras
