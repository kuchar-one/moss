"""Audio Encoder: Mask-based blending of Image and Sound Spectrograms.

Strategy:
- Load Target Audio -> Compute STFT (Magnitude + Phase)
- Load Target Image -> Resize to match STFT Magnitude dimensions
- Optimization Variable: A low-resolution "Mask" grid
- Synthesis:
    1. Upsample Mask to full spectrogram size
    2. Apply Gaussian Blur to Mask (Forcing Smoothness/Musicality)
    3. Blended Magnitude = Mask * Image + (1 - Mask) * Audio
    4. Reconstruct Audio = ISTFT(Blended Magnitude, Target Audio Phase)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math

from . import config


try:
    # Use torch.compile on modern PyTorch versions
    compile_fn = torch.compile
except Exception:
    # Fallback identity
    def compile_fn(x):
        return x


class MaskEncoder(nn.Module):
    """Encodes parameters into audio via mask-based spectrogram blending."""

    def __init__(
        self,
        target_image: torch.Tensor,
        target_audio_path: str,
        grid_height: int = 128,
        grid_width: int = 256,
        smoothing_sigma: float = 1.0,  # Reduced for low-res grid processing
        device: str = config.DEVICE,
    ):
        super().__init__()
        self.device = device
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.n_params = grid_height * grid_width
        self.smoothing_sigma = smoothing_sigma

        # 1. Load and Process Target Audio
        audio, sr = torchaudio.load(target_audio_path)
        if sr != config.SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, config.SAMPLE_RATE)

        # Mix to mono
        audio = audio.mean(dim=0, keepdim=True)

        # Trim to ensure valid STFT length (multiple of hop_length)
        # We NO LONGER crop to fixed duration. We keep full length.
        # n_samples = audio.shape[-1]

        # Determine valid length for centered STFT
        # Or just pad? Torchaudio/Torch stft handles this.
        # But to be safe for ISTFT invertibility, let's fix length to match N_FFT/Hop logic if needed.
        # Actually torch.stft center=True (default) handles padding.

        self.mask_processor = MaskProcessor(
            grid_height, grid_width, smoothing_sigma, device
        )

        self.register_buffer("target_audio_waveform", audio)

        # Compute STFT
        window = torch.hann_window(config.N_FFT).to(device)
        stft = torch.stft(
            audio.to(device),
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH,
            window=window,
            return_complex=True,
        )

        # Extract Magnitude and Phase
        self.audio_mag = stft.abs() + 1e-8
        self.audio_phase = stft.angle()

        self.full_height = self.audio_mag.shape[1]
        self.full_width = self.audio_mag.shape[2]

        # Pre-compute log-magnitude for audio (dB-like)
        self.audio_log = torch.log(self.audio_mag)

        # 2. Process Target Image to match STFT dimensions
        img = target_image.to(device)
        if img.dim() == 3:
            img = img.unsqueeze(0)  # (1, C, H, W)

        # Normalize and Scale Image

        # FREQUENCY MAPPING: Map image to lower frequencies to sound better.
        # Original: Image spans 0 - 11kHz.
        # New: Image spans 0 - 5.5kHz (Bottom Half).
        # We resize image to (H//2, W) and pad the top.

        # FREQUENCY MAPPING: Map Image to WHOLE Range (User Request).
        # Previously we cropped to 5.5kHz to avoid screech, but user wants "Whole Range".
        # We rely on GAMMA correction to reduce the High-Freq screech volume.

        target_h_visual = self.full_height
        img_full_freq = F.interpolate(
            img,
            size=(target_h_visual, self.full_width),
            mode="bilinear",
            align_corners=False,
        )
        img_resized = img_full_freq

        # Audio stats (Log domain)
        # ROBUST MAX: Use 98th percentile to ignore transient clicks/outliers.
        # This prevents the entire image from being scaled extremely loud due to one peak.
        audio_log_max = torch.quantile(self.audio_log, 0.98)

        # DYNAMIC GAIN STAGING (Auto-Match)
        # Calculate optimal headroom to maximize visibility without clipping.

        # 1. Find Audio Boundaries
        # Max: Absolute peak (safety)
        audio_max_val = self.audio_log.max()
        # Floor: 1st percentile (noise floor/quietest content)
        # We want even the quietest parts to be visible.
        audio_floor_val = torch.quantile(self.audio_log, 0.01)

        # 2. Target Ceiling: Place Image White just below the absolute peak (e.g. -0.5 nat safety)
        # This ensures we use the full available dynamic range of the file.
        target_ceiling = audio_max_val - 0.5

        # 3. Calculate Headroom relative to Robust Max (q98)
        # headroom = q98 - ceiling
        headroom_nat = (audio_log_max - target_ceiling).item()

        # 4. ADAPTIVE DYNAMIC RANGE
        # Stretch range to cover [Target Ceiling -> Audio Floor].
        # Ensures quietest audio is mapped to bottom of range, not clipped.
        # Add 0.5 nat buffer so floor isn't pure black.
        dynamic_range_nat = (target_ceiling - audio_floor_val).item() + 0.5

        # Clamp range to be reasonable (e.g. at least 4.0, max 12.0)
        dynamic_range_nat = max(4.0, min(dynamic_range_nat, 12.0))

        print("Dynamic Gain Staging:")
        print(f"  > Audio Max: {audio_max_val:.2f}, Floor (q01): {audio_floor_val:.2f}")
        print(f"  > Target Ceiling: {target_ceiling:.2f}")
        print(f"  > Adaptive Dynamic Range: {dynamic_range_nat:.2f}")
        print(f"  > Calculated Headroom: {headroom_nat:.2f} (Negative means Boost)")

        audio_log_ceil = audio_log_max - headroom_nat
        # audio_log_floor depends on the newly calculated dynamic range
        audio_log_floor = audio_log_ceil - dynamic_range_nat

        # Clamp audio_log bottom for mixing stability
        self.audio_log = torch.clamp(self.audio_log, min=audio_log_floor)

        # Map Image 0->1 to [audio_log_floor, audio_log_ceil]
        img_01 = img_resized.squeeze(0)
        img_01 = (img_01 - img_01.min()) / (img_01.max() - img_01.min() + 1e-8)

        # GAMMA CORRECTION (1.8)
        # Reduced from 2.5 (too dark) to 1.8 (brighter midtones).
        img_01 = img_01.pow(1.8)

        self.image_log = img_01 * (audio_log_ceil - audio_log_floor) + audio_log_floor

        # PRE-COMPUTE LINEAR MAGNITUDES FOR MIXING
        # We mix in Linear Domain to preserve volume (Arithmetic Mean vs Geometric Mean)
        # Log-mixing (Geometric) kills volume if Image is black (0).
        self.image_mag = torch.exp(self.image_log)
        self.audio_mag_static = torch.exp(self.audio_log)  # Clamped version

        # ---------------------------------------------------------------------
        # PROXY OPTIMIZATION SETUP (The "10x" Speedup)
        # ---------------------------------------------------------------------
        # We optimize on a lower-resolution proxy (128 bins) to save 16x compute.
        # The control grid is 64x128, so 513 bins is overkill for loss calculation.
        self.proxy_height = 129  # 1/4 of 513
        self.proxy_width = self.full_width // 2

        # Create Proxy Tensors (Downsampled)
        self.image_mag_proxy = F.interpolate(
            self.image_mag.unsqueeze(0),
            size=(self.proxy_height, self.proxy_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.audio_mag_proxy = F.interpolate(
            self.audio_mag_static.unsqueeze(0),
            size=(self.proxy_height, self.proxy_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Expose PROXY as the default reference for Optimizers
        self.image_mag_ref = self.image_mag_proxy  # Used by Optimizers
        self.audio_mag = self.audio_mag_proxy  # Used by Optimizers (renaming ref)

        # Keep FULL res for final Export/Reconstruction
        self.image_mag_full = self.image_mag
        self.audio_mag_full = self.audio_mag_static

        # Compile the heavy computation part: Mask -> Spectrogram Mixing
        self.compute_spectrogram = compile_fn(self._compute_spectrogram_uncompiled)

    def _compute_spectrogram_uncompiled(self, mask, img_mag, aud_mag):
        # 2. Blend in LINEAR DOMAIN
        # Mixed = Mask * Img + (1 - Mask) * Aud
        mixed_mag = mask * img_mag + (1 - mask) * aud_mag
        return mixed_mag

    def _reconstruct_audio(self, mixed_mag, phase):
        # 3. Reconstruct Audio (No JIT, complex ops)
        complex_stft = torch.polar(mixed_mag, phase)

        audio_recon = torch.istft(
            complex_stft,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH,
            window=torch.hann_window(config.N_FFT, device=mixed_mag.device),
        )

        # Normalize
        max_val = audio_recon.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        audio_recon = audio_recon / max_val * 0.9
        return audio_recon

    def forward(self, params: torch.Tensor, return_wav: bool = True):
        batch_size = params.shape[0]

        # 1. Generate Mask (Upsample + Smooth)
        # If optimization mode (no wav), target PROXY size
        target_h = self.proxy_height if not return_wav else self.full_height
        target_w = self.proxy_width if not return_wav else self.full_width

        mask = self.mask_processor(params, target_h, target_w)

        # Expand sources (Use PROXY or FULL based on mode)
        if not return_wav:
            # Optimization Path (Fast)
            img_mag = self.image_mag_ref.expand(batch_size, -1, -1)
            aud_mag = self.audio_mag.expand(
                batch_size, -1, -1
            )  # This is audio_mag_proxy
        else:
            # Export Path (High Quality)
            img_mag = self.image_mag_full.expand(batch_size, -1, -1)
            aud_mag = self.audio_mag_full.expand(batch_size, -1, -1)

        phase = self.audio_phase.expand(batch_size, -1, -1) if return_wav else None

        # Compute Spectrogram (JIT Optimized)
        mixed_mag = self.compute_spectrogram(mask, img_mag, aud_mag)

        audio_recon = None
        if return_wav:
            audio_recon = self._reconstruct_audio(mixed_mag, phase)

        return audio_recon, mixed_mag


class MaskProcessor(nn.Module):
    """Helper module for Mask generation to separate convolution logic."""

    def __init__(self, h, w, sigma, device):
        super().__init__()
        self.h = h
        self.w = w
        self.device = device
        self._init_gaussian_kernel(sigma)

        self.process = compile_fn(self._process_uncompiled)

    def _init_gaussian_kernel(self, sigma):
        kernel_size = int(2 * math.ceil(2 * sigma) + 1)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.0
        variance = sigma**2.0
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        self.blur_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(
            self.device
        )
        self.kernel_padding = kernel_size // 2

    def _process_uncompiled(self, params, target_h, target_w):
        B = params.shape[0]
        grid = params.view(B, 1, self.h, self.w)

        # CPU OPTIMIZATION: Blur the Low-Res Grid instead of the High-Res Up-sampled Mask.
        # This saves millions of FLOPs.
        grid_blurred = F.conv2d(grid, self.blur_kernel, padding=self.kernel_padding)

        mask = F.interpolate(
            grid_blurred,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        return mask.squeeze(1)

    def forward(self, params, target_h, target_w):
        return self.process(params, target_h, target_w)
