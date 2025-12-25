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


# try:
#     # Use torch.compile on modern PyTorch versions
#     compile_fn = torch.compile
# except:
#     # Fallback identity
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
        smoothing_sigma: float = 5.0,  # Increased sigma for smoother/musical morphs
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
        n_samples = audio.shape[-1]

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

        img_resized = F.interpolate(
            img,
            size=(self.full_height, self.full_width),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize and Scale Image
        log_min = self.audio_log.min()
        log_max = self.audio_log.max()

        img_01 = img_resized.squeeze(0)
        img_01 = (img_01 - img_01.min()) / (img_01.max() - img_01.min() + 1e-8)

        self.image_log = img_01 * (log_max - log_min) + log_min
        self.image_mag_ref = img_01

        # Compile the heavy computation part: Mask -> Spectrogram Mixing
        self.compute_spectrogram = compile_fn(self._compute_spectrogram_uncompiled)

    def _compute_spectrogram_uncompiled(self, mask, img_log, aud_log):
        # 2. Blend in LOG DOMAIN
        mixed_log = mask * img_log + (1 - mask) * aud_log
        mixed_mag = torch.exp(mixed_log)
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
        mask = self.mask_processor(params, self.full_height, self.full_width)

        # Expand sources
        img_log = self.image_log.expand(batch_size, -1, -1)
        aud_log = self.audio_log.expand(batch_size, -1, -1)
        phase = self.audio_phase.expand(batch_size, -1, -1)

        # Compute Spectrogram (JIT Optimized)
        mixed_mag = self.compute_spectrogram(mask, img_log, aud_log)

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
        mask = F.interpolate(
            grid, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

        # Apply Gaussian Smoothing using functional conv2d
        # weight must be (Out, In, kH, kW)
        mask = F.conv2d(
            mask,
            self.blur_kernel,  # Already on device
            padding=self.kernel_padding,
        )
        return mask.squeeze(1)

    def forward(self, params, target_h, target_w):
        return self.process(params, target_h, target_w)
