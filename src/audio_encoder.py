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


class MaskEncoder(nn.Module):
    """Encodes parameters into audio via mask-based spectrogram blending."""

    def __init__(
        self,
        target_image: torch.Tensor,
        target_audio_path: str,
        grid_height: int = 128,
        grid_width: int = 256,
        smoothing_sigma: float = 1.0,  # Controls smoothness (Standard deviation of Gaussian)
        device: str = config.DEVICE,
    ):
        super().__init__()
        self.device = device
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.n_params = grid_height * grid_width
        self.smoothing_sigma = smoothing_sigma

        # 1. Load and Process Target Audio (Source of Truth for Shape/Phase)
        audio, sr = torchaudio.load(target_audio_path)
        if sr != config.SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, config.SAMPLE_RATE)

        # Mix to mono
        audio = audio.mean(dim=0, keepdim=True)

        # Truncate/Pad to fixed duration (simplifies STFT consistency)
        target_len = config.NUM_SAMPLES
        if audio.shape[-1] > target_len:
            audio = audio[..., :target_len]
        else:
            audio = F.pad(audio, (0, target_len - audio.shape[-1]))

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
        # stft shape: (1, F, T) complex
        self.audio_mag = stft.abs() + 1e-8
        self.audio_phase = stft.angle()

        full_height, full_width = self.audio_mag.shape[1], self.audio_mag.shape[2]
        self.full_height = full_height
        self.full_width = full_width

        # Pre-compute log-magnitude for audio (dB-like)
        self.audio_log = torch.log(self.audio_mag)

        # 2. Process Target Image to match STFT dimensions
        img = target_image.to(device)
        # Ensure 4D for interpolate: (batch, channels, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        img_resized = F.interpolate(
            img, size=(full_height, full_width), mode="bilinear", align_corners=False
        )

        # Normalize and Scale Image
        # To blend faithfully in Log Domain, image should map to typical Log-Mag range of audio
        # Audio Log Mag range: roughly -10 to +5?

        log_min = self.audio_log.min()
        log_max = self.audio_log.max()

        img_01 = img_resized.squeeze(0)
        img_01 = (img_01 - img_01.min()) / (img_01.max() - img_01.min() + 1e-8)

        # Map Image 0-1 to Audio's Log Range
        # Ideally we want valid parts to be "audible" and background "silent"
        # Let's map 0 -> log_min (silence) and 1 -> log_max (loud)
        self.image_log = img_01 * (log_max - log_min) + log_min

        # Store linear image mag for visual comparison (SSIM works better on linear 0-1)
        self.image_mag_ref = img_01

        # Initialize Gaussian Blur Kernel
        self._init_gaussian_kernel(sigma=smoothing_sigma)

    def _init_gaussian_kernel(self, sigma):
        # Create a 2D Gaussian kernel
        kernel_size = int(2 * math.ceil(2 * sigma) + 1)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.0
        variance = sigma**2.0

        # Calculate the 2-D gaussian kernel
        gaussian_kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
            -torch.sum((xy_grid - mean) ** 2.0, dim=-1) / (2 * variance)
        )

        # Normalize
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape for conv2d: (out_channels, in_channels, kH, kW)
        # We apply to 1 channel (the mask)
        self.blur_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(
            self.device
        )
        self.kernel_padding = kernel_size // 2

    def forward(self, params: torch.Tensor):
        """
        Args:
            params: (batch, n_params) in [0, 1] - The Mask

        Returns:
            audio: (batch, samples)
            mixed_mag: (batch, F, T) - Linear Magnitude
        """
        batch_size = params.shape[0]

        # 1. Reshape and Upsample Mask
        grid = params.view(batch_size, 1, self.grid_height, self.grid_width)
        mask = F.interpolate(
            grid,
            size=(self.full_height, self.full_width),
            mode="bilinear",
            align_corners=False,
        )

        # 1.5 Apply Gaussian Smoothing to the Mask
        # This prevents "random pixel replacement" and forces "blobby/natural" transitions
        # Input mask is (B, 1, F, T)
        if self.smoothing_sigma > 0:
            mask = F.conv2d(
                mask,
                self.blur_kernel.expand(1, 1, -1, -1),  # Kernel already on device
                padding=self.kernel_padding,
            )

        mask = mask.squeeze(1)  # (B, F, T)

        # 2. Blend in LOG DOMAIN (dB mixing)
        # Expand sources
        img_log = self.image_log.expand(batch_size, -1, -1)
        aud_log = self.audio_log.expand(batch_size, -1, -1)

        mixed_log = mask * img_log + (1 - mask) * aud_log
        mixed_mag = torch.exp(mixed_log)

        # 3. Reconstruct Audio using Target Phase
        phase = self.audio_phase.expand(batch_size, -1, -1)
        complex_stft = torch.polar(mixed_mag, phase)

        # Inverse STFT
        audio_recon = torch.istft(
            complex_stft,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH,
            window=torch.hann_window(config.N_FFT).to(self.device),
            length=config.NUM_SAMPLES,
        )

        # Normalize output audio
        max_val = audio_recon.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        audio_recon = audio_recon / max_val * 0.9

        return audio_recon, mixed_mag
