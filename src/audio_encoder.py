"""Audio Encoder: Mask-based blending of Image and Sound Spectrograms.

Strategy:
- Load Target Audio -> Compute STFT (Magnitude + Phase)
- Load Target Image -> Resize to match STFT Magnitude dimensions
- Optimization Variable: A low-resolution "Mask" grid
- Synthesis:
    1. Upsample Mask to full spectrogram size
    2. Blended Magnitude = Mask * Image + (1 - Mask) * Audio
    3. Reconstruct Audio = ISTFT(Blended Magnitude, Target Audio Phase)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from . import config


class MaskEncoder(nn.Module):
    """Encodes parameters into audio via mask-based spectrogram blending."""

    def __init__(
        self,
        target_image: torch.Tensor,
        target_audio_path: str,
        grid_height: int = 32,
        grid_width: int = 64,
        device: str = config.DEVICE,
    ):
        super().__init__()
        self.device = device
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.n_params = grid_height * grid_width

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

        # 2. Process Target Image to match STFT dimensions
        # Image is typically linear RGB, we need a single channel magnitude map
        # input is (1, H, W)

        # Resize image to match spectrogram dimensions (Linear Frequency)
        # Note: Image Y=0 is top, Spectrogram Y=0 is 0Hz (bottom)
        # We generally want image top = high freq.
        # But verify alignment relies on visualize.py. Assuming standard orientation.

        img = target_image.to(device)
        # Ensure 4D for interpolate: (batch, channels, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        img_resized = F.interpolate(
            img, size=(full_height, full_width), mode="bilinear", align_corners=False
        )

        # Normalize image magnitude to match audio dynamic range roughly?
        # Or normalize both to [0,1] for checking?
        # For RECONSTRUCTION, we need the image to have valid STFT magnitude range.
        # Audio mag range: typically 0 to 100+. Image is 0-1.
        # Strategy: Scale image to match audio's max magnitude.

        ref_max = self.audio_mag.max()
        img_normalized = img_resized.squeeze(0)  # (1, F, T)
        img_normalized = (img_normalized - img_normalized.min()) / (
            img_normalized.max() - img_normalized.min() + 1e-8
        )

        # Scale image to be loud enough
        self.image_mag = img_normalized * ref_max

    def forward(self, params: torch.Tensor):
        """
        Args:
            params: (batch, n_params) in [0, 1] - The Mask

        Returns:
            audio: (batch, samples)
            mixed_spec_mag: (batch, F, T)
        """
        batch_size = params.shape[0]

        # 1. Reshape and Upsample Mask
        grid = params.view(batch_size, 1, self.grid_height, self.grid_width)
        mask = F.interpolate(
            grid,
            size=(self.full_height, self.full_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        # Mask: 1 = Image, 0 = Audio

        # 2. Blend Magnitudes
        # Expand sources to batch
        img = self.image_mag.expand(batch_size, -1, -1)
        aud = self.audio_mag.expand(batch_size, -1, -1)

        mixed_mag = mask * img + (1 - mask) * aud

        # 3. Reconstruct Audio using Target Phase
        # Reconstruct complex STFT: mag * e^(j*phase)
        phase = self.audio_phase.expand(batch_size, -1, -1)
        complex_stft = torch.polar(mixed_mag, phase)

        # Inverse STFT
        window = torch.hann_window(config.N_FFT).to(self.device).expand(batch_size, -1)
        # ISTFT requires careful batch handling. torch.istft defines batch as (..., F, T)

        # Reshape for istft: it expects (batch, freq, time) complex
        # Note: torch.istft handles batch processing natively in newer versions

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
