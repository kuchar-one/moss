"""Audio Encoder: Optimizes spectrogram to match image AND sound.

Core approach:
1. Decision variables: Low-res spectrogram grid (32x64)
2. Upsample to full resolution (128x~2500)
3. Griffin-Lim to convert spectrogram → audio
4. Compare against target image (SSIM) and target audio (spectral loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from . import config


class SpectrogramEncoder(nn.Module):
    """Encodes parameters into audio via spectrogram manipulation."""

    def __init__(
        self,
        target_image: torch.Tensor,
        target_audio: torch.Tensor = None,
        grid_height: int = 32,
        grid_width: int = 64,
    ):
        """
        Args:
            target_image: Target image as spectrogram (1, H, W) normalized to [0,1]
            target_audio: Target audio waveform (1, samples) or None
            grid_height: Height of low-res control grid
            grid_width: Width of low-res control grid
        """
        super().__init__()

        self.grid_height = grid_height
        self.grid_width = grid_width
        self.n_params = grid_height * grid_width

        # Store target image (normalized)
        target = target_image.squeeze()
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        self.register_buffer("target_image", target)

        self.full_height = target.shape[0]  # 128 mel bins
        self.full_width = target.shape[1]  # ~2500 time frames

        # Store target audio if provided
        if target_audio is not None:
            self.register_buffer("target_audio", target_audio.squeeze())
            self.has_target_audio = True
        else:
            self.has_target_audio = False

        # Mel spectrogram for comparing audio
        self.mel_spec = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX,
        )

    def forward(self, params: torch.Tensor):
        """Generate audio from parameters.

        Args:
            params: (batch, n_params) values in [0, 1]

        Returns:
            audio: (batch, samples) waveform
            spec: (batch, H, W) generated spectrogram (for visual comparison)
        """
        device = params.device
        batch_size = params.shape[0]

        # Reshape params into low-res grid
        grid = params.view(batch_size, 1, self.grid_height, self.grid_width)

        # Upsample to full spectrogram size
        full_spec = F.interpolate(
            grid,
            size=(self.full_height, self.full_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (batch, H, W)

        # Convert spectrogram to audio via Griffin-Lim
        audio = self._spec_to_audio(full_spec, device)

        return audio, full_spec

    def _spec_to_audio(self, spec: torch.Tensor, device) -> torch.Tensor:
        """Convert spectrogram to audio using Griffin-Lim."""
        batch_size = spec.shape[0]

        audios = []
        for i in range(batch_size):
            # Scale spectrogram for audibility
            spec_power = spec[i] ** 2 * 100

            # Upsample from mel to linear frequency bins
            linear_spec = F.interpolate(
                spec_power.unsqueeze(0).unsqueeze(0),
                size=(config.N_FFT // 2 + 1, spec_power.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            # Griffin-Lim phase reconstruction
            griffin_lim = T.GriffinLim(
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                power=2.0,
                n_iter=32,
            )
            audio = griffin_lim(linear_spec.cpu())

            # Pad/trim to target length
            target_len = config.NUM_SAMPLES
            if audio.shape[-1] < target_len:
                audio = F.pad(audio, (0, target_len - audio.shape[-1]))
            else:
                audio = audio[:target_len]

            audios.append(audio.to(device))

        audio = torch.stack(audios)

        # Normalize
        max_val = audio.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        audio = audio / max_val * 0.9

        return audio

    def get_audio_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram of audio for comparison."""
        # Create mel_spec on CPU to avoid device mismatch
        mel_spec = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.F_MIN,
            f_max=config.F_MAX,
        )

        spec = mel_spec(audio.cpu())

        # Normalize
        spec = spec / (
            spec.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8
        )

        return spec.to(audio.device)
