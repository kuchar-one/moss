"""Parametric spectrogram synthesizer with low parameter count.

Uses smooth parametric curves to control frequency and time patterns,
keeping parameter count low enough for GA to effectively explore.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import math

from . import config


# Low-parameter representation for GA tractability
SYNTH_PARAMS = {
    # Frequency profile (controls vertical brightness pattern)
    "freq_center": (0.0, 1.0),  # Where peak energy is (0=low, 1=high)
    "freq_width": (0.1, 0.9),  # How spread out in frequency
    "freq_skew": (-1.0, 1.0),  # Asymmetry (neg=more bass, pos=more treble)
    # Time profile (controls horizontal brightness pattern)
    "time_peak": (0.0, 1.0),  # Where time peak is
    "time_width": (0.1, 0.9),  # How spread in time
    "time_shape": (0.0, 1.0),  # 0=gaussian, 1=flat-top
    # Texture controls
    "texture_freq": (0.0, 5.0),  # Frequency of oscillation in pattern
    "texture_amp": (0.0, 0.5),  # Amplitude of texture
    "noise_amount": (0.0, 0.5),  # Random variation
    # Brightness levels for 4 frequency bands
    "band1": (0.0, 1.0),  # Sub-bass
    "band2": (0.0, 1.0),  # Bass-mid
    "band3": (0.0, 1.0),  # Mid-high
    "band4": (0.0, 1.0),  # High
    # Overall
    "brightness": (0.2, 1.0),  # Overall level
    "contrast": (0.5, 2.0),  # Dynamic range
}


class ParametricSynth(nn.Module):
    """Synthesizer with smooth parametric spectrogram control."""

    def __init__(self, batch_size: int = 1, sample_rate: int = None):
        super().__init__()

        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.num_samples = config.NUM_SAMPLES
        self.duration = config.DURATION
        self.batch_size = batch_size

        self.param_names = list(SYNTH_PARAMS.keys())
        self.n_params = len(self.param_names)

        self.n_mels = config.IMG_HEIGHT
        self.n_frames = config.IMG_WIDTH

    def _denormalize_params(self, params: torch.Tensor) -> dict:
        param_dict = {}
        for i, name in enumerate(self.param_names):
            low, high = SYNTH_PARAMS[name]
            param_dict[name] = params[:, i] * (high - low) + low
        return param_dict

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        device = params.device
        batch_size = params.shape[0]

        p = self._denormalize_params(params)

        # Generate parametric spectrogram
        spec = self._generate_spectrogram(p, device, batch_size)

        # Convert to audio using Griffin-Lim
        audios = []
        for i in range(batch_size):
            spec_single = spec[i]  # (n_mels, n_frames)
            spec_power = spec_single**2 * 100  # Scale for audibility

            # Upsample mel to linear spectrogram
            linear_spec = torch.nn.functional.interpolate(
                spec_power.unsqueeze(0).unsqueeze(0),
                size=(config.N_FFT // 2 + 1, spec_power.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            # Griffin-Lim on CPU (create fresh to avoid device issues)
            griffin_lim = T.GriffinLim(
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                power=2.0,
                n_iter=16,
            )
            audio = griffin_lim(linear_spec.cpu())

            # Ensure correct length
            if audio.shape[-1] < self.num_samples:
                audio = torch.nn.functional.pad(
                    audio, (0, self.num_samples - audio.shape[-1])
                )
            else:
                audio = audio[: self.num_samples]

            audios.append(audio.to(device))

        audio = torch.stack(audios)

        # Normalize
        max_val = audio.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        audio = audio / max_val * 0.9

        return audio

    def _generate_spectrogram(self, p, device, batch_size):
        """Generate spectrogram from parametric description."""
        # Create coordinate grids
        freq_coords = torch.linspace(0, 1, self.n_mels, device=device)
        time_coords = torch.linspace(0, 1, self.n_frames, device=device)
        freq_grid, time_grid = torch.meshgrid(freq_coords, time_coords, indexing="ij")

        # Expand for batch
        freq_grid = freq_grid.unsqueeze(0).expand(batch_size, -1, -1)
        time_grid = time_grid.unsqueeze(0).expand(batch_size, -1, -1)

        # Frequency profile (gaussian with skew)
        freq_center = p["freq_center"].view(batch_size, 1, 1)
        freq_width = p["freq_width"].view(batch_size, 1, 1)
        freq_skew = p["freq_skew"].view(batch_size, 1, 1)

        freq_diff = freq_grid - freq_center
        # Apply skew: different width above and below center
        freq_sigma = freq_width * (1 + freq_skew * torch.sign(freq_diff) * 0.5)
        freq_profile = torch.exp(-0.5 * (freq_diff / freq_sigma.clamp(min=0.05)) ** 2)

        # Time profile
        time_peak = p["time_peak"].view(batch_size, 1, 1)
        time_width = p["time_width"].view(batch_size, 1, 1)
        time_shape = p["time_shape"].view(batch_size, 1, 1)

        time_diff = torch.abs(time_grid - time_peak)
        time_gaussian = torch.exp(-0.5 * (time_diff / time_width.clamp(min=0.05)) ** 2)
        time_flat = (time_diff < time_width).float()
        time_profile = time_gaussian * (1 - time_shape) + time_flat * time_shape

        # Base pattern
        spec = freq_profile * time_profile

        # Add 4-band control
        band_bounds = [0.0, 0.25, 0.5, 0.75, 1.0]
        band_levels = [p["band1"], p["band2"], p["band3"], p["band4"]]

        for i, level in enumerate(band_levels):
            low, high = band_bounds[i], band_bounds[i + 1]
            mask = ((freq_grid >= low) & (freq_grid < high)).float()
            spec = spec * (1 - mask) + spec * mask * level.view(batch_size, 1, 1)

        # Add texture
        texture_freq = p["texture_freq"].view(batch_size, 1, 1)
        texture_amp = p["texture_amp"].view(batch_size, 1, 1)

        texture = texture_amp * torch.sin(2 * math.pi * texture_freq * time_grid * 10)
        texture *= torch.sin(2 * math.pi * texture_freq * freq_grid * 5)
        spec = spec * (1 + texture)

        # Add noise
        noise = torch.rand_like(spec) - 0.5
        noise_amount = p["noise_amount"].view(batch_size, 1, 1)
        spec = spec + noise * noise_amount * spec

        # Apply brightness and contrast
        brightness = p["brightness"].view(batch_size, 1, 1)
        contrast = p["contrast"].view(batch_size, 1, 1)

        spec = (spec - 0.5) * contrast + 0.5
        spec = spec * brightness

        return spec.clamp(0, 1)

    def random_params(self, batch_size: int = None) -> torch.Tensor:
        bs = batch_size or self.batch_size
        return torch.rand(bs, self.n_params)


# Update config
config.SYNTH_PARAMS = SYNTH_PARAMS
config.N_PARAMS = len(SYNTH_PARAMS)

# Aliases
AmbientDrone = ParametricSynth
AmbientSynth = ParametricSynth
SpectralDrone = ParametricSynth
FullSpectrumSynth = ParametricSynth
SpectrogramSynth = ParametricSynth
