"""SpectralDrone synthesizer optimized for visual spectrogram control.

This synth is designed to produce varied spectrograms that can match target images.
Uses additive synthesis with controllable frequency bands.
"""

import torch
import torch.nn as nn
import torchaudio.functional as F
import math

from . import config


# Define parameters that directly control spectral content
SYNTH_PARAMS = {
    # Frequency band levels (directly control spectrogram brightness at different heights)
    "band_low": (0.0, 1.0),  # 0-200 Hz (bottom of spectrogram)
    "band_mid_low": (0.0, 1.0),  # 200-500 Hz
    "band_mid": (0.0, 1.0),  # 500-1500 Hz
    "band_mid_high": (0.0, 1.0),  # 1500-4000 Hz
    "band_high": (0.0, 1.0),  # 4000-10000 Hz (top of spectrogram)
    # Time-varying modulation
    "time_fade": (-1.0, 1.0),  # -1=fade out, 0=constant, 1=fade in
    "time_pulse": (0.0, 5.0),  # Pulsing frequency (0=none)
    # Harmonic content (affects roughness)
    "harmonicity": (0.0, 1.0),  # 0=pure tone, 1=rich harmonics
    "detune": (0.0, 50.0),  # Cents detune between voices (affects roughness)
    # Global
    "fundamental": (40.0, 400.0),  # Base frequency
    "noise_level": (0.0, 0.5),  # Noise for texture
    "filter_q": (0.5, 10.0),  # Filter resonance
    # Envelope
    "attack": (0.01, 1.0),
    "release": (0.1, 2.0),
}


class SpectralDrone(nn.Module):
    """Synthesizer with direct spectral band control for visual matching."""

    def __init__(self, batch_size: int = 1, sample_rate: int = None):
        super().__init__()

        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.num_samples = config.NUM_SAMPLES
        self.batch_size = batch_size

        self.param_names = list(SYNTH_PARAMS.keys())
        self.n_params = len(self.param_names)

        # Pre-compute time array
        self.register_buffer("t", torch.linspace(0, config.DURATION, self.num_samples))

        # Frequency band centers (mel-spaced for better spectrogram coverage)
        self.band_centers = [100, 350, 900, 2500, 6000]
        self.band_widths = [100, 200, 600, 1500, 4000]

    def _denormalize_params(self, params: torch.Tensor) -> dict:
        """Convert normalized (0-1) params to actual values."""
        param_dict = {}
        for i, name in enumerate(self.param_names):
            low, high = SYNTH_PARAMS[name]
            param_dict[name] = params[:, i] * (high - low) + low
        return param_dict

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Render audio from parameter vector."""
        device = params.device
        batch_size = params.shape[0]
        t = self.t.to(device)

        p = self._denormalize_params(params)

        # Get band levels
        band_levels = torch.stack(
            [
                p["band_low"],
                p["band_mid_low"],
                p["band_mid"],
                p["band_mid_high"],
                p["band_high"],
            ],
            dim=1,
        )  # (batch, 5)

        # Generate each frequency band
        audio = torch.zeros(batch_size, self.num_samples, device=device)

        for i, (center, width) in enumerate(zip(self.band_centers, self.band_widths)):
            level = band_levels[:, i : i + 1]  # (batch, 1)

            # Base frequency in this band, modulated by fundamental
            freq = center * (p["fundamental"].unsqueeze(1) / 200.0)

            # Generate tone with harmonics
            tone = self._generate_band_tone(
                freq.squeeze(1),
                t,
                p["harmonicity"],
                p["detune"],
            )

            audio += tone * level

        # Add noise
        noise = torch.randn_like(audio) * p["noise_level"].unsqueeze(1)
        audio += noise

        # Apply time envelope
        time_env = self._generate_time_envelope(
            t, p["time_fade"], p["time_pulse"], p["attack"], p["release"]
        )
        audio *= time_env

        # Apply resonant filter at fundamental
        audio = self._apply_resonant_filter(audio, p["fundamental"], p["filter_q"])

        # Normalize
        max_val = audio.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        audio = audio / max_val * 0.9

        return audio

    def _generate_band_tone(
        self,
        freq: torch.Tensor,
        t: torch.Tensor,
        harmonicity: torch.Tensor,
        detune: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a tone with controllable harmonic content."""
        batch_size = freq.shape[0]

        # Fundamental
        wave = torch.sin(2 * math.pi * freq.unsqueeze(1) * t.unsqueeze(0))

        # Add harmonics based on harmonicity
        for h in range(2, 8):
            harmonic_freq = freq * h
            # Only add if below Nyquist
            mask = (harmonic_freq < self.sample_rate / 2).float()

            # Detune between even/odd harmonics for roughness
            detune_factor = 1.0 + (detune / 1200.0) * (h % 2)

            harmonic = torch.sin(
                2
                * math.pi
                * (harmonic_freq * detune_factor).unsqueeze(1)
                * t.unsqueeze(0)
            )

            # Weight by harmonicity and 1/h falloff
            weight = harmonicity.unsqueeze(1) * mask.unsqueeze(1) / h
            wave += harmonic * weight

        return wave

    def _generate_time_envelope(
        self,
        t: torch.Tensor,
        time_fade: torch.Tensor,
        time_pulse: torch.Tensor,
        attack: torch.Tensor,
        release: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time-varying envelope."""
        batch_size = time_fade.shape[0]
        duration = config.DURATION

        # Base envelope with attack/release
        t_norm = t.unsqueeze(0) / duration  # (1, samples)

        # Attack phase
        attack_norm = attack.unsqueeze(1) / duration
        attack_env = (t_norm / attack_norm.clamp(min=0.01)).clamp(0, 1)

        # Release phase
        release_norm = release.unsqueeze(1) / duration
        release_start = 1.0 - release_norm
        release_env = 1.0 - (
            (t_norm - release_start) / release_norm.clamp(min=0.01)
        ).clamp(0, 1)

        envelope = attack_env * release_env

        # Apply fade direction
        fade = t_norm * time_fade.unsqueeze(1)  # Linear fade
        envelope *= (0.5 + 0.5 * fade).clamp(0.1, 1.0)

        # Apply pulsing
        pulse = 0.5 + 0.5 * torch.cos(
            2 * math.pi * time_pulse.unsqueeze(1) * t.unsqueeze(0)
        )
        envelope *= pulse

        return envelope.clamp(0.01, 1.0)

    def _apply_resonant_filter(
        self,
        audio: torch.Tensor,
        cutoff: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        """Apply resonant low-pass filter."""
        filtered = []
        for i in range(audio.shape[0]):
            fc = cutoff[i].clamp(20, self.sample_rate / 2 - 100)
            q_val = q[i].clamp(0.5, 15)

            filt = F.lowpass_biquad(
                audio[i : i + 1],
                sample_rate=self.sample_rate,
                cutoff_freq=fc.item(),
                Q=q_val.item(),
            )
            filtered.append(filt)

        return torch.cat(filtered, dim=0)

    def random_params(self, batch_size: int = None) -> torch.Tensor:
        """Generate random normalized parameters."""
        bs = batch_size or self.batch_size
        return torch.rand(bs, self.n_params)


# Update the module-level config
config.SYNTH_PARAMS = SYNTH_PARAMS
config.N_PARAMS = len(SYNTH_PARAMS)


# Alias for backwards compatibility
AmbientDrone = SpectralDrone
