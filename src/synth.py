"""Noise-based synthesizer for smooth spectrogram painting.

PROBLEM: Additive synthesis with discrete carriers creates horizontal stripes
because each carrier appears as a thin horizontal line in the spectrogram.

SOLUTION: Use filtered noise as the primary sound source. Noise fills the
entire spectrum smoothly, and we can sculpt it with filters to control
which frequency regions are bright or dark.
"""

import torch
import torch.nn as nn
import math

from . import config


# Parameters for noise-based spectral shaping
SYNTH_PARAMS = {
    # Amplitude for each frequency region (controls spectrogram brightness)
    # 8 overlapping bands for smoother coverage
    "amp_low": (0.0, 1.0),  # 200-500 Hz
    "amp_lowmid": (0.0, 1.0),  # 400-1000 Hz
    "amp_mid": (0.0, 1.0),  # 800-2000 Hz
    "amp_midhigh": (0.0, 1.0),  # 1600-4000 Hz
    "amp_high": (0.0, 1.0),  # 3200-8000 Hz
    "amp_top": (0.0, 1.0),  # 6000-12000 Hz
    # Time evolution
    "time_center": (0.0, 1.0),  # Where peak energy is in time
    "time_spread": (0.3, 1.0),  # How spread out in time
    # Texture
    "tonal_mix": (0.0, 0.5),  # Add subtle tones for musicality
    "modulation": (0.0, 1.0),  # Temporal modulation depth
}


# Overlapping frequency bands for smooth coverage
FREQ_BANDS = [
    (200, 500),
    (400, 1000),
    (800, 2000),
    (1600, 4000),
    (3200, 8000),
    (6000, 12000),
]


class NoiseSynth(nn.Module):
    """Noise-based synthesizer with smooth spectral control."""

    def __init__(self, batch_size: int = 1, sample_rate: int = None):
        super().__init__()

        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.num_samples = config.NUM_SAMPLES
        self.duration = config.DURATION
        self.batch_size = batch_size

        self.param_names = list(SYNTH_PARAMS.keys())
        self.n_params = len(self.param_names)

        self.register_buffer("t", torch.linspace(0, self.duration, self.num_samples))

    def _denormalize_params(self, params: torch.Tensor) -> dict:
        param_dict = {}
        for i, name in enumerate(self.param_names):
            low, high = SYNTH_PARAMS[name]
            param_dict[name] = params[:, i] * (high - low) + low
        return param_dict

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        device = params.device
        batch_size = params.shape[0]
        t = self.t.to(device)

        p = self._denormalize_params(params)

        # Get band amplitudes
        band_amps = [
            p["amp_low"],
            p["amp_lowmid"],
            p["amp_mid"],
            p["amp_midhigh"],
            p["amp_high"],
            p["amp_top"],
        ]

        audio = torch.zeros(batch_size, self.num_samples, device=device)

        # Generate band-limited noise for each frequency region
        for (f_low, f_high), amp in zip(FREQ_BANDS, band_amps):
            # Create white noise
            noise = torch.randn(batch_size, self.num_samples, device=device)

            # Bandpass filter the noise (smooth spectral shaping)
            filtered = self._smooth_bandpass(noise, f_low, f_high, device)

            # Apply amplitude
            audio += filtered * amp.unsqueeze(1) * 0.3

        # Add subtle tonal component for musicality (optional)
        tonal_mix = p["tonal_mix"]
        if tonal_mix.max() > 0.05:
            tonal = self._generate_tonal(t, batch_size, device)
            audio = audio * (1 - tonal_mix.unsqueeze(1)) + tonal * tonal_mix.unsqueeze(
                1
            )

        # Apply time envelope
        envelope = self._time_envelope(t, p["time_center"], p["time_spread"])
        audio *= envelope

        # Apply modulation
        mod = p["modulation"]
        if mod.max() > 0.1:
            mod_signal = 1.0 - mod.unsqueeze(1) * 0.5 * (
                1 - torch.cos(2 * math.pi * 2 * t.unsqueeze(0))
            )
            audio *= mod_signal

        # Normalize
        max_val = audio.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        audio = audio / max_val * 0.9

        return audio

    def _smooth_bandpass(self, noise, f_low, f_high, device):
        """Apply bandpass filter using FFT with sharp sigmoid edges."""
        fft = torch.fft.rfft(noise)
        freqs = torch.fft.rfftfreq(noise.shape[-1], 1 / self.sample_rate).to(device)

        # Sigmoid edges for smooth but SHARP cutoff (no leakage)
        edge_width = max((f_high - f_low) * 0.05, 5.0)  # 5% of bandwidth

        low_mask = torch.sigmoid((freqs - f_low) / edge_width)
        high_mask = torch.sigmoid((f_high - freqs) / edge_width)
        mask = low_mask * high_mask

        fft_filtered = fft * mask.unsqueeze(0)
        return torch.fft.irfft(fft_filtered, n=noise.shape[-1])

    def _generate_tonal(self, t, batch_size, device):
        """Generate tonal content in MID frequencies (not bass)."""
        # Higher frequencies that don't pollute low end
        freqs = [880, 1100, 1320, 1760]  # A4-A5 range
        tonal = torch.zeros(batch_size, len(t), device=device)

        for f in freqs:
            tonal += torch.sin(2 * math.pi * f * t.unsqueeze(0)) * 0.05

        return tonal

    def _time_envelope(self, t, time_center, time_spread):
        """Generate time envelope."""
        batch_size = time_center.shape[0]
        t_norm = t.unsqueeze(0) / self.duration

        center = time_center.unsqueeze(1)
        spread = time_spread.unsqueeze(1)

        # Smooth envelope
        envelope = torch.exp(-0.5 * ((t_norm - center) / spread.clamp(min=0.1)) ** 2)
        envelope = envelope.clamp(min=0.1)  # Keep some minimum level

        # Smooth attack
        attack = (t_norm / 0.03).clamp(0, 1)
        envelope *= attack

        return envelope

    def random_params(self, batch_size: int = None) -> torch.Tensor:
        bs = batch_size or self.batch_size
        return torch.rand(bs, self.n_params)


# Update config
config.SYNTH_PARAMS = SYNTH_PARAMS
config.N_PARAMS = len(SYNTH_PARAMS)

# Aliases
AmbientDrone = NoiseSynth
AmbientSynth = NoiseSynth
SpectralDrone = NoiseSynth
FullSpectrumSynth = NoiseSynth
ParametricSynth = NoiseSynth
SpectrogramSynth = NoiseSynth
AdditiveSynth = NoiseSynth
