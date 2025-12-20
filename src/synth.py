"""Full-spectrum Ambient Synthesizer for spectrogram painting.

This synth is specifically designed to produce content across the ENTIRE audible
frequency range (20Hz - 16kHz), allowing it to "paint" any pattern in a spectrogram.
"""

import torch
import torch.nn as nn
import torchaudio.functional as F
import math

from . import config


# Parameters designed for FULL SPECTRUM coverage
# Each parameter directly controls some aspect of the spectrogram
SYNTH_PARAMS = {
    # === Direct frequency band control (spectral painting) ===
    # These directly control brightness in different spectrogram regions
    "sub_bass": (0.0, 1.0),  # 20-80 Hz (very bottom)
    "bass": (0.0, 1.0),  # 80-250 Hz
    "low_mid": (0.0, 1.0),  # 250-500 Hz
    "mid": (0.0, 1.0),  # 500-2000 Hz
    "high_mid": (0.0, 1.0),  # 2000-6000 Hz
    "high": (0.0, 1.0),  # 6000-16000 Hz (very top)
    # === Texture within each band ===
    "tonality": (0.0, 1.0),  # 0=noise-based, 1=tonal
    "pitch_height": (0.0, 1.0),  # Within-band pitch (affects exact position)
    # === Time evolution ===
    "time_shape": (0.0, 1.0),  # 0=constant, 0.5=fade in, 1=fade out
    "evolution_rate": (0.0, 2.0),  # Temporal modulation speed
    # === Musicality ===
    "harmonic_richness": (0.0, 1.0),  # More harmonics = richer timbre
    "consonance": (0.0, 1.0),  # Higher = more musical intervals
    # === Envelope ===
    "attack": (0.5, 4.0),  # Seconds
    "sustain": (0.3, 1.0),  # Level
}


# Frequency bands (Hz) - covers full mel spectrogram
FREQ_BANDS = [
    (20, 80),  # sub_bass
    (80, 250),  # bass
    (250, 500),  # low_mid
    (500, 2000),  # mid
    (2000, 6000),  # high_mid
    (6000, 16000),  # high
]


class FullSpectrumSynth(nn.Module):
    """Synthesizer that can paint across the entire spectrogram."""

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

        # Get band levels
        band_levels = [
            p["sub_bass"],
            p["bass"],
            p["low_mid"],
            p["mid"],
            p["high_mid"],
            p["high"],
        ]

        audio = torch.zeros(batch_size, self.num_samples, device=device)

        # Generate content for each frequency band
        for i, ((f_low, f_high), level) in enumerate(zip(FREQ_BANDS, band_levels)):
            band_audio = self._generate_band(
                t,
                f_low,
                f_high,
                level,
                p["tonality"],
                p["pitch_height"],
                p["harmonic_richness"],
                p["consonance"],
                batch_size,
                device,
            )
            audio += band_audio

        # Apply time envelope
        envelope = self._generate_envelope(
            t, p["time_shape"], p["attack"], p["sustain"]
        )
        audio *= envelope

        # Apply temporal evolution
        evolution = self._generate_evolution(t, p["evolution_rate"])
        audio *= evolution

        # Normalize
        max_val = audio.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        audio = audio / max_val * 0.9

        return audio

    def _generate_band(
        self,
        t,
        f_low,
        f_high,
        level,
        tonality,
        pitch_height,
        harmonic_richness,
        consonance,
        batch_size,
        device,
    ):
        """Generate audio content for a specific frequency band."""
        # Base frequency within the band, controlled by pitch_height
        f_center = f_low + pitch_height.unsqueeze(1) * (f_high - f_low)

        # Mix of tonal and noise content
        # Tonal part
        tonal = self._generate_tones(t, f_center, harmonic_richness, consonance)

        # Noise part (band-limited)
        noise = self._generate_bandlimited_noise(t, f_low, f_high, batch_size, device)

        # Mix based on tonality parameter
        mix = tonality.unsqueeze(1) * tonal + (1 - tonality.unsqueeze(1)) * noise

        # Apply band level
        return mix * level.unsqueeze(1) * 0.3  # Scale factor for mixing

    def _generate_tones(self, t, f_center, harmonic_richness, consonance):
        """Generate tonal content with harmonics."""
        batch_size = f_center.shape[0]

        # Fundamental
        wave = torch.sin(2 * math.pi * f_center * t.unsqueeze(0))

        # Add harmonics
        for h in range(2, 8):
            # Consonant harmonics at octaves and fifths
            if h in [2, 4]:  # Octaves
                harmonic_f = f_center * h
            elif h == 3:  # Fifth
                harmonic_f = f_center * 1.5 * (consonance.unsqueeze(1) + 0.5)
            else:
                harmonic_f = f_center * h * (0.5 + 0.5 * consonance.unsqueeze(1))

            # Only add if below Nyquist
            mask = (harmonic_f < self.sample_rate / 2).float()

            # Amplitude decreases with harmonic number, controlled by richness
            amp = harmonic_richness.unsqueeze(1) * mask / (h**1.5)

            wave += torch.sin(2 * math.pi * harmonic_f * t.unsqueeze(0)) * amp

        return wave

    def _generate_bandlimited_noise(self, t, f_low, f_high, batch_size, device):
        """Generate noise limited to a specific frequency band."""
        # Generate white noise
        noise = torch.randn(batch_size, len(t), device=device)

        # FFT filter to band
        fft = torch.fft.rfft(noise)
        freqs = torch.fft.rfftfreq(len(t), 1 / self.sample_rate).to(device)

        # Bandpass filter
        mask = ((freqs >= f_low) & (freqs <= f_high)).float()
        # Smooth edges
        mask = mask.unsqueeze(0)

        fft_filtered = fft * mask
        return torch.fft.irfft(fft_filtered, n=len(t))

    def _generate_envelope(self, t, time_shape, attack, sustain):
        """Generate amplitude envelope."""
        batch_size = time_shape.shape[0]
        t_norm = t.unsqueeze(0) / self.duration

        # Attack
        attack_norm = attack.unsqueeze(1) / self.duration
        attack_env = (t_norm / attack_norm.clamp(min=0.01)).clamp(0, 1)

        # Sustain level
        envelope = attack_env * sustain.unsqueeze(1)

        # Time shape: 0=constant, 0.5=fade in, 1=fade out
        # Map to fade direction
        fade = 2 * (time_shape.unsqueeze(1) - 0.5)  # -1 to 1
        fade_env = 1.0 + fade * (t_norm - 0.5)
        envelope *= fade_env.clamp(0.1, 2.0)

        return envelope.clamp(0.01, 1.0)

    def _generate_evolution(self, t, evolution_rate):
        """Generate temporal modulation."""
        batch_size = evolution_rate.shape[0]

        modulation = 0.7 + 0.3 * torch.sin(
            2 * math.pi * evolution_rate.unsqueeze(1) * t.unsqueeze(0)
        )
        return modulation

    def random_params(self, batch_size: int = None) -> torch.Tensor:
        bs = batch_size or self.batch_size
        return torch.rand(bs, self.n_params)


# Update config
config.SYNTH_PARAMS = SYNTH_PARAMS
config.N_PARAMS = len(SYNTH_PARAMS)

# Aliases
AmbientDrone = FullSpectrumSynth
AmbientSynth = FullSpectrumSynth
SpectralDrone = FullSpectrumSynth
