"""Ambient Synthesizer designed for evolving drone textures and musical spectrograms.

This synth creates slowly evolving ambient pads that can produce varied spectrograms
while still sounding musical. Uses multiple oscillator layers with slow modulation.
"""

import torch
import torch.nn as nn
import torchaudio.functional as F
import math

from . import config


# Synth parameters optimized for ambient music with spectral control
SYNTH_PARAMS = {
    # === Pitch content (determines VERTICAL position in spectrogram) ===
    "root_note": (30.0, 70.0),  # MIDI note (30=~50Hz, 70=~500Hz)
    "chord_type": (0.0, 1.0),  # 0=unison, 0.5=fifth, 1=octave spread
    "harmonic_density": (0.0, 1.0),  # How many harmonics (affects spectral height)
    # === Spectral distribution (where energy appears in spectrogram) ===
    "brightness": (0.0, 1.0),  # Filter cutoff - MOST IMPORTANT for visual
    "low_weight": (0.0, 1.0),  # Weight of bass frequencies
    "mid_weight": (0.0, 1.0),  # Weight of mid frequencies
    "high_weight": (0.0, 1.0),  # Weight of high frequencies
    # === Time evolution (determines HORIZONTAL patterns in spectrogram) ===
    "evolution_speed": (0.02, 0.5),  # How fast the sound evolves (Hz)
    "pulse_rate": (0.0, 2.0),  # Rhythmic pulsing (0=none)
    "fade_direction": (-1.0, 1.0),  # -1=fade out, 0=sustain, 1=fade in
    # === Texture (affects timbre and musicality) ===
    "noise_amount": (0.0, 0.3),  # Noise for texture
    "detuning": (0.0, 30.0),  # Cents - adds warmth, increases dissonance
    "reverb_amount": (0.3, 0.9),  # Ambient space
    "stereo_width": (0.0, 1.0),  # Stereo spread
    # === Envelope ===
    "attack": (1.0, 8.0),  # Long attacks for ambient (seconds)
    "release": (2.0, 10.0),  # Long releases (seconds)
}


class AmbientSynth(nn.Module):
    """Ambient synthesizer for generating evolving drone textures."""

    def __init__(self, batch_size: int = 1, sample_rate: int = None):
        super().__init__()

        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.num_samples = config.NUM_SAMPLES
        self.duration = config.DURATION
        self.batch_size = batch_size

        self.param_names = list(SYNTH_PARAMS.keys())
        self.n_params = len(self.param_names)

        # Pre-compute time array
        self.register_buffer("t", torch.linspace(0, self.duration, self.num_samples))

    def _denormalize_params(self, params: torch.Tensor) -> dict:
        """Convert normalized (0-1) params to actual values."""
        param_dict = {}
        for i, name in enumerate(self.param_names):
            low, high = SYNTH_PARAMS[name]
            param_dict[name] = params[:, i] * (high - low) + low
        return param_dict

    def _midi_to_hz(self, midi: torch.Tensor) -> torch.Tensor:
        """Convert MIDI note to frequency."""
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Render ambient audio from parameter vector."""
        device = params.device
        batch_size = params.shape[0]
        t = self.t.to(device)

        p = self._denormalize_params(params)

        # Base frequency from root note
        base_freq = self._midi_to_hz(p["root_note"])

        # Generate chord frequencies based on chord_type
        # 0 = unison, 0.5 = perfect fifth, 1 = octave spread
        chord_spread = p["chord_type"]
        freqs = [
            base_freq,
            base_freq * (1.0 + chord_spread * 0.5),  # Up to perfect fifth
            base_freq * (1.0 + chord_spread * 1.0),  # Up to octave
        ]

        # Initialize output
        audio = torch.zeros(batch_size, self.num_samples, device=device)

        # Generate each voice with detuning
        detune_cents = p["detuning"].unsqueeze(1)

        for voice_idx, freq in enumerate(freqs):
            # Apply slight detuning per voice for warmth
            freq_detuned = freq.unsqueeze(1) * (
                2.0 ** ((detune_cents * (voice_idx - 1)) / 1200.0)
            )

            # Generate harmonics based on harmonic_density
            voice = self._generate_harmonic_tone(
                freq_detuned,
                t,
                p["harmonic_density"],
                p["brightness"],
            )

            audio += voice

        # Apply spectral weighting
        audio = self._apply_spectral_bands(
            audio,
            p["low_weight"],
            p["mid_weight"],
            p["high_weight"],
        )

        # Add noise texture
        noise = torch.randn(batch_size, self.num_samples, device=device)
        noise = self._apply_lowpass_simple(noise, 4000)  # Band-limited noise
        audio += noise * p["noise_amount"].unsqueeze(1)

        # Apply time evolution (LFO-like modulation)
        evolution = self._generate_evolution(
            t,
            p["evolution_speed"],
            p["pulse_rate"],
        )
        audio *= evolution

        # Apply fade envelope
        fade_env = self._generate_fade_envelope(
            t,
            p["fade_direction"],
            p["attack"],
            p["release"],
        )
        audio *= fade_env

        # Apply simple reverb (delay-based)
        audio = self._apply_reverb(audio, p["reverb_amount"])

        # Apply brightness filter
        audio = self._apply_brightness_filter(audio, p["brightness"])

        # Normalize
        max_val = audio.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        audio = audio / max_val * 0.9

        return audio

    def _generate_harmonic_tone(
        self,
        freq: torch.Tensor,
        t: torch.Tensor,
        harmonic_density: torch.Tensor,
        brightness: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a tone with controllable harmonic content."""
        batch_size = freq.shape[0]

        # Fundamental
        phase = 2 * math.pi * freq * t.unsqueeze(0)
        wave = torch.sin(phase)

        # Add harmonics with natural rolloff
        max_harmonics = 16
        for h in range(2, max_harmonics + 1):
            harmonic_freq = freq * h
            # Skip if above Nyquist
            mask = (harmonic_freq < self.sample_rate / 2).float()

            # Harmonic amplitude: decreases with order, controlled by density
            # More density = more high harmonics
            harmonic_amp = harmonic_density.unsqueeze(1) * mask / (h**1.5)

            # Also modulate by brightness
            brightness_factor = brightness.unsqueeze(1) ** (h / 4.0)
            harmonic_amp *= brightness_factor

            harmonic_phase = 2 * math.pi * harmonic_freq * t.unsqueeze(0)
            wave += torch.sin(harmonic_phase) * harmonic_amp

        return wave

    def _apply_spectral_bands(
        self,
        audio: torch.Tensor,
        low_weight: torch.Tensor,
        mid_weight: torch.Tensor,
        high_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Apply frequency band weighting using simple filters."""
        batch_size = audio.shape[0]

        # Split into frequency bands using filters
        low = self._apply_lowpass_simple(audio, 300)
        high = audio - self._apply_lowpass_simple(audio, 2000)
        mid = audio - low - high

        # Weight each band
        result = (
            low * low_weight.unsqueeze(1)
            + mid * mid_weight.unsqueeze(1)
            + high * high_weight.unsqueeze(1)
        )

        return result

    def _apply_lowpass_simple(self, audio: torch.Tensor, cutoff: float) -> torch.Tensor:
        """Apply simple lowpass filter using FFT."""
        fft = torch.fft.rfft(audio)
        freqs = torch.fft.rfftfreq(audio.shape[-1], 1 / self.sample_rate).to(
            audio.device
        )

        # Smooth rolloff
        rolloff = 1.0 / (1.0 + (freqs / cutoff) ** 4)

        fft_filtered = fft * rolloff.unsqueeze(0)
        return torch.fft.irfft(fft_filtered, n=audio.shape[-1])

    def _generate_evolution(
        self,
        t: torch.Tensor,
        evolution_speed: torch.Tensor,
        pulse_rate: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time-varying modulation."""
        batch_size = evolution_speed.shape[0]

        # Slow evolution LFO
        evolution = 0.7 + 0.3 * torch.sin(
            2 * math.pi * evolution_speed.unsqueeze(1) * t.unsqueeze(0)
        )

        # Add pulsing if pulse_rate > 0
        pulse_mask = (pulse_rate > 0.1).float().unsqueeze(1)
        pulse = 0.7 + 0.3 * torch.sin(
            2 * math.pi * pulse_rate.unsqueeze(1) * t.unsqueeze(0)
        )

        return evolution * (1 - pulse_mask) + pulse * evolution * pulse_mask

    def _generate_fade_envelope(
        self,
        t: torch.Tensor,
        fade_direction: torch.Tensor,
        attack: torch.Tensor,
        release: torch.Tensor,
    ) -> torch.Tensor:
        """Generate fade envelope for ambient sustain."""
        batch_size = fade_direction.shape[0]
        t_norm = t.unsqueeze(0) / self.duration  # 0 to 1

        # Attack
        attack_norm = attack.unsqueeze(1) / self.duration
        attack_env = (t_norm / attack_norm.clamp(min=0.01)).clamp(0, 1)

        # Release
        release_norm = release.unsqueeze(1) / self.duration
        release_start = 1.0 - release_norm
        release_env = 1.0 - (
            (t_norm - release_start).clamp(min=0) / release_norm.clamp(min=0.01)
        ).clamp(0, 1)

        # Base envelope
        envelope = attack_env * release_env

        # Apply fade direction
        fade_multiplier = 1.0 + fade_direction.unsqueeze(1) * (t_norm - 0.5)
        envelope *= fade_multiplier.clamp(0.1, 2.0)

        return envelope.clamp(0.01, 1.0)

    def _apply_reverb(self, audio: torch.Tensor, amount: torch.Tensor) -> torch.Tensor:
        """Apply simple delay-based reverb."""
        batch_size = audio.shape[0]

        # Multiple delay taps for diffuse reverb
        delays_ms = [23, 47, 73, 97, 127, 163, 197, 233]
        decay = 0.5

        reverbed = audio.clone()
        for i, delay_ms in enumerate(delays_ms):
            delay_samples = int(delay_ms * self.sample_rate / 1000)
            if delay_samples >= audio.shape[1]:
                continue

            delayed = torch.zeros_like(audio)
            delayed[:, delay_samples:] = audio[:, :-delay_samples]
            reverbed += delayed * (decay ** (i + 1))

        # Mix dry/wet
        wet = amount.unsqueeze(1)
        return audio * (1 - wet * 0.5) + reverbed * wet * 0.5

    def _apply_brightness_filter(
        self, audio: torch.Tensor, brightness: torch.Tensor
    ) -> torch.Tensor:
        """Apply brightness-controlled lowpass filter."""
        filtered = []
        for i in range(audio.shape[0]):
            # Map brightness 0-1 to cutoff 500-12000 Hz
            cutoff = 500 + brightness[i].item() * 11500
            cutoff = min(cutoff, self.sample_rate / 2 - 100)

            filt = F.lowpass_biquad(
                audio[i : i + 1],
                sample_rate=self.sample_rate,
                cutoff_freq=cutoff,
                Q=0.7,
            )
            filtered.append(filt)

        return torch.cat(filtered, dim=0)

    def random_params(self, batch_size: int = None) -> torch.Tensor:
        """Generate random normalized parameters."""
        bs = batch_size or self.batch_size
        return torch.rand(bs, self.n_params)


# Update config with these parameters
config.SYNTH_PARAMS = SYNTH_PARAMS
config.N_PARAMS = len(SYNTH_PARAMS)

# Alias for compatibility
AmbientDrone = AmbientSynth
SpectralDrone = AmbientSynth
