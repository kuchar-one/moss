"""AmbientDrone synthesizer using TorchSynth modules.

This is the "Genome Decoder" - takes parameter vectors and renders audio.
"""

import torch
import torch.nn as nn
import torchaudio.functional as F
from torchsynth.module import (
    SineVCO,
    SquareSawVCO,
    Noise,
    VCA,
    ADSR,
    AudioMixer,
    ControlRateUpsample,
    MonophonicKeyboard,
)
from torchsynth.config import SynthConfig

from . import config


class AmbientDrone(nn.Module):
    """A modular synthesizer voice designed for ambient drone textures.

    Architecture:
        - VCO1 (Sine): Sub-bass foundation
        - VCO2 (SquareSaw): Harmonic texture with LFO modulation
        - Noise: White noise for air/texture
        - AudioMixer: Blend all sources
        - VCA: Final amplitude envelope
        - Low-pass filter: Controlled by ADSR envelope for brightness

    The synth takes normalized parameters (0-1) and renders batched audio.
    """

    def __init__(
        self, batch_size: int = 1, sample_rate: int = None, buffer_size: int = None
    ):
        super().__init__()

        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.buffer_size = buffer_size or config.NUM_SAMPLES
        self.batch_size = batch_size

        # Create synth config
        self.synth_config = SynthConfig(
            batch_size=batch_size,
            sample_rate=self.sample_rate,
            buffer_size_seconds=config.DURATION,
        )

        # Initialize modules
        self.keyboard = MonophonicKeyboard(self.synth_config)
        self.vco1 = SineVCO(self.synth_config)  # Sub-bass
        self.vco2 = SquareSawVCO(self.synth_config)  # Harmonic texture
        self.noise = Noise(self.synth_config)
        self.adsr = ADSR(self.synth_config)
        self.upsample = ControlRateUpsample(self.synth_config)
        self.vca = VCA(self.synth_config)
        self.mixer = AudioMixer(self.synth_config, n_input=3)

        # Parameter names in order (must match config.SYNTH_PARAMS keys)
        self.param_names = list(config.SYNTH_PARAMS.keys())
        self.n_params = len(self.param_names)

    def _denormalize_params(self, params: torch.Tensor) -> dict:
        """Convert normalized (0-1) params to actual values.

        Args:
            params: Tensor of shape (batch_size, n_params) with values in [0, 1]

        Returns:
            Dictionary mapping param names to denormalized tensors
        """
        param_dict = {}
        for i, name in enumerate(self.param_names):
            low, high = config.SYNTH_PARAMS[name]
            param_dict[name] = params[:, i] * (high - low) + low
        return param_dict

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Render audio from normalized parameter vector.

        Args:
            params: Tensor of shape (batch_size, n_params) with values in [0, 1]

        Returns:
            Audio tensor of shape (batch_size, num_samples)
        """
        device = params.device
        batch_size = params.shape[0]

        # Denormalize parameters
        p = self._denormalize_params(params)

        # Convert frequencies to MIDI for keyboard/VCO
        # f = 440 * 2^((m-69)/12) => m = 12 * log2(f/440) + 69
        vco1_midi = 12 * torch.log2(p["vco1_freq"] / 440.0) + 69
        vco2_midi = 12 * torch.log2(p["vco2_freq"] / 440.0) + 69

        # Normalize ADSR values for torchsynth (expects 0-1)
        attack_norm = (p["attack"] - 0.01) / (2.0 - 0.01)
        decay_norm = (p["decay"] - 0.1) / (2.0 - 0.1)
        sustain_norm = p["sustain"]  # Already 0-1 range conceptually
        release_norm = (p["release"] - 0.1) / (3.0 - 0.1)

        # Generate ADSR envelope
        envelope = self.adsr(
            attack_norm.unsqueeze(1),
            decay_norm.unsqueeze(1),
            sustain_norm.unsqueeze(1),
            release_norm.unsqueeze(1),
            torch.ones(batch_size, 1, device=device),  # alpha (curve)
        )
        envelope_audio = self.upsample(envelope)

        # Create simple LFO (sine wave at control rate, then upsample)
        t = torch.linspace(0, config.DURATION, envelope.shape[-1], device=device)
        lfo = torch.sin(2 * torch.pi * p["lfo_rate"].unsqueeze(1) * t)
        lfo_depth_cents = p["lfo_depth"].unsqueeze(1)
        # LFO modulates pitch in cents: cents -> ratio = 2^(cents/1200)
        lfo_ratio = 2 ** (lfo * lfo_depth_cents / 1200)
        lfo_audio = self.upsample(lfo_ratio)

        # Generate VCO1 (sub-bass sine)
        vco1_pitch = vco1_midi.unsqueeze(1).expand(-1, 1)
        vco1_audio = self.vco1(
            vco1_pitch / 127.0,  # Normalize to 0-1
            torch.zeros(batch_size, 1, device=device),  # mod_depth
        )

        # Generate VCO2 (saw/square with LFO modulation)
        vco2_pitch = vco2_midi.unsqueeze(1).expand(-1, 1)
        vco2_base = self.vco2(
            vco2_pitch / 127.0,
            p["vco2_shape"].unsqueeze(1),  # shape: 0=saw, 1=square
        )
        # Apply LFO pitch modulation by resampling (simple approximation)
        # For true pitch mod we'd need to integrate, but this gives drift effect
        vco2_audio = vco2_base * lfo_audio

        # Generate noise
        noise_audio = self.noise()

        # Mix sources with levels
        vco1_level = p["vco1_level"].unsqueeze(1).unsqueeze(2)
        vco2_level = p["vco2_level"].unsqueeze(1).unsqueeze(2)
        noise_level = p["noise_level"].unsqueeze(1).unsqueeze(2)

        # Stack and mix
        sources = torch.stack(
            [
                vco1_audio * vco1_level.squeeze(2),
                vco2_audio * vco2_level.squeeze(2),
                noise_audio * noise_level.squeeze(2),
            ],
            dim=1,
        )

        # Simple sum mix
        mixed = sources.sum(dim=1)

        # Apply VCA envelope
        output = mixed * envelope_audio

        # Apply low-pass filter
        # Filter cutoff modulated by envelope for filter sweep
        cutoff = p["filter_cutoff"].unsqueeze(1)
        Q = p["filter_q"].unsqueeze(1)

        # Apply filter per-sample in batch
        output = self._apply_lowpass(output, cutoff, Q)

        # Normalize output
        max_val = output.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        output = output / max_val * 0.9

        return output

    def _apply_lowpass(
        self, audio: torch.Tensor, cutoff: torch.Tensor, Q: torch.Tensor
    ) -> torch.Tensor:
        """Apply biquad lowpass filter to audio.

        Args:
            audio: Shape (batch, samples)
            cutoff: Shape (batch, 1) - cutoff frequency in Hz
            Q: Shape (batch, 1) - filter Q/resonance

        Returns:
            Filtered audio
        """
        # Process each batch item separately due to different filter params
        filtered = []
        for i in range(audio.shape[0]):
            # Clamp cutoff to valid range
            fc = cutoff[i, 0].clamp(20, self.sample_rate / 2 - 100)
            q = Q[i, 0].clamp(0.1, 20)

            filt = F.lowpass_biquad(
                audio[i : i + 1],
                sample_rate=self.sample_rate,
                cutoff_freq=fc.item(),
                Q=q.item(),
            )
            filtered.append(filt)

        return torch.cat(filtered, dim=0)

    def random_params(self, batch_size: int = None) -> torch.Tensor:
        """Generate random normalized parameters.

        Args:
            batch_size: Number of parameter sets to generate

        Returns:
            Tensor of shape (batch_size, n_params) with values in [0, 1]
        """
        bs = batch_size or self.batch_size
        return torch.rand(bs, self.n_params)
