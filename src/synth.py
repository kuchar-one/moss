"""AmbientDrone synthesizer using pure PyTorch.

This is the "Genome Decoder" - takes parameter vectors and renders audio.
Uses pure PyTorch for GPU-accelerated batch audio synthesis.
"""

import torch
import torch.nn as nn
import torchaudio.functional as F

from . import config


class AmbientDrone(nn.Module):
    """A synthesizer voice designed for ambient drone textures.

    Architecture:
        - VCO1 (Sine): Sub-bass foundation
        - VCO2 (Sawtooth/Square): Harmonic texture with LFO modulation
        - Noise: White noise for air/texture
        - Mix: Blend all sources
        - ADSR Envelope: Amplitude shaping
        - Low-pass filter: Controlled by cutoff for brightness

    The synth takes normalized parameters (0-1) and renders batched audio.
    """

    def __init__(self, batch_size: int = 1, sample_rate: int = None):
        super().__init__()

        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.num_samples = config.NUM_SAMPLES
        self.batch_size = batch_size

        # Parameter names in order (must match config.SYNTH_PARAMS keys)
        self.param_names = list(config.SYNTH_PARAMS.keys())
        self.n_params = len(self.param_names)

        # Pre-compute time array for efficiency
        self.register_buffer("t", torch.linspace(0, config.DURATION, self.num_samples))

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

    def _generate_sine(self, freq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generate sine wave.

        Args:
            freq: Frequencies of shape (batch,)
            t: Time array of shape (samples,)

        Returns:
            Audio of shape (batch, samples)
        """
        return torch.sin(2 * torch.pi * freq.unsqueeze(1) * t.unsqueeze(0))

    def _generate_sawtooth(self, freq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generate sawtooth wave using additive synthesis.

        Args:
            freq: Frequencies of shape (batch,)
            t: Time array of shape (samples,)

        Returns:
            Audio of shape (batch, samples)
        """
        # Simple sawtooth approximation using first 20 harmonics
        wave = torch.zeros(freq.shape[0], t.shape[0], device=freq.device)
        for n in range(1, 21):
            harmonic_freq = freq * n
            # Skip harmonics above Nyquist
            mask = harmonic_freq < (self.sample_rate / 2)
            harmonic = torch.sin(
                2 * torch.pi * harmonic_freq.unsqueeze(1) * t.unsqueeze(0)
            )
            wave += (harmonic * mask.unsqueeze(1).float()) / n
        return wave * (2 / torch.pi)  # Normalize

    def _generate_square(self, freq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generate square wave using additive synthesis.

        Args:
            freq: Frequencies of shape (batch,)
            t: Time array of shape (samples,)

        Returns:
            Audio of shape (batch, samples)
        """
        # Square wave using odd harmonics
        wave = torch.zeros(freq.shape[0], t.shape[0], device=freq.device)
        for n in range(1, 21, 2):  # Odd harmonics only
            harmonic_freq = freq * n
            mask = harmonic_freq < (self.sample_rate / 2)
            harmonic = torch.sin(
                2 * torch.pi * harmonic_freq.unsqueeze(1) * t.unsqueeze(0)
            )
            wave += (harmonic * mask.unsqueeze(1).float()) / n
        return wave * (4 / torch.pi)  # Normalize

    def _generate_adsr(
        self,
        attack: torch.Tensor,
        decay: torch.Tensor,
        sustain: torch.Tensor,
        release: torch.Tensor,
        t: torch.Tensor,
        duration: float,
    ) -> torch.Tensor:
        """Generate ADSR envelope.

        Args:
            attack, decay, sustain, release: ADSR parameters of shape (batch,)
            t: Time array of shape (samples,)
            duration: Total duration in seconds

        Returns:
            Envelope of shape (batch, samples)
        """
        batch_size = attack.shape[0]
        envelope = torch.zeros(batch_size, t.shape[0], device=attack.device)

        # Expand t for batch processing
        t_exp = t.unsqueeze(0).expand(batch_size, -1)

        # Attack phase
        attack_mask = t_exp < attack.unsqueeze(1)
        attack_env = t_exp / attack.unsqueeze(1).clamp(min=1e-6)
        envelope = torch.where(attack_mask, attack_env, envelope)

        # Decay phase
        decay_start = attack.unsqueeze(1)
        decay_end = (attack + decay).unsqueeze(1)
        decay_mask = (t_exp >= decay_start) & (t_exp < decay_end)
        decay_progress = (t_exp - decay_start) / decay.unsqueeze(1).clamp(min=1e-6)
        decay_env = 1.0 - (1.0 - sustain.unsqueeze(1)) * decay_progress
        envelope = torch.where(decay_mask, decay_env, envelope)

        # Sustain phase
        sustain_end = duration - release.unsqueeze(1)
        sustain_mask = (t_exp >= decay_end) & (t_exp < sustain_end)
        envelope = torch.where(sustain_mask, sustain.unsqueeze(1), envelope)

        # Release phase
        release_mask = t_exp >= sustain_end
        release_progress = (t_exp - sustain_end) / release.unsqueeze(1).clamp(min=1e-6)
        release_env = sustain.unsqueeze(1) * (1.0 - release_progress.clamp(0, 1))
        envelope = torch.where(release_mask, release_env.clamp(min=0), envelope)

        return envelope.clamp(0, 1)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Render audio from normalized parameter vector.

        Args:
            params: Tensor of shape (batch_size, n_params) with values in [0, 1]

        Returns:
            Audio tensor of shape (batch_size, num_samples)
        """
        device = params.device
        batch_size = params.shape[0]

        # Ensure time buffer is on correct device
        t = self.t.to(device)

        # Denormalize parameters
        p = self._denormalize_params(params)

        # Generate VCO1 (sub-bass sine)
        vco1 = self._generate_sine(p["vco1_freq"], t)

        # Generate VCO2 with shape blend
        # Apply LFO to VCO2 frequency
        lfo = torch.sin(2 * torch.pi * p["lfo_rate"].unsqueeze(1) * t.unsqueeze(0))
        lfo_cents = lfo * p["lfo_depth"].unsqueeze(1)
        vco2_freq_mod = p["vco2_freq"].unsqueeze(1) * (2 ** (lfo_cents / 1200))
        # Average modulated frequency for oscillator (simplified)
        vco2_freq_avg = p["vco2_freq"] + p["vco2_detune"] / 100 * p["vco2_freq"]

        # Blend between saw and square based on shape
        saw = self._generate_sawtooth(vco2_freq_avg, t)
        square = self._generate_square(vco2_freq_avg, t)
        shape = p["vco2_shape"].unsqueeze(1)
        vco2 = saw * (1 - shape) + square * shape

        # Apply LFO as amplitude modulation for "drift" effect
        vco2 = vco2 * (1 + 0.3 * lfo)

        # Generate noise
        noise = torch.randn(batch_size, t.shape[0], device=device)

        # Mix sources
        mixed = (
            vco1 * p["vco1_level"].unsqueeze(1)
            + vco2 * p["vco2_level"].unsqueeze(1)
            + noise * p["noise_level"].unsqueeze(1)
        )

        # Generate and apply ADSR envelope
        envelope = self._generate_adsr(
            p["attack"], p["decay"], p["sustain"], p["release"], t, config.DURATION
        )
        output = mixed * envelope

        # Apply low-pass filter
        output = self._apply_lowpass(output, p["filter_cutoff"], p["filter_q"])

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
            cutoff: Shape (batch,) - cutoff frequency in Hz
            Q: Shape (batch,) - filter Q/resonance

        Returns:
            Filtered audio
        """
        # Process each batch item separately due to different filter params
        filtered = []
        for i in range(audio.shape[0]):
            # Clamp cutoff to valid range
            fc = cutoff[i].clamp(20, self.sample_rate / 2 - 100)
            q = Q[i].clamp(0.1, 20)

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
