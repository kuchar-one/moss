"""ImageBlendSynth: Blends target image with musical spectrogram.

PERFORMANCE: Use forward_spec_only() during optimization (no Griffin-Lim).
             Use forward() only for final audio rendering.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T

from . import config


SYNTH_PARAMS = {
    "blend_global": (0.0, 1.0),
    "blend_low": (0.0, 1.0),
    "blend_mid": (0.0, 1.0),
    "blend_high": (0.0, 1.0),
    "fundamental_bin": (0.1, 0.8),
    "harmonic_spread": (0.3, 1.0),
    "n_harmonics": (0.2, 1.0),
    "harmonic_decay": (0.3, 0.9),
    "attack": (0.0, 0.5),
    "sustain": (0.5, 1.0),
    "intensity": (0.5, 1.0),
}


class ImageBlendSynth(nn.Module):
    """Synthesizer that blends target image with musical spectrogram."""

    def __init__(self, target_image: torch.Tensor, sample_rate: int = None):
        super().__init__()

        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.num_samples = config.NUM_SAMPLES
        self.duration = config.DURATION

        self.param_names = list(SYNTH_PARAMS.keys())
        self.n_params = len(self.param_names)

        target = target_image.squeeze(0)
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        self.register_buffer("target_image", target)

        self.n_mels = target.shape[0]
        self.n_frames = target.shape[1]
        self._last_blended_spec = None

    def _denormalize_params(self, params: torch.Tensor) -> dict:
        param_dict = {}
        for i, name in enumerate(self.param_names):
            low, high = SYNTH_PARAMS[name]
            param_dict[name] = params[:, i] * (high - low) + low
        return param_dict

    def forward_spec_only(self, params: torch.Tensor) -> torch.Tensor:
        """FAST: Returns blended spectrogram only (no audio).

        Use this during optimization for visual loss computation.
        """
        device = params.device
        batch_size = params.shape[0]

        p = self._denormalize_params(params)
        music_spec = self._generate_musical_spectrogram(p, device, batch_size)
        target = self.target_image.unsqueeze(0).expand(batch_size, -1, -1)
        blend_map = self._create_blend_map(p, device, batch_size)

        blended_spec = blend_map * music_spec + (1 - blend_map) * target
        self._last_blended_spec = blended_spec.detach()

        return blended_spec

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """SLOW: Returns audio (uses Griffin-Lim).

        Use only for final audio rendering after optimization.
        """
        blended_spec = self.forward_spec_only(params)
        audio = self._spectrogram_to_audio(blended_spec, params.device)
        return audio

    def get_last_blended_spec(self) -> torch.Tensor:
        return self._last_blended_spec

    def _generate_musical_spectrogram(self, p, device, batch_size):
        H, W = self.n_mels, self.n_frames

        spec = torch.zeros(batch_size, H, W, device=device)
        mel_bins = torch.arange(H, device=device, dtype=torch.float32)
        time_bins = torch.arange(W, device=device, dtype=torch.float32)

        fund_bin = (p["fundamental_bin"] * H).long().clamp(0, H - 1)
        n_harm = (2 + p["n_harmonics"] * 8).long()

        for batch_idx in range(batch_size):
            fb = fund_bin[batch_idx].item()
            nh = n_harm[batch_idx].item()
            spread = p["harmonic_spread"][batch_idx].item()
            decay = p["harmonic_decay"][batch_idx].item()
            attack = p["attack"][batch_idx].item()
            sustain = p["sustain"][batch_idx].item()
            intensity = p["intensity"][batch_idx].item()

            for i in range(nh):
                h_bin = int(fb + i * spread * H / nh)
                if h_bin >= H:
                    break

                amp = intensity / (1 + i * decay)
                gaussian = torch.exp(-0.5 * ((mel_bins - h_bin) / 2) ** 2)

                t_norm = time_bins / W
                env = torch.ones_like(t_norm)
                env = env * (t_norm / (attack + 0.01)).clamp(0, 1)
                env = env * max(0.3, min(1.0, sustain))

                spec[batch_idx] += gaussian.unsqueeze(1) * env.unsqueeze(0) * amp

        for i in range(batch_size):
            s = spec[i]
            if s.max() > 0:
                spec[i] = s / s.max()

        return spec

    def _create_blend_map(self, p, device, batch_size):
        H, W = self.n_mels, self.n_frames

        blend = p["blend_global"].view(batch_size, 1, 1).expand(-1, H, W).clone()
        freq_bins = torch.linspace(0, 1, H, device=device)

        low_mask = (freq_bins < 0.33).float().view(1, H, 1)
        blend = blend + (p["blend_low"].view(batch_size, 1, 1) - 0.5) * low_mask * 0.3

        mid_mask = ((freq_bins >= 0.33) & (freq_bins < 0.66)).float().view(1, H, 1)
        blend = blend + (p["blend_mid"].view(batch_size, 1, 1) - 0.5) * mid_mask * 0.3

        high_mask = (freq_bins >= 0.66).float().view(1, H, 1)
        blend = blend + (p["blend_high"].view(batch_size, 1, 1) - 0.5) * high_mask * 0.3

        return blend.clamp(0, 1)

    def _spectrogram_to_audio(self, spec, device):
        batch_size = spec.shape[0]

        audios = []
        for i in range(batch_size):
            spec_power = spec[i] ** 2 * 100

            linear_spec = torch.nn.functional.interpolate(
                spec_power.unsqueeze(0).unsqueeze(0),
                size=(config.N_FFT // 2 + 1, spec_power.shape[-1]),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

            griffin_lim = T.GriffinLim(
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                power=2.0,
                n_iter=32,
            )
            audio = griffin_lim(linear_spec.cpu())

            if audio.shape[-1] < self.num_samples:
                audio = torch.nn.functional.pad(
                    audio, (0, self.num_samples - audio.shape[-1])
                )
            else:
                audio = audio[: self.num_samples]

            audios.append(audio.to(device))

        audio = torch.stack(audios)
        max_val = audio.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        audio = audio / max_val * 0.9

        return audio

    def random_params(self, batch_size: int = 1) -> torch.Tensor:
        return torch.rand(batch_size, self.n_params)


config.SYNTH_PARAMS = SYNTH_PARAMS
config.N_PARAMS = len(SYNTH_PARAMS)
