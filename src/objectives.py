"""Objective functions for multi-objective optimization.

Two conflicting objectives:
1. Visual Loss: How well does the spectrogram match the target image?
2. Musical Loss: How "noisy" vs "tonal" is the audio?

Key insight: Good visual match requires complex spectral content to fill the spectrogram,
while good musicality requires simpler, more harmonic/tonal content.
"""

import torch
import numpy as np
from pytorch_msssim import ssim
from scipy.signal import find_peaks

from . import config


def calc_visual_loss(
    generated_spec: torch.Tensor, target_img: torch.Tensor
) -> torch.Tensor:
    """Calculate visual similarity loss using SSIM.

    Args:
        generated_spec: Tensor of shape (batch, height, width)
        target_img: Tensor of shape (1, height, width) - the target image

    Returns:
        Loss tensor of shape (batch,) where lower is better (1 - SSIM)
    """
    batch_size = generated_spec.shape[0]
    device = generated_spec.device

    target_expanded = target_img.expand(batch_size, -1, -1).to(device)

    gen_4d = generated_spec.unsqueeze(1)
    tgt_4d = target_expanded.unsqueeze(1)

    losses = []
    for i in range(batch_size):
        ssim_val = ssim(
            gen_4d[i : i + 1],
            tgt_4d[i : i + 1],
            data_range=1.0,
            size_average=True,
        )
        losses.append(1.0 - ssim_val)

    return torch.stack(losses)


def calc_musical_loss(audio: torch.Tensor) -> torch.Tensor:
    """Calculate musical quality loss based on spectral entropy and tonality.

    This objective CONFLICTS with visual matching because:
    - Tonal sounds (musical) = peaked spectrum = predictable spectrograms
    - Complex/noisy sounds (visual matching) = flat spectrum = can paint varied patterns

    Uses:
    1. Spectral Entropy: Measures disorder in frequency distribution
       - Low entropy = tonal = musical
       - High entropy = noisy = less musical but good for visual variety
    2. Spectral Crest: Peak-to-mean ratio
       - High crest = tonal peaks = musical
       - Low crest = flat = noisy

    Args:
        audio: Tensor of shape (batch, samples)

    Returns:
        Loss tensor of shape (batch,) in [0, 1] where lower is more musical
    """
    batch_size = audio.shape[0]
    device = audio.device
    losses = []

    for i in range(batch_size):
        wave = audio[i]

        # Silence penalty
        energy = torch.mean(wave**2)
        if energy < 0.001:
            losses.append(torch.tensor(1.0, device=device))
            continue

        # Compute magnitude spectrum
        fft = torch.fft.rfft(wave)
        magnitude = torch.abs(fft)

        # Focus on musical frequency range (50Hz - 8kHz)
        freq_per_bin = (config.SAMPLE_RATE / 2) / len(magnitude)
        low_bin = max(1, int(50 / freq_per_bin))
        high_bin = min(len(magnitude), int(8000 / freq_per_bin))
        magnitude = magnitude[low_bin:high_bin]

        if len(magnitude) < 10:
            losses.append(torch.tensor(0.5, device=device))
            continue

        # Normalize to probability distribution
        mag_sum = magnitude.sum() + 1e-10
        p = magnitude / mag_sum

        # Spectral Entropy (higher = noisier = less musical)
        log_p = torch.log2(p + 1e-10)
        entropy = -torch.sum(p * log_p)
        max_entropy = torch.log2(torch.tensor(float(len(p)), device=device))
        normalized_entropy = (entropy / max_entropy).clamp(0, 1)

        # Spectral Crest: peakiness (higher = more tonal = more musical)
        spectral_crest = magnitude.max() / (magnitude.mean() + 1e-10)
        # Invert: low crest -> high loss, high crest -> low loss
        # Typical crest values: 5-50 for tonal, 2-10 for noisy
        crest_loss = 1.0 - (spectral_crest / (spectral_crest + 10.0))

        # Combined loss: emphasize entropy
        musical_loss = 0.7 * normalized_entropy + 0.3 * crest_loss

        losses.append(musical_loss.clamp(0, 1))

    return torch.stack(losses)


def calc_roughness(audio: torch.Tensor) -> torch.Tensor:
    """Calculate perceptual roughness using Plomp-Levelt model (alternative metric)."""
    batch_size = audio.shape[0]
    device = audio.device
    losses = []

    for i in range(batch_size):
        wave = audio[i]

        energy = torch.mean(wave**2)
        if energy < 0.001:
            losses.append(torch.tensor(1.0, device=device))
            continue

        fft = torch.fft.rfft(wave)
        magnitude = torch.abs(fft)
        mag_np = magnitude.cpu().numpy()

        peaks, _ = find_peaks(mag_np, height=mag_np.max() * 0.1)

        if len(peaks) < 2:
            losses.append(torch.tensor(0.1, device=device))
            continue

        peak_heights = mag_np[peaks]
        top_indices = np.argsort(peak_heights)[-10:]
        top_peaks = peaks[top_indices]

        freqs = top_peaks * config.SAMPLE_RATE / config.NUM_SAMPLES
        amps = mag_np[top_peaks]
        amps = amps / (amps.max() + 1e-10)

        roughness = _plomp_levelt_roughness(freqs, amps)
        # Normalize to 0-1 range (typical roughness: 0 to 2)
        normalized_roughness = min(roughness / 2.0, 1.0)

        losses.append(
            torch.tensor(normalized_roughness, device=device, dtype=torch.float32)
        )

    return torch.stack(losses)


def _plomp_levelt_roughness(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Calculate perceptual roughness using Plomp-Levelt model."""
    n = len(freqs)
    if n < 2:
        return 0.0

    total_roughness = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = min(freqs[i], freqs[j]), max(freqs[i], freqs[j])
            a1, a2 = amps[i], amps[j]

            fc = (f1 + f2) / 2
            cb = 25 + 75 * (1 + 1.4 * (fc / 1000) ** 2) ** 0.69
            s = abs(f2 - f1) / cb

            if s < 1.2:
                d = np.exp(-3.5 * s) - np.exp(-5.75 * s)
                d = max(0, d)
            else:
                d = 0.0

            roughness = d * a1 * a2
            total_roughness += roughness

    return total_roughness
