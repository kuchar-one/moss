"""Objective functions for multi-objective optimization.

Two competing objectives:
1. Visual Loss: How well does the spectrogram match the target image?
2. Musical Loss: How rough/dissonant does the audio sound?
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

    SSIM (Structural Similarity) is better than MSE because it tolerates
    slight frequency shifts that would ruin pixel-perfect matching.

    Args:
        generated_spec: Tensor of shape (batch, height, width)
        target_img: Tensor of shape (1, height, width) - the target image

    Returns:
        Loss tensor of shape (batch,) where lower is better (1 - SSIM)
    """
    batch_size = generated_spec.shape[0]
    device = generated_spec.device

    # Expand target to batch size
    target_expanded = target_img.expand(batch_size, -1, -1).to(device)

    # Add channel dimension for SSIM: (B, C, H, W)
    gen_4d = generated_spec.unsqueeze(1)
    tgt_4d = target_expanded.unsqueeze(1)

    # Compute SSIM (returns single value for batch)
    # We need per-sample SSIM, so compute individually
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
    """Calculate musical roughness/dissonance loss.

    Uses spectral roughness based on Plomp-Levelt dissonance curve:
    - FFT to find partials (frequency peaks)
    - Sum beat frequency dissonance between all pairs

    Includes silence penalty to prevent the "silence cheat" where
    the algorithm discovers silence has zero roughness.

    Args:
        audio: Tensor of shape (batch, samples)

    Returns:
        Loss tensor of shape (batch,) where lower is more musical
    """
    batch_size = audio.shape[0]
    device = audio.device
    losses = []

    for i in range(batch_size):
        wave = audio[i]

        # Silence penalty: if energy too low, force high loss
        energy = torch.mean(wave**2)
        if energy < 0.001:
            losses.append(torch.tensor(9999.0, device=device))
            continue

        # Compute FFT
        fft = torch.fft.rfft(wave)
        magnitude = torch.abs(fft)

        # Convert to numpy for peak finding
        mag_np = magnitude.cpu().numpy()

        # Find top peaks (partials)
        peaks, properties = find_peaks(mag_np, height=mag_np.max() * 0.1)

        if len(peaks) < 2:
            # Not enough peaks, low roughness
            losses.append(torch.tensor(0.1, device=device))
            continue

        # Get top 10 peaks by height
        peak_heights = mag_np[peaks]
        top_indices = np.argsort(peak_heights)[-10:]
        top_peaks = peaks[top_indices]

        # Convert bin indices to frequencies
        freqs = top_peaks * config.SAMPLE_RATE / config.NUM_SAMPLES
        amps = mag_np[top_peaks]
        amps = amps / amps.max()  # Normalize amplitudes

        # Calculate roughness using Plomp-Levelt curve
        roughness = _plomp_levelt_roughness(freqs, amps)

        losses.append(torch.tensor(roughness, device=device, dtype=torch.float32))

    return torch.stack(losses)


def _plomp_levelt_roughness(freqs: np.ndarray, amps: np.ndarray) -> float:
    """Calculate perceptual roughness using Plomp-Levelt model.

    The dissonance between two frequencies depends on their
    critical bandwidth separation. Maximum dissonance occurs
    at about 1/4 of the critical bandwidth apart.

    Args:
        freqs: Array of partial frequencies in Hz
        amps: Array of normalized amplitudes (0-1)

    Returns:
        Total roughness value (lower is smoother)
    """
    n = len(freqs)
    if n < 2:
        return 0.0

    total_roughness = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = min(freqs[i], freqs[j]), max(freqs[i], freqs[j])
            a1, a2 = amps[i], amps[j]

            # Critical bandwidth (Bark scale approximation)
            fc = (f1 + f2) / 2
            cb = 25 + 75 * (1 + 1.4 * (fc / 1000) ** 2) ** 0.69

            # Frequency difference relative to critical bandwidth
            s = abs(f2 - f1) / cb

            # Plomp-Levelt dissonance curve
            # Maximum at s ≈ 0.25, decreasing to 0 at s = 0 and s > 1.2
            if s < 1.2:
                d = np.exp(-3.5 * s) - np.exp(-5.75 * s)
                d = max(0, d)
            else:
                d = 0.0

            # Weight by amplitude product
            roughness = d * a1 * a2
            total_roughness += roughness

    return total_roughness


def calc_spectral_centroid_jerk(audio: torch.Tensor) -> torch.Tensor:
    """Calculate spectral centroid jerk (optional smoothness metric).

    The jerk (rate of change of acceleration) of the spectral centroid
    measures how "jumpy" the timbre is. Low jerk = smooth ambient drone.

    Args:
        audio: Tensor of shape (batch, samples)

    Returns:
        Jerk tensor of shape (batch,)
    """
    batch_size = audio.shape[0]
    device = audio.device
    jerks = []

    for i in range(batch_size):
        wave = audio[i]

        # Compute STFT
        stft = torch.stft(
            wave,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            return_complex=True,
        )
        magnitude = torch.abs(stft)  # (freq_bins, time_frames)

        # Compute spectral centroid per frame
        freq_bins = torch.arange(magnitude.shape[0], device=device, dtype=torch.float32)
        centroids = (magnitude * freq_bins.unsqueeze(1)).sum(dim=0) / (
            magnitude.sum(dim=0) + 1e-8
        )

        # Compute jerk (third derivative)
        if len(centroids) < 4:
            jerks.append(torch.tensor(0.0, device=device))
            continue

        velocity = centroids[1:] - centroids[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        jerk = acceleration[1:] - acceleration[:-1]

        jerk_magnitude = torch.mean(torch.abs(jerk))
        jerks.append(jerk_magnitude)

    return torch.stack(jerks)
