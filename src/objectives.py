"""Objectives for Image-Sound Encoding MOO.

Two objectives:
1. Image SSIM: How well the spectrogram matches the target image
2. Sound Similarity: Multi-scale spectral loss to target audio
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def calc_image_loss(
    generated_spec: torch.Tensor, target_image: torch.Tensor
) -> torch.Tensor:
    """Calculate image similarity loss using SSIM.

    Args:
        generated_spec: Generated spectrogram (batch, H, W) in [0, 1]
        target_image: Target image (H, W) or (1, H, W) in [0, 1]

    Returns:
        loss: (batch,) tensor where 0 = perfect match, 1 = no match
    """
    # Ensure 4D for SSIM: (batch, channel, H, W)
    if generated_spec.dim() == 3:
        gen = generated_spec.unsqueeze(1)
    else:
        gen = generated_spec

    target = target_image.squeeze()
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Expand target to batch size
    batch_size = gen.shape[0]
    target = target.expand(batch_size, -1, -1, -1)

    # Compute SSIM (returns similarity, so 1 - ssim = loss)
    ssim_values = []
    for i in range(batch_size):
        s = ssim(gen[i : i + 1], target[i : i + 1], data_range=1.0, size_average=True)
        ssim_values.append(1.0 - s)

    return torch.stack(ssim_values)


def calc_sound_loss(
    generated_audio: torch.Tensor,
    target_spec: torch.Tensor,
    generated_spec: torch.Tensor = None,
) -> torch.Tensor:
    """Calculate sound similarity using multi-scale spectral loss.

    Compares spectrograms at multiple resolutions for perceptual similarity.

    Args:
        generated_audio: Generated waveform (batch, samples) or None for spec-only
        target_spec: Target audio mel spectrogram (1, H, W) or (H, W)
        generated_spec: Pre-computed generated spectrogram

    Returns:
        loss: (batch,) tensor where 0 = perfect match
    """
    if generated_spec is None:
        raise ValueError("generated_spec must be provided for sound loss")

    batch_size = generated_spec.shape[0]
    device = generated_spec.device

    target = target_spec.squeeze()
    if target.dim() == 2:
        target = target.unsqueeze(0)  # (1, H, W)

    # Multi-scale comparison
    scales = [1.0, 0.5, 0.25]
    total_loss = torch.zeros(batch_size, device=device)

    for scale in scales:
        if scale < 1.0:
            # Downsample both specs
            h = int(target.shape[-2] * scale)
            w = int(target.shape[-1] * scale)

            t_scaled = F.interpolate(
                target.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False
            ).squeeze(0)

            g_scaled = F.interpolate(
                generated_spec.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        else:
            # Ensure same size
            t_scaled = target
            if generated_spec.shape[-1] != target.shape[-1]:
                g_scaled = F.interpolate(
                    generated_spec.unsqueeze(1),
                    size=(target.shape[-2], target.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                g_scaled = generated_spec

        # L1 loss at this scale
        t_expanded = t_scaled.expand(batch_size, -1, -1)
        scale_loss = (g_scaled - t_expanded).abs().mean(dim=(1, 2))
        total_loss += scale_loss

    # Average across scales
    return total_loss / len(scales)


def calc_spectral_flatness(spec: torch.Tensor) -> torch.Tensor:
    """Spectral flatness as a proxy for tonality.

    Lower flatness = more tonal (musical)
    Higher flatness = more noise-like
    """
    eps = 1e-8
    spec = spec.clamp(min=eps)

    # Geometric mean / arithmetic mean
    log_spec = spec.log()
    geo_mean = log_spec.mean(dim=-2).exp()
    arith_mean = spec.mean(dim=-2)

    flatness = geo_mean / (arith_mean + eps)
    return flatness.mean(dim=-1)
