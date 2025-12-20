"""Objectives for Mask-Based Optimization.

1. Image Loss: SSIM(Mixed_Mag, Target_Image_Mag)
2. Audio Loss: MSE(Mixed_Mag, Target_Audio_Mag)
   (Since phase is fixed, magnitude difference is the primary audio degradation metric)
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def calc_image_loss(
    mixed_mag: torch.Tensor, target_image_mag: torch.Tensor
) -> torch.Tensor:
    """Calculate visual similarity.

    Args:
        mixed_mag: (batch, F, T) Magnitude spectrogram
        target_image_mag: (1, F, T) Scaled target image magnitude

    Returns:
        loss (batch,)
    """
    # SSIM requires inputs in [0, 1] range ideally.
    # Our magnitudes are arbitrary (0 to 100+).
    # Normalize per sample for visual comparison?

    mixed_norm = _normalize_minmax(mixed_mag)
    target_norm = _normalize_minmax(target_image_mag)

    # Expand target
    batch_size = mixed_mag.shape[0]
    target_norm = target_norm.expand(batch_size, -1, -1)

    # Add channel dim for SSIM: (B, 1, F, T)
    mixed_norm = mixed_norm.unsqueeze(1)
    target_norm = target_norm.unsqueeze(1)

    ssim_vals = []
    for i in range(batch_size):
        s = ssim(
            mixed_norm[i : i + 1],
            target_norm[i : i + 1],
            data_range=1.0,
            size_average=True,
        )
        ssim_vals.append(1.0 - s)

    return torch.stack(ssim_vals)


def calc_audio_mag_loss(
    mixed_mag: torch.Tensor, target_audio_mag: torch.Tensor
) -> torch.Tensor:
    """Calculate audio similarity based on spectrogram magnitude.

    MSE or L1 distance effectively captures how modified the spectrum is.
    """
    target = target_audio_mag.expand_as(mixed_mag)

    # L1 Loss is often better for spectral magnitude
    loss = F.l1_loss(mixed_mag, target, reduction="none").mean(dim=(1, 2))
    return loss


def _normalize_minmax(x):
    # Normalize to [0, 1] per sample
    # x shape (..., F, T)
    flat = x.flatten(start_dim=-2)
    min_val = flat.min(dim=-1, keepdim=True)[0]
    max_val = flat.max(dim=-1, keepdim=True)[0]
    diff = (max_val - min_val).clamp(min=1e-8)

    min_val = min_val.unsqueeze(-1)
    diff = diff.unsqueeze(-1)

    return (x - min_val) / diff
