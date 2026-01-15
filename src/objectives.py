"""Objectives for Mask-Based Optimization.

1. Image Loss: SSIM(Mixed_LogMag, Target_Image)
2. Audio Loss: L1(Mixed_LogMag, Target_Audio_LogMag)

Comparing in Log domain (dB) matches human perception (audio) and typical spectrogram visualization (visual).
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def calc_image_loss(
    mixed_mag: torch.Tensor, target_image_01: torch.Tensor
) -> torch.Tensor:
    """Calculate visual similarity in LOG Domain.

    Args:
        mixed_mag: (batch, F, T) Linear Magnitude spectrogram
        target_image_01: (1, F, T) Target image in [0, 1]

    Returns:
        loss (batch,)
    """
    # Convert Mixed to Log domain for visual comparison
    mixed_log = torch.log(mixed_mag + 1e-8)

    # Normalize Mixed Log for SSIM (dynamic range agnostic comparison)
    mixed_norm = _normalize_minmax(mixed_log)

    # Target is already [0, 1] (and represents "intensity" which visually maps to log mag)
    target_norm = target_image_01.expand_as(mixed_norm)

    # Add channel dim for SSIM: (B, 1, F, T)
    mixed_norm = mixed_norm.unsqueeze(1)
    target_norm = target_norm.unsqueeze(1)

    # Batch SSIM
    # data_range=1.0 since we normalized to [0,1]
    s = ssim(
        mixed_norm,
        target_norm,
        data_range=1.0,
        size_average=False,  # Return (B,) tensor
    )
    
    # ssim returns (B, 1, 1, 1) or (B,) depending on size_average=False?
    # pytorch_msssim with size_average=False returns (B,)
    
    return 1.0 - s


def calc_audio_mag_loss(
    mixed_mag: torch.Tensor, target_audio_mag: torch.Tensor
) -> torch.Tensor:
    """Calculate audio similarity based on LOG spectrogram magnitude.

    L1 in Log domain = per-bin dB difference.
    """
    mixed_log = torch.log(mixed_mag + 1e-8)
    target_log = torch.log(target_audio_mag + 1e-8)

    target_log = target_log.expand_as(mixed_log)

    loss = F.l1_loss(mixed_log, target_log, reduction="none").mean(dim=(1, 2))
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
