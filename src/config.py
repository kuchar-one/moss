"""Global configuration constants for MOSS."""

import os
import torch

# Audio configuration
SAMPLE_RATE = 16000  # Lower SR for max performance
# DURATION is dynamic. NUM_SAMPLES/IMG_WIDTH derived at runtime.

# STFT configuration
N_FFT = 1024  # Lower freq resolution (513 bins)
HOP_LENGTH = 256  # Overlap
WIN_LENGTH = N_FFT

# Objective config
# Image/Spectrogram logic moved to Encoder

# Device configuration
DEVICE = "cpu"

# Resource Limiting: Use max 80% of CPU cores
try:
    num_cores = os.cpu_count() or 1
    limit_cores = max(1, int(num_cores * 0.8))
    torch.set_num_threads(limit_cores)
    # Also set OMP/MKL for good measure via os.environ (though torch often overrides)
    os.environ["OMP_NUM_THREADS"] = str(limit_cores)
    print(f"RESOURCE LIMIT: Restricted to {limit_cores}/{num_cores} cores (80%).")
except Exception as e:
    print(f"RESOURCE LIMIT WARNING: Could not set thread limit: {e}")


# Image Magnitude Difference calculation
def calc_image_loss_fn(pred, target):
    from src.objectives import calc_image_loss

    return calc_image_loss(pred, target)


def calc_audio_loss_fn(pred, target):
    from src.objectives import calc_audio_mag_loss

    return calc_audio_mag_loss(pred, target)
