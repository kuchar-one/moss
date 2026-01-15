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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resource Limiting: Use 100% of CPU cores for maximum performance
try:
    num_cores = os.cpu_count() or 1
    # Use all cores
    torch.set_num_threads(num_cores)
    # Also set OMP/MKL
    os.environ["OMP_NUM_THREADS"] = str(num_cores)
    print(f"MAX PERFORMANCE: Using {num_cores}/{num_cores} cores (100%).")
except Exception as e:
    print(f"RESOURCE LIMIT WARNING: Could not set thread limit: {e}")


# Image Magnitude Difference calculation
def calc_image_loss_fn(pred, target):
    from src.objectives import calc_image_loss

    return calc_image_loss(pred, target)


def calc_audio_loss_fn(pred, target):
    from src.objectives import calc_audio_mag_loss

    return calc_audio_mag_loss(pred, target)
