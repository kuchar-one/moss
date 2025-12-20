"""Global configuration constants for MOSS."""

import torch

# Audio configuration
SAMPLE_RATE = 22050  # Lower SR to focus image in audible range (0-11kHz)
# DURATION is dynamic. NUM_SAMPLES/IMG_WIDTH derived at runtime.

# STFT configuration
N_FFT = 2048  # High frequency resolution
HOP_LENGTH = 512  # Overlap
WIN_LENGTH = N_FFT

# Objective config
# Image/Spectrogram logic moved to Encoder

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Image Magnitude Difference calculation
def calc_image_loss_fn(pred, target):
    from src.objectives import calc_image_loss

    return calc_image_loss(pred, target)


def calc_audio_loss_fn(pred, target):
    from src.objectives import calc_audio_mag_loss

    return calc_audio_mag_loss(pred, target)
