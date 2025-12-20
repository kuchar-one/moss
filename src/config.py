"""Global configuration constants for MOSS."""

import torch

# Audio configuration
SAMPLE_RATE = 44100
DURATION = 60.0  # 60 seconds for finer temporal structures
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# STFT configuration - larger FFT for better frequency resolution
N_FFT = 4096
HOP_LENGTH = 1024
WIN_LENGTH = N_FFT

# Image/Spectrogram configuration
IMG_HEIGHT = 128  # Mel frequency bins (Y-axis)
IMG_WIDTH = int((NUM_SAMPLES / HOP_LENGTH) + 1)  # Time steps (X-axis)

# Genetic algorithm configuration
POP_SIZE = 100
N_GEN = 500

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Synth parameter ranges - these will be set by the synth module
SYNTH_PARAMS = {}
N_PARAMS = 0
