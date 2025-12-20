"""Global configuration constants for MOSS."""

import torch

# Audio configuration
SAMPLE_RATE = 22050  # Lower SR to focus image in audible range (0-11kHz)
DURATION = 30.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# STFT configuration
N_FFT = 2048  # High frequency resolution
HOP_LENGTH = 512  # Overlap
WIN_LENGTH = N_FFT

# Image/Spectrogram configuration
IMG_HEIGHT = 128  # Mel frequency bins (Y-axis)
IMG_WIDTH = int((NUM_SAMPLES / HOP_LENGTH) + 1)  # Time steps (X-axis)

# Mel spectrogram config
N_MELS = 128
F_MIN = 20.0
F_MAX = 16000.0

# Genetic algorithm configuration
POP_SIZE = 100
N_GEN = 500

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Synth parameter ranges - these will be set by the synth module
SYNTH_PARAMS = {}
N_PARAMS = 0
