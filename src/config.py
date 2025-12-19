"""Global configuration constants for MOSS."""

import torch

# Audio configuration
SAMPLE_RATE = 44100
DURATION = 4.0  # seconds (start short for speed)
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

# STFT configuration
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = N_FFT

# Image/Spectrogram configuration
IMG_HEIGHT = 128  # Mel frequency bins (Y-axis)
IMG_WIDTH = int((NUM_SAMPLES / HOP_LENGTH) + 1)  # Time steps (X-axis), ~344 for 4s

# Genetic algorithm configuration
POP_SIZE = 100
N_GEN = 500

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Synth parameter ranges (all normalized 0-1)
# These define the boundaries for the optimization genome
SYNTH_PARAMS = {
    "vco1_freq": (20.0, 200.0),  # Sub-bass Hz
    "vco2_freq": (80.0, 800.0),  # Main melodic Hz
    "vco2_detune": (-50.0, 50.0),  # Cents
    "vco2_shape": (0.0, 1.0),  # 0=saw, 1=square blend
    "noise_level": (0.0, 0.5),  # White noise mix
    "lfo_rate": (0.05, 5.0),  # Hz
    "lfo_depth": (0.0, 100.0),  # Cents modulation
    "filter_cutoff": (200.0, 8000.0),  # Hz - most important for visual brightness
    "filter_q": (0.5, 10.0),  # Resonance
    "attack": (0.01, 2.0),  # ADSR seconds
    "decay": (0.1, 2.0),
    "sustain": (0.3, 1.0),
    "release": (0.1, 3.0),
    "vco1_level": (0.0, 1.0),
    "vco2_level": (0.0, 1.0),
}

N_PARAMS = len(SYNTH_PARAMS)
