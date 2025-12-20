import torch
import torchaudio
import numpy as np
from src import config


def generate_ambient_drone(duration=30.0, sr=44100):
    t = torch.linspace(0, duration, int(sr * duration))

    # Fundamental frequencies for a nice ambient chord (A minor 9)
    freqs = [55, 110, 165, 220, 261.63, 329.63, 392.00]
    weights = [0.8, 0.6, 0.4, 0.3, 0.2, 0.2, 0.1]

    audio = torch.zeros_like(t)
    for f, w in zip(freqs, weights):
        # Add slight modulation
        mod = torch.sin(2 * np.pi * 0.1 * t) * 0.5 + 1.0
        audio += w * torch.sin(2 * np.pi * f * t) * mod

    # Add some noise
    noise = torch.randn_like(t) * 0.05
    audio += noise

    # Fade in/out
    fade_len = int(sr * 2.0)
    fade_in = torch.linspace(0, 1, fade_len)
    fade_out = torch.linspace(1, 0, fade_len)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out

    # Normalize
    audio = audio / audio.abs().max() * 0.9

    return audio.unsqueeze(0)


if __name__ == "__main__":
    import os

    os.makedirs("data/input", exist_ok=True)
    audio = generate_ambient_drone()
    torchaudio.save("data/input/target_ambient.wav", audio, 44100)
    print("Generated data/input/target_ambient.wav")
