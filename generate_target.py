import torch
import torchaudio
import numpy as np
from src import config


def generate_ambient_drone(duration=30.0, sr=config.SAMPLE_RATE):
    t = torch.linspace(0, duration, int(sr * duration))

    # Richer Chord (Gm9 add11) spanning low to mid frequencies
    # 55Hz (A1) up to ~800Hz, plus harmonics will extend higher
    freqs = [
        55.00,
        110.00,  # A1, A2 (Bass)
        165.00,
        220.00,  # E3, A3
        261.63,
        329.63,  # C4, E4
        392.00,
        440.00,  # G4, A4
        523.25,
        659.25,  # C5, E5
    ]
    weights = [1.0, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.4, 0.3, 0.3]

    audio = torch.zeros_like(t)

    for f, w in zip(freqs, weights):
        # Slow modulation
        mod_rate = np.random.uniform(0.05, 0.2)
        mod = torch.sin(2 * np.pi * mod_rate * t) * 0.3 + 0.7

        # Add Saw/Square distinct timbre to fill harmonics
        # Simple additive syntheisis approx
        tone = torch.sin(2 * np.pi * f * t)
        # Add 2nd and 3rd harmonics for richness
        tone += 0.5 * torch.sin(2 * np.pi * (2 * f) * t)
        tone += 0.25 * torch.sin(2 * np.pi * (3 * f) * t)

        audio += w * tone * mod

    # Pink Noise for texture (1/f) - helps fill gaps
    # Simple approx via cumsum of random? No, filtered noise.
    white = torch.randn_like(t)
    # Simple LPF for "warm" noise
    noise = torch.zeros_like(t)
    # Very inefficient implementation, but simple:
    # Use torchaudio functional if possible, or just white noise for now
    noise = white * 0.05

    audio += noise

    # Fade in/out
    fade_len = int(sr * 2.0)
    fade_in = torch.linspace(0, 1, fade_len)
    fade_out = torch.linspace(1, 0, fade_len)
    audio[:fade_len] *= fade_in
    audio[-fade_len:] *= fade_out

    # Normalize
    max_val = audio.abs().max()
    if max_val > 0:
        audio = audio / max_val * 0.9

    return audio.unsqueeze(0)


if __name__ == "__main__":
    import os

    os.makedirs("data/input", exist_ok=True)
    print(f"Generating target at {config.SAMPLE_RATE}Hz...")
    audio = generate_ambient_drone(duration=config.DURATION, sr=config.SAMPLE_RATE)
    torchaudio.save("data/input/target_ambient.wav", audio, config.SAMPLE_RATE)
    print("Generated data/input/target_ambient.wav")
