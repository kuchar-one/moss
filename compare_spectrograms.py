import torch
import torchaudio
import matplotlib.pyplot as plt

# Load Sample 1 from linear_mix
wav_path = "data/output/linear_mix/sample_1_idx18_mix24.wav"

audio, sr = torchaudio.load(wav_path)
# Standard STFT
n_fft = 2048
hop = 512
win = torch.hann_window(n_fft)
stft = torch.stft(audio, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
mag = stft.abs()
# Log Mag (dB)
mag_db = 20 * torch.log10(mag + 1e-8)

print(f"Mag Range: Min {mag_db.min()}, Max {mag_db.max()}")

# Plot
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# 1. Autoscaled (What Dashboard Uses - roughly)
im1 = axs[0].imshow(mag_db[0].numpy(), aspect="auto", origin="lower", cmap="magma")
axs[0].set_title(f"Autoscaled (Min {mag_db.min():.1f} dB)")
plt.colorbar(im1, ax=axs[0])

# 2. Fixed Range (Standard Viewer: -80dB to 0dB)
# Assume Max is 0dB (normalized)
mag_norm = mag_db - mag_db.max()
im2 = axs[1].imshow(
    mag_norm[0].numpy(), aspect="auto", origin="lower", cmap="magma", vmin=-80, vmax=0
)
axs[1].set_title("Standard Viewer (-80dB to 0dB)")
plt.colorbar(im2, ax=axs[1])

plt.savefig("spectrogram_comparison.png")
print("Saved spectrogram_comparison.png")
