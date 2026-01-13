import torch
import torchaudio
import matplotlib.pyplot as plt
from src.audio_utils import preprocess_image


def analyze_gain(image_path, audio_path):
    print("Analyzing Gain Staging...")
    device = "cpu"

    # 1. Load Audio
    waveform, sr = torchaudio.load(audio_path)
    # Spectrogram logic from AudioEncoder
    n_fft = 2048
    hop = 512
    window = torch.hann_window(n_fft).to(device)
    stft = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop, window=window, return_complex=True
    )
    mag = stft.abs() + 1e-8
    audio_log = torch.log(mag)

    # 2. Logic from AudioEncoder (Current State)
    # Subsample for quantile calculation to avoid memory error
    # Take every 10th element from a flattened view
    audio_log_sub = audio_log.flatten()[::100]
    q98 = torch.quantile(audio_log_sub, 0.98)
    audio_max_real = audio_log.max()

    dynamic_range_nat = 10.0
    headroom_nat = 0.5  # Current setting

    audio_log_ceil = q98 - headroom_nat
    audio_log_floor = q98 - dynamic_range_nat

    print("Audio Statistics (Log Domain):")
    print(f"  Max: {audio_max_real:.2f}")
    print(f"  98th Percentile: {q98:.2f}")
    print(f"  Ceiling (Image White): {audio_log_ceil:.2f}")
    print(f"  Floor (Image Black): {audio_log_floor:.2f}")

    # 3. Load Image
    # 3. Load Image
    # preprocess_image only loads; we need to resize manually
    img_tensor = preprocess_image(image_path).to(device)

    # Resize to match spectrogram [1, F, T]
    img_resized = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0),
        size=(mag.shape[1], mag.shape[2]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    img_01 = img_resized.squeeze(0)
    # Normalize 0-1
    img_01 = (img_01 - img_01.min()) / (img_01.max() - img_01.min() + 1e-8)
    # Gamma
    img_gamma = img_01.pow(1.8)

    # Map
    image_log = img_gamma * (audio_log_ceil - audio_log_floor) + audio_log_floor

    # 4. Plot Histograms
    plt.figure(figsize=(10, 6))

    # Audio Histogram
    audio_flat = audio_log.numpy().flatten()
    plt.hist(
        audio_flat,
        bins=100,
        alpha=0.5,
        label="Audio Log Levels",
        density=True,
        color="blue",
        range=(-10, 10),
    )

    # Image Histogram
    image_flat = image_log.numpy().flatten()
    plt.hist(
        image_flat,
        bins=100,
        alpha=0.5,
        label="Image Log Levels (Mapped)",
        density=True,
        color="orange",
        range=(-10, 10),
    )

    plt.axvline(q98, color="blue", linestyle="--", label="Audio 98%")
    plt.axvline(
        audio_log_ceil, color="red", linestyle="--", label="Image White (Ceiling)"
    )

    plt.title("Gain Staging Analysis: Audio vs Mapped Image")
    plt.xlabel("Log Magnitude (Natural Log)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("gain_analysis.png")
    print("Saved gain_analysis.png")


if __name__ == "__main__":
    analyze_gain("data/input/monalisa.jpg", "data/input/06 - III Allegro assai.flac")
