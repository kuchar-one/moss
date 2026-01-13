import os
import torchaudio
import matplotlib.pyplot as plt


def verify_audio(audio_path, image_path):
    print(f"Verifying Audio: {audio_path}")

    # Load Audio
    # torchaudio.load normalizes by default for wav/mp3, but flac is lossless.
    # It returns [Channels, Time] in float32 [-1, 1] range.
    waveform, sample_rate = torchaudio.load(audio_path)

    # 1. Check Peak Clipping (Time Domain)
    peak_val = waveform.abs().max().item()
    clipped_percentage = (waveform.abs() >= 0.999).float().mean().item() * 100

    print(f"  > Peak Amplitude: {peak_val:.4f} (Max 1.0)")
    if peak_val > 1.0:
        print("  > WARNING: Audio clips above 1.0!")
    else:
        print("  > OK: Peak within range.")

    print(f"  > Clipped Samples (>= 0.999): {clipped_percentage:.4f}%")

    # 2. Check Spectrogram Saturation (Frequency Domain)
    n_fft = 2048
    hop = 512
    # Use standard Spectrogram
    transformer = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop, power=None
    )
    spectrogram_c = transformer(waveform)
    spectrogram_mag = spectrogram_c.abs()

    # Convert to dB
    to_db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
    spectrogram_db = to_db(spectrogram_mag)

    # Analysis
    # "White Saturation" typically means a large percentage of pixels are near 0dB (Max).
    # Since we set top_db=80, the range is [-80, 0].
    # Let's count pixels > -3dB (very bright).
    bright_pixels = (spectrogram_db > -3.0).float().mean().item() * 100
    median_loudness = spectrogram_db.median().item()

    print(f"  > Spectrogram Max dB: {spectrogram_db.max().item():.2f} (Should be 0.0)")
    print(f"  > Median Loudness: {median_loudness:.2f} dB")
    print(
        f"  > Bright Pixels (> -3dB): {bright_pixels:.2f}% (Should be low, only drums)"
    )

    # Save a verification plot
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db[0].numpy(), aspect="auto", origin="lower", cmap="magma")
    plt.colorbar()
    plt.title(f"Verification Spectrogram (Max Peak: {peak_val:.2f})")
    plt.savefig("verification_plot.png")
    print("  > Saved verification_plot.png")


if __name__ == "__main__":
    # Hardcoded check for the last known output
    path = "data/output/test_quick_vis/sample_0_idx0_mix0.wav"
    if os.path.exists(path):
        verify_audio(path, "")
    else:
        print("Test file not found:", path)
