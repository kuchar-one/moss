import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from src.audio_utils import preprocess_image
import torchaudio.transforms as T


def verify_post_mix(wav_path, target_image_path):
    print(f"Verifying Post-Mix Gain: {wav_path}")

    # 1. Load Output WAV
    waveform, sr = torchaudio.load(wav_path)
    # Spectrogram
    n_fft = 2048
    hop = 512
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop,
        window=torch.hann_window(n_fft),
        return_complex=True,
    )
    mag = spec.abs() + 1e-8
    log_spec = torch.log(mag)

    # 2. Load Target Image
    img_tensor = preprocess_image(target_image_path)
    # Resize to match spectrogram
    img_resized = torch.nn.functional.interpolate(
        img_tensor.unsqueeze(0),
        size=(mag.shape[1], mag.shape[2]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    img_01 = img_resized.squeeze(0)
    img_01 = (img_01 - img_01.min()) / (img_01.max() - img_01.min() + 1e-8)

    # 3. Correlation?
    # Check if bright areas in image correspond to loud areas in spectrogram

    # 4. Histogram Comparison
    plt.figure(figsize=(10, 6))
    plt.hist(
        log_spec.flatten().numpy(),
        bins=100,
        alpha=0.5,
        label="Output Spectrogram",
        density=True,
    )

    # We don't know the exact mapping the encoder used, but we can see if the "Image Structure" exists.
    # If the output spectrogram is just "Audio", it will look like the original audio.
    # If it has the image, it should have a 'floor' or structure related to the image.

    plt.title("Output Spectrogram Histogram")
    plt.legend()
    plt.savefig("post_mix_analysis.png")
    print("Saved post_mix_analysis.png")

    # 5. Visual Check
    # Save the spectrogram itself
    plt.figure(figsize=(10, 5))
    to_db = T.AmplitudeToDB(stype="magnitude", top_db=None)
    db_spec = to_db(mag).squeeze(0).numpy()

    # Robust Scaling for visualization
    vmax = np.percentile(db_spec, 99.5)
    vmin = vmax - 80

    plt.imshow(
        db_spec, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax
    )
    plt.title(f"Output Spectrogram (Robust Scaling) Max={db_spec.max():.2f}")
    plt.savefig("post_mix_spectrogram.png")
    print("Saved post_mix_spectrogram.png")


if __name__ == "__main__":
    verify_post_mix(
        "data/output/test_quick_vis/sample_0_idx0_mix0.wav", "data/input/monalisa.jpg"
    )
