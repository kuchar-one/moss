"""Audio-to-spectrogram conversion and image preprocessing utilities."""

import torch
import torchaudio.transforms as T
from PIL import Image
import numpy as np

from . import config


def audio_to_spectrogram(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio waveform to mel spectrogram image.

    Critical: Uses Mel-scale to compress high frequencies, giving
    "canvas space" in the musical mid-range where notes exist.

    Args:
        audio: Tensor of shape (batch, samples) or (samples,)

    Returns:
        Spectrogram tensor of shape (batch, height, width) normalized to [0, 1]
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    device = audio.device
    batch_size = audio.shape[0]

    # Create mel spectrogram transform
    mel_transform = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.IMG_HEIGHT,
        power=2.0,  # Power spectrogram
        f_min=20.0,  # Minimum frequency
        f_max=config.SAMPLE_RATE // 2,  # Nyquist
    ).to(device)

    # Compute mel spectrogram
    mel_spec = mel_transform(audio)  # (batch, n_mels, time)

    # Add small epsilon to avoid log(0)
    mel_spec = mel_spec + 1e-10

    # Convert to log scale (dB-like but normalized per-sample)
    mel_log = torch.log10(mel_spec)

    # Normalize each sample independently to [0, 1]
    # This ensures the full dynamic range is used
    batch_min = mel_log.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(2)
    batch_max = mel_log.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(2)
    mel_normalized = (mel_log - batch_min) / (batch_max - batch_min + 1e-8)

    # Resize to target dimensions if needed
    if mel_normalized.shape[-1] != config.IMG_WIDTH:
        mel_normalized = torch.nn.functional.interpolate(
            mel_normalized.unsqueeze(1),  # Add channel dim
            size=(config.IMG_HEIGHT, config.IMG_WIDTH),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    return mel_normalized


def preprocess_image(path: str) -> torch.Tensor:
    """Load and preprocess target image for spectrogram matching.

    Critical: Flips vertically because low frequencies are at the
    bottom of a spectrogram, but image index 0 is at the top.

    Args:
        path: Path to the target image file

    Returns:
        Tensor of shape (1, height, width) normalized to [0, 1]
    """
    # Load image as grayscale
    img = Image.open(path).convert("L")

    # Resize to match spectrogram dimensions
    img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.Resampling.LANCZOS)

    # Convert to numpy and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Flip vertically (low frequencies should be at bottom)
    img_array = np.flipud(img_array)

    # Convert to tensor
    tensor = torch.from_numpy(img_array.copy()).unsqueeze(0)  # (1, H, W)

    return tensor


def spectrogram_to_image(spec: torch.Tensor) -> np.ndarray:
    """Convert spectrogram tensor back to displayable image.

    Args:
        spec: Tensor of shape (height, width) or (1, height, width)

    Returns:
        Numpy array of shape (height, width) with values in [0, 255]
    """
    if spec.dim() == 3:
        spec = spec.squeeze(0)

    # Flip back for display (low freq at bottom)
    img = spec.cpu().numpy()
    img = np.flipud(img)

    # Scale to 0-255
    img = (img * 255).astype(np.uint8)

    return img
