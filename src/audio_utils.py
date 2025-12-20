"""Audio-to-spectrogram conversion and image preprocessing utilities."""

import torch
import torchaudio.transforms as T
from PIL import Image
import numpy as np

from . import config


def audio_to_spectrogram(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio waveform to mel spectrogram image.

    Output shape: (batch, n_mels, time) where row 0 = lowest frequency.
    When displayed with origin='lower', low freq is at bottom.

    Args:
        audio: Tensor of shape (batch, samples) or (samples,)

    Returns:
        Spectrogram tensor of shape (batch, height, width) normalized to [0, 1]
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    device = audio.device
    batch_size = audio.shape[0]

    # Mel spectrogram: focuses on 500Hz-12kHz where detail can be perceived
    # Low frequencies (20-500Hz) provide little visual resolution
    mel_transform = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.IMG_HEIGHT,
        power=2.0,
        f_min=200.0,  # Start at 200Hz, not 20Hz (skip sub-bass)
        f_max=12000.0,  # Up to 12kHz
    ).to(device)

    mel_spec = mel_transform(audio)  # (batch, n_mels, time)
    mel_spec = mel_spec + 1e-10

    # Log scale
    mel_log = torch.log10(mel_spec)

    # Normalize per-sample to [0, 1]
    batch_min = mel_log.reshape(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(2)
    batch_max = mel_log.reshape(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(2)
    mel_normalized = (mel_log - batch_min) / (batch_max - batch_min + 1e-8)

    # Resize to target dimensions if needed
    if mel_normalized.shape[-1] != config.IMG_WIDTH:
        mel_normalized = torch.nn.functional.interpolate(
            mel_normalized.unsqueeze(1),
            size=(config.IMG_HEIGHT, config.IMG_WIDTH),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    return mel_normalized


def preprocess_image(path: str) -> torch.Tensor:
    """Load and preprocess target image for spectrogram matching.

    The image is flipped vertically so that:
    - TOP of the original image → HIGH frequencies (top of spectrogram display)
    - BOTTOM of the original image → LOW frequencies (bottom of spectrogram display)

    This makes images appear RIGHT-SIDE-UP when displayed with origin='lower'.

    Args:
        path: Path to the target image file

    Returns:
        Tensor of shape (1, height, width) normalized to [0, 1]
    """
    img = Image.open(path).convert("L")
    img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0

    # FLIP VERTICALLY: row 0 of image becomes high frequency (top of display)
    img_array = np.flipud(img_array)

    tensor = torch.from_numpy(img_array.copy()).unsqueeze(0)
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

    img = spec.cpu().numpy()
    img = (img * 255).astype(np.uint8)

    return img
