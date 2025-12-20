"""Audio-to-spectrogram conversion and image preprocessing utilities."""

import torch
import torchaudio.transforms as T
from PIL import Image
import numpy as np

from . import config


def audio_to_spectrogram(audio: torch.Tensor) -> torch.Tensor:
    """Convert audio waveform to mel spectrogram image.

    Output spectrogram has low frequencies at index 0 (bottom when displayed with origin='lower').

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
    # f_min to f_max covers the full audible range
    mel_transform = T.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.IMG_HEIGHT,
        power=2.0,
        f_min=20.0,
        f_max=16000.0,  # Up to 16kHz for full spectrum
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

    The image bottom should correspond to low frequencies, top to high frequencies.
    Most images have (0,0) at top-left, so we flip to match spectrogram convention.

    Args:
        path: Path to the target image file

    Returns:
        Tensor of shape (1, height, width) normalized to [0, 1]
    """
    img = Image.open(path).convert("L")
    img = img.resize((config.IMG_WIDTH, config.IMG_HEIGHT), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0

    # NO flip - the image as-is will be matched
    # Top of image = high frequency, bottom = low frequency
    # This is intuitive: bright areas at top = high pitched sounds

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
    # NO flip - display directly
    img = (img * 255).astype(np.uint8)

    return img
