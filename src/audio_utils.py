"""Audio-to-spectrogram conversion and image preprocessing utilities."""

import torch
from PIL import Image
import numpy as np



def preprocess_image(path: str) -> torch.Tensor:
    """Load and preprocess target image for spectrogram matching.

    The image is flipped vertically so that:
    - TOP of the original image → HIGH frequencies (top of spectrogram display)
    - BOTTOM of the original image → LOW frequencies (bottom of spectrogram display)

    Args:
        path: Path to the target image file

    Returns:
        Tensor of shape (1, height, width) normalized to [0, 1]
    """
    img = Image.open(path).convert("L")

    # Resize to a reasonable standard resolution.
    # MaskEncoder will resize this to the exact STFT dimensions later.
    # We choose 1024x1024 to preserve detail before downsampling.
    img = img.resize((1024, 1024), Image.Resampling.LANCZOS)

    img_array = np.array(img, dtype=np.float32) / 255.0

    # FLIP VERTICALLY: row 0 of image becomes high frequency (top of display)
    img_array = np.flipud(img_array)

    tensor = torch.from_numpy(img_array.copy()).unsqueeze(0)
    return tensor


def spectrogram_to_image(spec: torch.Tensor) -> np.ndarray:
    """Convert spectrogram tensor back to displayable image."""
    if spec.dim() == 3:
        spec = spec.squeeze(0)

    img = spec.cpu().numpy()
    img = (img * 255).astype(np.uint8)

    return img
