"""MOO Problem for Image-Sound Encoding.

PERFORMANCE: Compares spectrograms directly during optimization (no Griffin-Lim).
             Audio is only generated for final solutions.
"""

import numpy as np
import torch
from pymoo.core.problem import Problem

from . import config
from .audio_encoder import SpectrogramEncoder
from .audio_utils import preprocess_image
from .objectives import calc_image_loss, calc_sound_loss


class ImageSoundProblem(Problem):
    """Multi-objective optimization for image-sound encoding.

    FAST MODE: Compares upsampled spectrogram grids directly.
    No Griffin-Lim during optimization (audio only for final solutions).
    """

    def __init__(
        self,
        target_image_path: str,
        target_audio_path: str = None,
        grid_height: int = 32,
        grid_width: int = 64,
    ):
        self.device = config.DEVICE

        # Load target image
        target_image = preprocess_image(target_image_path).to(self.device)

        # Load target audio (or None)
        if target_audio_path:
            import torchaudio

            target_audio, sr = torchaudio.load(target_audio_path)
            if sr != config.SAMPLE_RATE:
                target_audio = torchaudio.functional.resample(
                    target_audio, sr, config.SAMPLE_RATE
                )
            target_audio = target_audio.mean(dim=0)
            if target_audio.shape[-1] < config.NUM_SAMPLES:
                target_audio = torch.nn.functional.pad(
                    target_audio, (0, config.NUM_SAMPLES - target_audio.shape[-1])
                )
            else:
                target_audio = target_audio[: config.NUM_SAMPLES]
            target_audio = target_audio.to(self.device)
        else:
            target_audio = None

        # Create encoder
        self.encoder = SpectrogramEncoder(
            target_image=target_image,
            target_audio=target_audio,
            grid_height=grid_height,
            grid_width=grid_width,
        ).to(self.device)

        # Compute target audio spectrogram for sound objective
        if target_audio is not None:
            self.target_audio_spec = self.encoder.get_audio_spectrogram(
                target_audio.unsqueeze(0)
            ).squeeze(0)
        else:
            # Use image as sound target (objectives will align)
            self.target_audio_spec = self.encoder.target_image

        self.grid_height = grid_height
        self.grid_width = grid_width
        n_var = self.encoder.n_params

        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )

        print(f"Initialized ImageSoundProblem:")
        print(f"  - Device: {self.device}")
        print(f"  - Grid: {grid_height}x{grid_width} = {n_var} params")
        print(f"  - Target image: {self.encoder.target_image.shape}")
        print(f"  - Has target audio: {target_audio is not None}")

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate batch - FAST (no audio generation)."""
        batch_size = X.shape[0]
        params = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # FAST: Only compute upsampled spectrogram (no Griffin-Lim)
            grid = params.view(batch_size, 1, self.grid_height, self.grid_width)
            full_spec = torch.nn.functional.interpolate(
                grid,
                size=(self.encoder.full_height, self.encoder.full_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)  # (batch, H, W)

        # Objective 1: Image SSIM
        image_loss = calc_image_loss(full_spec, self.encoder.target_image)

        # Objective 2: Sound loss (spec-to-spec comparison)
        sound_loss = calc_sound_loss(None, self.target_audio_spec, full_spec)

        f1 = image_loss.cpu().numpy()
        f2 = sound_loss.cpu().numpy()

        out["F"] = np.column_stack([f1, f2])
