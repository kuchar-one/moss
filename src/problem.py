"""MOO Problem for Mask-based Image-Sound Encoding."""

import numpy as np
import torch
from pymoo.core.problem import Problem

from . import config
from .audio_encoder import MaskEncoder
from .audio_utils import preprocess_image
from .objectives import calc_image_loss, calc_audio_mag_loss


class MaskOptimizationProblem(Problem):
    def __init__(
        self,
        target_image_path: str,
        target_audio_path: str,
        grid_height: int = 32,
        grid_width: int = 64,
    ):
        self.device = config.DEVICE

        # Load Raw Image
        target_image = preprocess_image(target_image_path)

        # Initialize Encoder
        self.encoder = MaskEncoder(
            target_image=target_image,
            target_audio_path=target_audio_path,
            grid_height=grid_height,
            grid_width=grid_width,
            device=self.device,
        ).to(self.device)

        n_var = self.encoder.n_params

        super().__init__(
            n_var=n_var, n_obj=2, n_constr=0, xl=np.zeros(n_var), xu=np.ones(n_var)
        )

        print(f"Initialized MaskOptimizationProblem:")
        print(f"  - Device: {self.device}")
        print(f"  - Grid: {grid_height}x{grid_width} = {n_var} params")

    def _evaluate(self, X, out, *args, **kwargs):
        params = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples = params.shape[0]

        # Process in chunks to save memory
        chunk_size = 4
        f1_list = []
        f2_list = []

        with torch.no_grad():
            for i in range(0, n_samples, chunk_size):
                chunk_params = params[i : i + chunk_size]

                audio, mixed_mag = self.encoder(chunk_params)

                # Calculate losses for chunk
                chunk_f1 = calc_image_loss(mixed_mag, self.encoder.image_mag_ref)
                chunk_f2 = calc_audio_mag_loss(mixed_mag, self.encoder.audio_mag)

                f1_list.append(chunk_f1.cpu().numpy())
                f2_list.append(chunk_f2.cpu().numpy())

        f1 = np.concatenate(f1_list)
        f2 = np.concatenate(f2_list)

        out["F"] = np.column_stack([f1, f2])
