"""Mask Optimization Problem definition (PyMoo)."""

import numpy as np
import torch
from pymoo.core.problem import Problem

from src import config
from src.audio_encoder import MaskEncoder
from src.objectives import calc_image_loss, calc_audio_mag_loss
from src.audio_utils import preprocess_image


class MaskOptimizationProblem(Problem):
    def __init__(
        self,
        target_image_path: str,
        target_audio_path: str,
        grid_height: int = 128,
        grid_width: int = 256,
        smoothing_sigma: float = 3.0,
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
            smoothing_sigma=smoothing_sigma,
            device=self.device,
        ).to(self.device)

        n_var = self.encoder.n_params

        super().__init__(
            n_var=n_var, n_obj=2, n_constr=0, xl=np.zeros(n_var), xu=np.ones(n_var)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # x is (pop_size, n_var) in range [0, 1]
        n_samples = x.shape[0]

        # Convert to Tensor
        params = torch.tensor(x, dtype=torch.float32, device=self.device)

        # Process in chunks to save memory
        chunk_size = 16  # Increased batch size
        f1_list = []
        f2_list = []

        with torch.no_grad():
            for i in range(0, n_samples, chunk_size):
                chunk_params = params[i : i + chunk_size]

                audio, mixed_mag = self.encoder(chunk_params)

                # Calculate losses for chunk
                chunk_f1 = calc_image_loss(
                    mixed_mag, self.encoder.image_mag_ref
                )  # Corrected ref name
                chunk_f2 = calc_audio_mag_loss(mixed_mag, self.encoder.audio_mag)

                f1_list.append(chunk_f1.cpu().numpy())
                f2_list.append(chunk_f2.cpu().numpy())

        f1 = np.concatenate(f1_list)
        f2 = np.concatenate(f2_list)

        out["F"] = np.column_stack([f1, f2])
