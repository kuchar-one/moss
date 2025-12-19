"""Pymoo problem wrapper connecting GPU synth to optimization."""

import numpy as np
import torch
from pymoo.core.problem import ElementwiseProblem

from . import config
from .synth import AmbientDrone
from .audio_utils import audio_to_spectrogram, preprocess_image
from .objectives import calc_visual_loss, calc_musical_loss


class SpectralOptimization(ElementwiseProblem):
    """Multi-objective optimization problem for spectral synthesis.

    Bridges the GPU-world (PyTorch) to the CPU-world (Pymoo).

    Objectives:
        - f1: Visual loss (1 - SSIM with target image)
        - f2: Musical loss (spectral roughness)

    Decision variables:
        - Normalized synth parameters (0-1 range)
    """

    def __init__(self, target_image_path: str, visual_only: bool = False):
        """Initialize the optimization problem.

        Args:
            target_image_path: Path to the target image file
            visual_only: If True, only optimize visual loss (Phase 2 testing)
        """
        self.visual_only = visual_only
        self.device = config.DEVICE

        # Initialize synthesizer
        self.synth = AmbientDrone(batch_size=1).to(self.device)

        # Load and preprocess target image
        self.target_image = preprocess_image(target_image_path).to(self.device)

        # Define problem dimensions
        n_var = self.synth.n_params
        n_obj = 1 if visual_only else 2

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=np.zeros(n_var),  # Lower bounds (all 0)
            xu=np.ones(n_var),  # Upper bounds (all 1)
        )

        print(f"Initialized SpectralOptimization:")
        print(f"  - Device: {self.device}")
        print(f"  - Parameters: {n_var}")
        print(f"  - Objectives: {n_obj}")
        print(f"  - Target image shape: {self.target_image.shape}")

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate a single solution.

        Args:
            x: Parameter vector of shape (n_var,)
            out: Output dictionary for objectives
        """
        # Convert to PyTorch tensor
        params = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Render audio
        with torch.no_grad():
            audio = self.synth(params)

        # Convert to spectrogram
        spec = audio_to_spectrogram(audio)

        # Calculate visual loss
        visual_loss = calc_visual_loss(spec, self.target_image)
        f1 = visual_loss[0].cpu().numpy()

        if self.visual_only:
            out["F"] = [f1]
        else:
            # Calculate musical loss
            musical_loss = calc_musical_loss(audio)
            f2 = musical_loss[0].cpu().numpy()
            out["F"] = [f1, f2]


class BatchSpectralOptimization(SpectralOptimization):
    """Batch-evaluated version for better GPU utilization.

    Evaluates entire population at once rather than element-wise.
    """

    def __init__(self, target_image_path: str, visual_only: bool = False):
        # Call grandparent init to avoid ElementwiseProblem restrictions
        self.visual_only = visual_only
        self.device = config.DEVICE

        self.synth = AmbientDrone(batch_size=config.POP_SIZE).to(self.device)
        self.target_image = preprocess_image(target_image_path).to(self.device)

        n_var = self.synth.n_params
        n_obj = 1 if visual_only else 2

        # Use Problem base class for batch evaluation
        from pymoo.core.problem import Problem

        Problem.__init__(
            self,
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )

        print(f"Initialized BatchSpectralOptimization:")
        print(f"  - Device: {self.device}")
        print(f"  - Parameters: {n_var}")
        print(f"  - Batch size: {config.POP_SIZE}")

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate entire population in batch.

        Args:
            X: Population matrix of shape (pop_size, n_var)
            out: Output dictionary for objectives
        """
        pop_size = X.shape[0]

        # Convert to PyTorch tensor
        params = torch.tensor(X, dtype=torch.float32, device=self.device)

        # Update synth batch size if needed
        if self.synth.batch_size != pop_size:
            self.synth = AmbientDrone(batch_size=pop_size).to(self.device)

        # Batch render audio
        with torch.no_grad():
            audio = self.synth(params)

        # Convert to spectrograms
        specs = audio_to_spectrogram(audio)

        # Calculate losses
        visual_losses = calc_visual_loss(specs, self.target_image)
        f1 = visual_losses.cpu().numpy()

        if self.visual_only:
            out["F"] = f1.reshape(-1, 1)
        else:
            musical_losses = calc_musical_loss(audio)
            f2 = musical_losses.cpu().numpy()
            out["F"] = np.column_stack([f1, f2])
