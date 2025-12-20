"""Pymoo problem wrapper connecting GPU synth to optimization.

PERFORMANCE: Uses synth.forward_spec_only() during optimization (no Griffin-Lim).
             Audio is only generated for final solutions in run_optimization.py.
"""

import numpy as np
import torch
from pymoo.core.problem import ElementwiseProblem, Problem

from . import config
from .synth import ImageBlendSynth
from .audio_utils import preprocess_image
from .objectives import calc_visual_loss, calc_musical_loss


class SpectralOptimization(ElementwiseProblem):
    """Multi-objective optimization for spectral synthesis.

    FAST: Uses forward_spec_only() - no Griffin-Lim during optimization.
    Visual loss computed on blended spectrogram directly.
    """

    def __init__(self, target_image_path: str, visual_only: bool = False):
        self.visual_only = visual_only
        self.device = config.DEVICE

        self.target_image = preprocess_image(target_image_path).to(self.device)
        self.synth = ImageBlendSynth(self.target_image).to(self.device)
        self.target_normalized = self.synth.target_image.unsqueeze(0)

        n_var = self.synth.n_params
        n_obj = 1 if visual_only else 2

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )

        print(f"Initialized SpectralOptimization:")
        print(f"  - Device: {self.device}")
        print(f"  - Parameters: {n_var}")
        print(f"  - Objectives: {n_obj}")
        print(f"  - Target image shape: {self.target_normalized.shape}")

    def _evaluate(self, x, out, *args, **kwargs):
        params = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            # FAST: Only compute blended spec (no Griffin-Lim)
            blended_spec = self.synth.forward_spec_only(params)

        # Visual loss on blended spec
        visual_loss = calc_visual_loss(blended_spec, self.target_normalized)
        f1 = visual_loss[0].cpu().numpy()

        if self.visual_only:
            out["F"] = [f1]
        else:
            # Musical loss: use spectral properties of blended spec as proxy
            # (No audio generation during optimization)
            musical_loss = self._estimate_musical_loss(blended_spec)
            f2 = musical_loss[0].cpu().numpy()
            out["F"] = [f1, f2]

    def _estimate_musical_loss(self, spec):
        """Estimate musicality from spectrogram (proxy for audio-based loss)."""
        # Spectral flatness: lower = more tonal (musical)
        # Higher flatness = more noise-like
        geo_mean = torch.exp(spec.mean(dim=1).log().mean(dim=1))
        arith_mean = spec.mean(dim=(1, 2))
        flatness = geo_mean / (arith_mean + 1e-8)

        # Harmonic structure: look for horizontal lines (good for music)
        # High variance along time axis = bad, low variance = good
        time_variance = spec.var(dim=2).mean(dim=1)

        # Combine: lower is better for both
        musical_loss = flatness * 0.5 + time_variance * 0.5
        return musical_loss.clamp(0, 1)


class BatchSpectralOptimization(Problem):
    """Batch-evaluated version for better GPU utilization."""

    def __init__(self, target_image_path: str, visual_only: bool = False):
        self.visual_only = visual_only
        self.device = config.DEVICE

        self.target_image = preprocess_image(target_image_path).to(self.device)
        self.synth = ImageBlendSynth(self.target_image).to(self.device)
        self.target_normalized = self.synth.target_image.unsqueeze(0)

        n_var = self.synth.n_params
        n_obj = 1 if visual_only else 2

        super().__init__(
            n_var=n_var,
            n_obj=n_obj,
            n_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )

        print(f"Initialized BatchSpectralOptimization:")
        print(f"  - Device: {self.device}")
        print(f"  - Parameters: {n_var}")
        print(f"  - Target image shape: {self.target_normalized.shape}")

    def _evaluate(self, X, out, *args, **kwargs):
        params = torch.tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # FAST: Only compute blended specs
            blended_specs = self.synth.forward_spec_only(params)

        # Visual loss
        visual_losses = calc_visual_loss(blended_specs, self.target_normalized)
        f1 = visual_losses.cpu().numpy()

        if self.visual_only:
            out["F"] = f1.reshape(-1, 1)
        else:
            # Musical loss proxy
            musical_losses = self._estimate_musical_loss(blended_specs)
            f2 = musical_losses.cpu().numpy()
            out["F"] = np.column_stack([f1, f2])

    def _estimate_musical_loss(self, spec):
        """Estimate musicality from spectrogram."""
        geo_mean = torch.exp(spec.mean(dim=1).log().mean(dim=1))
        arith_mean = spec.mean(dim=(1, 2))
        flatness = geo_mean / (arith_mean + 1e-8)
        time_variance = spec.var(dim=2).mean(dim=1)
        musical_loss = flatness * 0.5 + time_variance * 0.5
        return musical_loss.clamp(0, 1)
