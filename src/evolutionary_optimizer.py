import logging
import numpy as np
import torch
from pymoo.core.problem import Problem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from src.objectives import calc_audio_mag_loss

logger = logging.getLogger(__name__)


class MOSSProblem(Problem):
    """
    Vectorized Problem Definition for MOSS.
    Evaluates the entire population in a single PyTorch batch.
    """

    def __init__(self, encoder, scale_vis=1.0, scale_aud=1.0):
        # 128x256 = 32768 variables
        self.n_params = encoder.grid_height * encoder.grid_width
        self.encoder = encoder
        self.scale_vis = scale_vis
        self.scale_aud = scale_aud

        super().__init__(n_var=self.n_params, n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        # x is a 2D numpy array of shape (Pop, n_params) in range [0, 1]

        # 1. Convert to Batched Tensor
        with torch.no_grad():
            # (Pop, H*W) -> (Pop, 1, H, W)
            mask_flat = torch.from_numpy(x).float().to(self.encoder.device)
            # Ensure input to encoder matches MaskProcessor expectation
            # MaskProcessor expects flattened or it reshapes internally.
            # But we can pass it as is.

            # 2. Batched Forward Pass
            # Encoder processes all individuals efficiently in parallel (OpenMP/SIMD)
            # Returns mixed_mag of shape (Pop, H, W) - NO Channel dim
            _, mixed_mag = self.encoder(mask_flat, return_wav=False)

            # 3. Calculate Losses (Batched)

            # Visual Loss: Mean Abs Diff
            # mixed_mag: (Pop, H, W)
            # image_mag_ref: (1, H, W)
            diff = torch.abs(mixed_mag - self.encoder.image_mag_ref)
            loss_vis = diff.mean(dim=(1, 2)).cpu().numpy() * self.scale_vis  # (Pop,)

            # Audio Loss
            # Audio Loss: Log L1 (Matches Gradient Optimizer & Service Reporting)
            # aud_diff = torch.abs(mixed_mag - self.encoder.audio_mag)
            # loss_aud = aud_diff.mean(dim=(1, 2)).cpu().numpy() * self.scale_aud
            loss_aud_tensor = calc_audio_mag_loss(mixed_mag, self.encoder.audio_mag)
            loss_aud = loss_aud_tensor.cpu().numpy() * self.scale_aud

            # 4. Construct Output
            F = np.column_stack([loss_vis, loss_aud])
            out["F"] = F

            # 5. Track History - REMOVED
            # We track history in Callback to capture SURVIVORS, not just EVALUATIONS.
            # self.history.append(F.copy())


class ProgressCallback(Callback):
    def __init__(self, n_gen, report_progress_fn):
        super().__init__()
        self.n_gen = n_gen
        self.report_progress_fn = report_progress_fn
        self.history = []

    def notify(self, algorithm):
        # Current generation is algorithm.n_gen
        # Progress 0.0 -> 1.0 relative to evolutionary phase
        progress = algorithm.n_gen / self.n_gen
        logger.info(
            f"Evolutionary Progress: Gen {algorithm.n_gen}/{self.n_gen} -> {progress:.2f}"
        )
        self.report_progress_fn(progress)

        # Log History of Survivors
        # algorithm.pop is the population AFTER survival (the current front)
        F = algorithm.pop.get("F")
        self.history.append(F)

    def get_history(self):
        return self.history


def run_evolutionary_optimization(
    seed_masks: np.ndarray,
    encoder,
    scale_vis: float,
    scale_aud: float,
    pop_size: int = 100,
    n_gen: int = 20,
    progress_callback=None,
):
    """
    Runs NSGA-II initialized with seed_masks using Vectorized Evaluation.
    """

    problem = MOSSProblem(encoder, scale_vis, scale_aud)

    n_seeds = len(seed_masks)
    if n_seeds > pop_size:
        X_init = seed_masks[:pop_size]
    else:
        # Fill with random
        n_missing = pop_size - n_seeds
        X_rand = np.random.rand(n_missing, problem.n_params).astype(np.float32)
        X_init = np.vstack([seed_masks, X_rand])

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=X_init,
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    # Setup Callback
    callback = None
    if progress_callback:
        callback = ProgressCallback(n_gen, progress_callback)

    # We can turn off annoying verbose or keep it
    res = minimize(
        problem, algorithm, ("n_gen", n_gen), verbose=True, seed=42, callback=callback
    )

    return res.X, res.F, callback.get_history()
