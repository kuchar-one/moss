import torch
import numpy as np
import time
from pathlib import Path
import uuid
import logging
from typing import Optional, Dict, Any, Tuple

from src import config
from src.audio_encoder import MaskEncoder
from src.audio_utils import preprocess_image
from src.gradient_optimizer import ParetoManager

# Configure Logging
logger = logging.getLogger(__name__)


class MossService:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.output_dir = data_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def start_optimization(
        self,
        image_path: str,
        audio_path: str,
        mode: str = "single",
        weights: Optional[Tuple[float, float]] = None,
        seed_mask: Optional[np.ndarray] = None,
    ) -> str:
        """
        Starts an optimization task in the background (or synchronously for now).
        TODO: Make this truly async/background.
        """
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "created_at": time.time(),
            "mode": mode,
            "params": {
                "image_path": image_path,
                "audio_path": audio_path,
                "weights": weights,
                "seed_mask_provided": seed_mask is not None,
            },
        }

        # For prototype, run synchronously (blocking) but return ID.
        # In production, use BackgroundTasks or Celery.
        try:
            self._run_optimization(
                task_id, image_path, audio_path, mode, weights, seed_mask
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)

        return task_id

    def _run_optimization(
        self,
        task_id: str,
        image_path: str,
        audio_path: str,
        mode: str,
        weights: Optional[Tuple[float, float]],
        seed_mask: Optional[np.ndarray],
    ):
        logger.info(f"Task {task_id}: Starting optimization (Mode: {mode})")
        self.tasks[task_id]["status"] = "running"

        # Device Setup
        device = config.DEVICE

        # 1. Load Resources
        target_image = preprocess_image(image_path).to(device)

        # Audio Grid Calculation
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        duration_sec = waveform.shape[-1] / sr
        raw_width = int(duration_sec * 8.5333)
        grid_width = ((raw_width + 15) // 16) * 16
        if grid_width < 16:
            grid_width = 16
        grid_height = 128

        # 2. Initialize Encoder & Optimizer
        sigma = 5.0  # Default
        encoder = MaskEncoder(
            target_image,
            audio_path,
            grid_height=grid_height,
            grid_width=grid_width,
            smoothing_sigma=sigma,
            device=device,
        ).to(device)

        # Apply Seeding
        if seed_mask is not None:
            logger.info(f"Task {task_id}: Applying seed mask.")
            # Seed is in [0, 1]. Convert to logits.
            # Encoder expects (1, H, W) or (pop, H, W)?
            # Encoder.mask_logits is nn.Parameter of shape (1, H, W) usually optimized.
            # But ParetoManager initializes population.
            pass

        # Optimization Parameters
        steps = 300
        # If Single mode, pop_size=1?
        # Actually gradient optimizer (ParetoManager) is designed for population.
        # But we can use it with weights.

        pop_size = 50 if mode == "pareto" else 1
        lr = 0.05

        manager = ParetoManager(encoder, pop_size=pop_size, learning_rate=lr)

        # Seeding Implementation for ParetoManager
        if seed_mask is not None:
            # seed_mask shape: (GridH * GridW) or (GridH, GridW)
            # manager.mask_logits shape: (Pop, GridH * GridW)

            # Ensure seed is flat
            seed_flat = seed_mask.flatten()

            # Clip to safely logit
            seed_clamped = np.clip(seed_flat, 1e-4, 1 - 1e-4)
            seed_logits = np.log(seed_clamped / (1 - seed_clamped))
            seed_tensor = torch.tensor(seed_logits, device=device, dtype=torch.float32)

            # Set for ALL population or just one?
            # If Single Point: Pop=1. Set it.
            # If Pareto: We might want to seed the whole population (perturbation?)
            # or just one anchor?
            # For now, let's seed the WHOLE population with the same starting point
            # layout, assuming differentiation will happen via weights.

            with torch.no_grad():
                manager.mask_logits.data[:] = seed_tensor.unsqueeze(0).expand(
                    pop_size, -1
                )

            logger.info("Seeding applied to manager.")

        # Normalization
        manager.calculate_normalization(encoder.image_mag_ref, encoder.audio_mag)

        # Steering Logic for Single Point
        if mode == "single" and weights:
            # weights is (w_vis, w_aud)
            # Manager expects weights_img logic.
            # In ParetoManager: self.weights_img[i] = 1.0 - (i / (pop_size - 1))
            # For single point, we manually override the weights tensor.
            w_img, w_aud = weights
            # Normalize to sum 1 just in case
            total = w_img + w_aud
            w_img /= total

            # Override internal weights
            manager.weights_img = torch.tensor(
                [w_img], device=device, dtype=torch.float32
            )
            manager.weights_aud = 1.0 - manager.weights_img

        # Optimization Loop
        for epoch in range(1, steps + 1):
            loss_vis, loss_aud = manager.optimize_step(
                encoder.image_mag_ref, encoder.audio_mag, micro_batch_size=8
            )

            if epoch % 10 == 0:
                self.tasks[task_id]["progress"] = epoch / steps

        # 3. Post-Processing & Saving
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Calculate Final Pareto Front / Point
        with torch.no_grad():
            masks_logits = manager.mask_logits
            X = torch.sigmoid(masks_logits).flatten(1).cpu().numpy()

            # Calculate F (Losses)
            # Batching logic if pop_size is large
            # reusing logic from run_optimization roughly
            mask = torch.sigmoid(masks_logits)
            _, mixed_mag = encoder(mask, return_wav=False)

            diff = torch.abs(mixed_mag - encoder.image_mag_ref)
            f_vis = diff.mean(dim=(1, 2)).cpu().numpy() * manager.scale_vis
            f_aud = (
                config.calc_audio_loss_fn(mixed_mag, encoder.audio_mag).cpu().numpy()
                * manager.scale_aud
            )

            F = np.column_stack([f_vis, f_aud])
            # Ensure scalar for single
            if mode == "single":
                f_vis = np.array([f_vis]) if np.isscalar(f_vis) else f_vis
                f_aud = np.array([f_aud]) if np.isscalar(f_aud) else f_aud

            F = np.column_stack([f_vis, f_aud])
            logger.info(
                f"Calculated Losses - Vis: {f_vis}, Aud: {f_aud}, F shape: {F.shape}"
            )

            # Save NPY
            np.save(task_dir / "X.npy", X)
            np.save(task_dir / "F.npy", F)

            # Save Spectrogram Image (High Res for Frontend)
            self._save_spectrogram(mixed_mag[0], task_dir / "spectrogram.png")

            # Save Audio (If single, save it. If pareto, maybe don't save all 50?)
            if mode == "single":
                rec_wav, _ = encoder(mask, return_wav=True)
                self._save_audio(rec_wav, task_dir / "output.wav")

        self.tasks[task_id]["status"] = "completed"
        self.tasks[task_id]["result_path"] = str(task_dir)
        # Store metrics: F is (N, 2). If single mode, takes first row.
        # If pareto, we might want to return all?
        # For now, let's just return the first one or mean if multiple
        # (Though task response schema expects List[float], suggesting single point)
        # If Pareto, we probably want to fetch F.npy separately.
        # But for 'single' stepping, F[0] is enough.
        if mode == "single":
            self.tasks[task_id]["result_metrics"] = F[0].tolist()
        else:
            # For pareto, maybe just store the mean or nothing?
            # Or invalid metrics?
            self.tasks[task_id]["result_metrics"] = F.mean(axis=0).tolist()

        logger.info(f"Task {task_id}: Completed.")

    def _save_spectrogram(self, mag, path):
        import matplotlib.pyplot as plt

        spec_db = 20 * torch.log10(mag + 1e-8).cpu().numpy()
        # Robust scaling
        ref_max = np.percentile(spec_db, 99.5)
        vmin = ref_max - 80
        vmax = ref_max

        plt.imsave(path, spec_db, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)

    def _save_audio(self, wav, path):
        import torchaudio

        wav_cpu = wav.squeeze().cpu()
        wav_cpu = wav_cpu / (torch.max(torch.abs(wav_cpu)) + 1e-6)
        torchaudio.save(path, wav_cpu.unsqueeze(0), config.SAMPLE_RATE)


# Global Instance
moss_service = MossService(data_dir=Path(__file__).resolve().parent.parent / "data")
