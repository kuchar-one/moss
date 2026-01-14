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
from src.evolutionary_optimizer import run_evolutionary_optimization

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
        raw_width = int(duration_sec * 4.0)  # Reduced from 8.5
        grid_width = ((raw_width + 15) // 16) * 16
        if grid_width < 16:
            grid_width = 16
        grid_height = 64  # Reduced from 128

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

        # Hybrid Config:
        # If Pareo:
        #   Pop Size = 10 (Gradient Seeds)
        #   LR = 0.05
        # If Single:
        #   Pop Size = 1
        #   LR = 0.05
        pop_size = 10 if mode == "pareto" else 1
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
        if mode == "single":
            # Optimization Loop (Single)
            # Increased from 100 to 500 since Proxy Optimization is so fast
            steps = 500
            history = []

            # Indicate start
            self.tasks[task_id]["progress"] = 0.01

            for epoch in range(1, steps + 1):
                loss_vis, loss_aud = manager.optimize_step(
                    encoder.image_mag_ref, encoder.audio_mag, micro_batch_size=1
                )
                # ...
                # Ensure scalar for logging
                v = float(loss_vis)
                a = float(loss_aud)
                history.append(np.array([[v, a]]))

                if epoch % 5 == 0:  # Update every 1% (5 steps)
                    self.tasks[task_id]["progress"] = epoch / steps

        elif mode == "pareto":
            # HYBRID STRATEGY:
            # 1. Gradient Seeds (Pop=10, Steps=100)
            logger.info("Starting Hybrid: Phase 1 - Gradient Seeds")
            grad_steps = 200  # Increased from 100 for denser seeding
            phase1_history = []

            for epoch in range(1, grad_steps + 1):
                # Vectorized Step: Process all 10 seeds at once!
                loss_vis, loss_aud = manager.optimize_step(
                    encoder.image_mag_ref, encoder.audio_mag, micro_batch_size=10
                )
                # loss_vis, loss_aud are sums or means?
                # ParetoManager.optimize_step returns weighted sum if Population?
                # Actually ParetoManager with population returns None?
                # Let's check src/gradient_optimizer.py.
                # Assuming it returns metrics.
                pass

                # We want the LOSS of EACH INDIVIDUAL to plot the cloud.
                # The manager has self.mask_logits.
                # We can do a quick check? No, too slow.
                # Let's assume optimize_step returns something useful or we skip detailed tracking here?
                # User wants "how seeding runs moved".
                # To do this effectively, we need to evaluate the population.
                # MaskEncoder is fast.

                # Let's evaluate the current population in the manager "lightly".
                if epoch % 5 == 0:  # Every 5 steps
                    with torch.no_grad():
                        curr_masks = torch.sigmoid(manager.mask_logits)
                        _, curr_mag = encoder(curr_masks, return_wav=False)

                        diff = torch.abs(curr_mag - encoder.image_mag_ref)
                        lv = diff.mean(dim=(1, 2)).cpu().numpy() * manager.scale_vis

                        # Use correct loss function matching Optimizer (Log L1)
                        # adiff = torch.abs(curr_mag - encoder.audio_mag)
                        # la = adiff.mean(dim=(1, 2)).cpu().numpy() * manager.scale_aud
                        la = (
                            config.calc_audio_loss_fn(curr_mag, encoder.audio_mag)
                            .cpu()
                            .numpy()
                            * manager.scale_aud
                        )

                        phase1_history.append(np.column_stack([lv, la]))

                if epoch % 10 == 0:
                    # Phase 1 is 50% progress
                    self.tasks[task_id]["progress"] = (epoch / grad_steps) * 0.5

            # 2. Extract Seeds
            with torch.no_grad():
                seed_masks = torch.sigmoid(manager.mask_logits).cpu().numpy()

            # 3. Evolutionary Expansion (Pop=100)
            # 3. Evolutionary Expansion (Pop=100)
            logger.info("Starting Hybrid: Phase 2 - Evolutionary Expansion")

            def report_progress(p):
                # Map 0.0-1.0 to 0.5-1.0
                self.tasks[task_id]["progress"] = 0.5 + (p * 0.5)

            final_X, final_F, evo_history = run_evolutionary_optimization(
                seed_masks,
                encoder,
                manager.scale_vis,
                manager.scale_aud,
                pop_size=100,
                n_gen=50,  # Reduced from 100 since seeding is now 200 steps
                progress_callback=report_progress,
            )

            # Combine History
            full_history = phase1_history + evo_history

        # 3. Post-Processing & Saving
        # 3. Post-Processing & Saving
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Generate Animation if history exists
        if mode == "single" and "history" in locals():
            self._generate_animation(
                history, task_dir / "history.mp4", title="Single Seed Optimization"
            )
        elif mode == "pareto" and "full_history" in locals():
            # Combine History if not already combined (it was combined above but local var scope?)
            # Actually full_history is defined in the block above if mode==pareto.
            self._generate_animation(
                full_history,
                task_dir / "history.mp4",
                title="Hybrid Pareto Optimization",
            )

        if mode == "single":
            # Calculate Final Single Point
            with torch.no_grad():
                masks_logits = manager.mask_logits
                X = torch.sigmoid(masks_logits).flatten(1).cpu().numpy()
                mask = torch.sigmoid(masks_logits)
                _, mixed_mag = encoder(mask, return_wav=False)

                diff = torch.abs(mixed_mag - encoder.image_mag_ref)
                f_vis = diff.mean(dim=(1, 2)).cpu().numpy() * manager.scale_vis
                f_aud = (
                    config.calc_audio_loss_fn(mixed_mag, encoder.audio_mag)
                    .cpu()
                    .numpy()
                    * manager.scale_aud
                )
                # Ensure scalar for single
                f_vis = np.array([f_vis]) if np.isscalar(f_vis) else f_vis
                f_aud = np.array([f_aud]) if np.isscalar(f_aud) else f_aud
                F = np.column_stack([f_vis, f_aud])

                # Save Single Output
                np.save(task_dir / "X.npy", X)
                np.save(task_dir / "F.npy", F)
                self._save_spectrogram(mixed_mag[0], task_dir / "spectrogram.png")
                rec_wav, _ = encoder(mask, return_wav=True)
                self._save_audio(rec_wav, task_dir / "output.wav")

                self.tasks[task_id]["result_metrics"] = F[0].tolist()

        elif mode == "pareto":
            # Save Evolutionary Results
            # final_X is (Pop, Params), final_F is (Pop, 2)
            np.save(task_dir / "X.npy", final_X)
            np.save(task_dir / "F.npy", final_F)

            # Save one representative spectrogram (e.g. median)
            # Find closest to 50/50 relative balance? Or just random.
            # Picking index 50 (middle) if sorted? NSGA2 doesn't sort by weights.
            # Just pick the first one for the thumbnail.
            # Or better, generates a 'morph' video later.
            # For now, save the LAST one (usually towards audio or image?).
            # Save index 0.

            # We need to construct the spectrogram for index 0 to save png
            with torch.no_grad():
                # Convert back to tensor mask for one example
                example_mask_np = final_X[0]
                example_mask = (
                    torch.from_numpy(example_mask_np)
                    .float()
                    .to(device)
                    .view(1, grid_height, grid_width)
                )
                _, mixed_mag = encoder(example_mask, return_wav=False)
                self._save_spectrogram(mixed_mag[0], task_dir / "spectrogram.png")

            self.tasks[task_id]["result_metrics"] = (
                final_F.tolist()
            )  # Return Full Front [[v, a], ...]
            self.tasks[task_id]["progress"] = 1.0

        self.tasks[task_id]["status"] = "completed"
        self.tasks[task_id]["result_path"] = str(task_dir)

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

    def get_individual_media(self, task_id: str, index: int, media_type: str):
        """
        Generates audio or spectrogram on-demand for a specific individual in a Pareto task.
        media_type: 'audio' or 'spectrogram'
        """
        if task_id not in self.tasks:
            raise ValueError("Task not found")

        task = self.tasks[task_id]
        if task["status"] != "completed":
            raise ValueError("Task not completed")

        task_dir = self.output_dir / task_id
        x_path = task_dir / "X.npy"

        if not x_path.exists():
            raise ValueError("Result data not found")

        # 1. Load Mask
        import numpy as np

        X = np.load(x_path)
        if index < 0 or index >= len(X):
            raise ValueError("Index out of bounds")

        mask_np = X[index]

        # 2. Re-create Encoder (This is heavy, maybe cache?)
        # For now, just re-create. Code reuse from _run_optimization would be good.
        params = task["params"]
        # Params store relative paths (from request)
        image_path = self.data_dir / params["image_path"]
        audio_path = self.data_dir / params["audio_path"]

        # Determine Grid Size (Logic duplicated from run_opt... strictly should refactor)
        # But for speed, let's copy the logic or assume standard?
        # WARNING: If logic changes, this breaks.
        # Ideally we valid the saved shape against valid params?
        # Let's read the shape from X to deduce grid size?
        # X shape is (Pop, H*W). We know H=64. W=?
        # W = n_params / 64.

        grid_height = 64
        n_params = mask_np.size
        grid_width = n_params // grid_height

        # Device
        device = config.DEVICE

        # Load Resources
        target_image = preprocess_image(str(image_path)).to(device)

        encoder = MaskEncoder(
            target_image,
            str(audio_path),
            grid_height=grid_height,
            grid_width=grid_width,
            smoothing_sigma=5.0,  # Match optimization default
            device=device,
        ).to(device)

        # 3. Generate
        with torch.no_grad():
            mask_tensor = (
                torch.from_numpy(mask_np)
                .float()
                .to(device)
                .view(1, grid_height, grid_width)
            )
            audio_recon, mixed_mag = encoder(mask_tensor, return_wav=True)

        # 4. Return Bytes
        import io

        if media_type == "spectrogram":
            # Save to buffer
            buf = io.BytesIO()
            self._save_spectrogram(mixed_mag[0], buf)
            buf.seek(0)
            return buf

        elif media_type == "audio":
            buf = io.BytesIO()
            self._save_audio(audio_recon, buf)
            buf.seek(0)
            return buf

        else:
            raise ValueError("Unknown media type")

    def _save_spectrogram(self, mag, path_or_buf):
        import matplotlib.pyplot as plt

        spec_db = 20 * torch.log10(mag + 1e-8).cpu().numpy()
        # Robust scaling
        ref_max = np.percentile(spec_db, 99.5)
        vmin = ref_max - 80
        vmax = ref_max

        plt.imsave(
            path_or_buf,
            spec_db,
            cmap="magma",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            format="png",
        )

    def _save_audio(self, wav, path_or_buf):
        import soundfile as sf

        wav_cpu = wav.squeeze().cpu()
        wav_cpu = wav_cpu / (torch.max(torch.abs(wav_cpu)) + 1e-6)

        # Use soundfile directly for better buffer support
        if hasattr(path_or_buf, "write"):
            # It's a file-like object (BytesIO)
            sf.write(path_or_buf, wav_cpu.numpy(), config.SAMPLE_RATE, format="WAV")
        else:
            # It's a path string/Path object
            sf.write(str(path_or_buf), wav_cpu.numpy(), config.SAMPLE_RATE)

    def _generate_animation(self, history, output_path, title="Optimization History"):
        """
        Generates an MP4 animation of the optimization history.
        history: List of np.arrays of shape (Pop, 2)

        Features:
        - Smooth transitions: Points interpolate between frames
        - Fade trails: Previous generations fade out over time
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from scipy.spatial.distance import cdist

        if not history or len(history) == 0:
            logger.warning("No history to animate")
            return

        # Phase 1 is 200 steps / 5 = 40 frames
        phase1_frames = 40

        # Setup Figure
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#09090b")
        ax.set_facecolor("#09090b")
        ax.set_title(title, color="white", fontsize=14, pad=10)
        ax.set_xlabel("Visual Loss", color="#666")
        ax.set_ylabel("Audio Loss", color="#666")
        ax.tick_params(colors="#444")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(True, alpha=0.15, color="#444")

        # Find global bounds
        all_points = np.vstack(history)
        min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
        min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()

        pad_x = (max_x - min_x) * 0.1 if max_x > min_x else 0.1
        pad_y = (max_y - min_y) * 0.1 if max_y > min_y else 0.1

        ax.set_xlim(min_x - pad_x, max_x + pad_x)
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

        # Scatter for current frame
        scat = ax.scatter(
            [], [], c="#a855f7", alpha=0.8, s=60, edgecolors="white", linewidths=0.5
        )
        # Trail scatters for fade effect (last N generations)
        trail_depth = 5
        trail_scats = [
            ax.scatter([], [], c="#4ade80", alpha=0, s=40) for _ in range(trail_depth)
        ]
        text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes, color="white", fontsize=10, va="top"
        )

        # Store previous positions for smooth interpolation
        prev_positions = None

        def match_points(prev, curr):
            """Match points between frames using nearest neighbor for smooth transition."""
            if prev is None or len(prev) != len(curr):
                return curr
            # Use Hungarian algorithm approximation via greedy matching
            dist = cdist(prev, curr)
            matched = np.zeros_like(curr)
            used = set()
            for i in range(len(prev)):
                # Find closest unused point
                dists = dist[i].copy()
                dists[list(used)] = np.inf
                j = np.argmin(dists)
                matched[i] = curr[j]
                used.add(j)
            return matched

        def update(frame):
            nonlocal prev_positions

            if frame >= len(history):
                return (scat, text, *trail_scats)

            data = history[frame]

            # Smooth transition (match to previous frame)
            if frame < phase1_frames:
                data = match_points(prev_positions, data)
            prev_positions = data.copy()

            scat.set_offsets(data)

            # Phase-based coloring
            if frame < phase1_frames:
                scat.set_facecolors("#a855f7")  # Purple for seeding
                text.set_text(f"Phase 1: Gradient Seeding (Step {frame * 5})")
                # Clear trails during Phase 1
                for ts in trail_scats:
                    ts.set_offsets(np.empty((0, 2)))
            else:
                scat.set_facecolors("#4ade80")  # Green for evolution
                gen = frame - phase1_frames
                text.set_text(f"Phase 2: Evolutionary (Gen {gen})")

                # Update trails with fade effect
                for i, ts in enumerate(trail_scats):
                    trail_frame = frame - (i + 1)
                    if trail_frame >= phase1_frames and trail_frame < len(history):
                        ts.set_offsets(history[trail_frame])
                        # Fade: newer trails are more visible
                        alpha = 0.3 * (1 - (i / trail_depth))
                        ts.set_alpha(alpha)
                    else:
                        ts.set_offsets(np.empty((0, 2)))

            return (scat, text, *trail_scats)

        ani = animation.FuncAnimation(
            fig, update, frames=len(history), blit=True, interval=100
        )

        # Save
        writer = animation.FFMpegWriter(
            fps=10, metadata=dict(artist="MOSS"), bitrate=2000
        )
        try:
            ani.save(str(output_path), writer=writer)
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")
        finally:
            plt.close(fig)


# Global Instance
moss_service = MossService(data_dir=Path(__file__).resolve().parent.parent / "data")
