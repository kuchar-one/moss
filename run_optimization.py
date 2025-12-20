#!/usr/bin/env python3
"""Run Mask-Based Image-Sound Encoding Optimization (Adam or NSGA-II)."""

import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# Gradient Imports
from src.gradient_optimizer import ParetoManager

# GA Imports
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling
from pymoo.termination import get_termination
from src.problem import MaskOptimizationProblem


class AnchorSampling(Sampling):
    """Custom sampling that injects extreme solutions (All-0 and All-1)."""

    def _do(self, problem, n_samples, **kwargs):
        X = np.random.random((n_samples, problem.n_var))
        X[0, :] = 0.0
        if n_samples > 1:
            X[1, :] = 1.0
        return X


from src import config
from src.audio_encoder import MaskEncoder
from src.visualize import plot_pareto_front, select_diverse_solutions
from src.audio_utils import preprocess_image


def parse_args():
    parser = argparse.ArgumentParser(description="MOSS: Dual-Algo Mask Optimization")
    parser.add_argument("--image", "-i", type=str, required=True)
    parser.add_argument("--audio", "-a", type=str, required=True)
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["adam", "nsgaii"],
        default="adam",
        help="Optimization algorithm",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=300,
        help="Epochs for Adam / Generations for NSGA-II",
    )
    parser.add_argument("--pop-size", "-p", type=int, default=30)
    parser.add_argument(
        "--output", "-o", type=str, default="data/output/mask_experiment"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning Rate for Adam")
    parser.add_argument(
        "--sigma", type=float, default=5.0, help="Gaussian Smoothing Sigma"
    )
    return parser.parse_args()


def run_adam(target_image, target_audio_path, grid_h, grid_w, params):
    """Run Gradient Descent Optimization."""
    print("Initializing Adam Pipeline...")
    encoder = MaskEncoder(
        target_image=target_image,
        target_audio_path=target_audio_path,
        grid_height=grid_h,
        grid_width=grid_w,
        smoothing_sigma=params.sigma,
        device=config.DEVICE,
    ).to(config.DEVICE)

    manager = ParetoManager(
        encoder, pop_size=params.pop_size, learning_rate=params.lr
    ).to(config.DEVICE)

    start_time = time.time()
    print(f"Starting Adam Optimization ({params.steps} epochs)...")

    for epoch in range(1, params.steps + 1):
        loss_vis, loss_aud = manager.optimize_step(
            encoder.image_mag_ref, encoder.audio_mag
        )
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d}: Vis={loss_vis.mean():.4f}, Aud={loss_aud.mean():.4f}"
            )

    print(f"Adam finished in {time.time() - start_time:.2f}s")

    # Extract
    with torch.no_grad():
        masks_logits = manager.mask_logits
        X = torch.sigmoid(masks_logits).flatten(1).cpu().numpy()
        # Calculate final F
        masks = torch.sigmoid(masks_logits)
        _, mixed_mag = encoder(masks)
        F_vis = (
            config.calc_image_loss_fn(mixed_mag, encoder.image_mag_ref).cpu().numpy()
        )  # Helper added to config
        F_aud = config.calc_audio_loss_fn(mixed_mag, encoder.audio_mag).cpu().numpy()
        F = np.column_stack([F_vis, F_aud])

    return encoder, X, F


def run_nsgaii(target_image, target_audio_path, grid_h, grid_w, params):
    """Run Genetic Algorithm Optimization."""
    print("Initializing NSGA-II Pipeline...")
    # Problem handles Encoder init internally
    # Wait, problem takes path, not loaded tensor.
    # But we should ensure dynamic sizing is consistent.
    # We pass grid dimensions.

    # Problem definition expects path for image
    problem = MaskOptimizationProblem(
        target_image_path=params.image,  # Pass path for consistency with problem.py interface
        target_audio_path=target_audio_path,
        grid_height=grid_h,
        grid_width=grid_w,
        smoothing_sigma=params.sigma,
    )

    algorithm = NSGA2(pop_size=params.pop_size, sampling=AnchorSampling())
    termination = get_termination("n_gen", params.steps)

    start_time = time.time()
    print(f"Starting NSGA-II Optimization ({params.steps} generations)...")

    res = minimize(problem, algorithm, termination, seed=42, verbose=True)

    print(f"NSGA-II finished in {time.time() - start_time:.2f}s")

    return problem.encoder, res.X, res.F


def save_spectrogram_plot(spec, title, path):
    plt.figure(figsize=(10, 4))
    spec_db = 20 * torch.log10(spec + 1e-8)
    plt.imshow(spec_db.cpu().numpy(), aspect="auto", origin="lower", cmap="magma")
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def render_morph_video(encoder, X_pareto, F_pareto, output_path: Path):
    print(f"Rendering morph video to {output_path}...")
    sorted_idx = np.argsort(F_pareto[:, 0])
    X_sorted = X_pareto[sorted_idx]
    F_sorted = F_pareto[sorted_idx]

    n_steps = 10
    fig, ax = plt.subplots(figsize=(10, 6))
    ims = []

    for i in range(len(X_sorted) - 1):
        start_genes = X_sorted[i]
        end_genes = X_sorted[i + 1]
        start_loss = F_sorted[i]
        end_loss = F_sorted[i + 1]

        for step in range(n_steps):
            alpha = step / n_steps
            interp_genes = start_genes * (1 - alpha) + end_genes * alpha

            params_t = torch.tensor(
                interp_genes, dtype=torch.float32, device=config.DEVICE
            ).unsqueeze(0)
            with torch.no_grad():
                _, mixed_mag = encoder(params_t)

            spec_db = 20 * torch.log10(mixed_mag[0] + 1e-8).cpu().numpy()
            im = ax.imshow(
                spec_db, aspect="auto", origin="lower", cmap="magma", animated=True
            )
            cur_vis = start_loss[0] * (1 - alpha) + end_loss[0] * alpha
            cur_aud = start_loss[1] * (1 - alpha) + end_loss[1] * alpha
            title = ax.text(
                0.5,
                1.01,
                f"Morph: L_vis={cur_vis:.2f}, L_aud={cur_aud:.2f}",
                ha="center",
                va="bottom",
                transform=ax.transAxes,
                fontsize=12,
            )
            ims.append([im, title])

    # Final Frame
    params_t = torch.tensor(
        X_sorted[-1], dtype=torch.float32, device=config.DEVICE
    ).unsqueeze(0)
    with torch.no_grad():
        _, mixed_mag = encoder(params_t)
    spec_db = 20 * torch.log10(mixed_mag[0] + 1e-8).cpu().numpy()
    im = ax.imshow(spec_db, aspect="auto", origin="lower", cmap="magma", animated=True)
    title = ax.text(
        0.5,
        1.01,
        f"Final: L_vis={F_sorted[-1, 0]:.2f}, L_aud={F_sorted[-1, 1]:.2f}",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=12,
    )
    ims.append([im, title])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(str(output_path), writer="pillow", fps=20)
    plt.close()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except:
            pass

    output_dir = Path(args.output) / args.algorithm
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"MOSS Optimization: {args.algorithm.upper()}")
    print(f"Image: {args.image}")
    print(f"Audio: {args.audio}")
    print("=" * 60)

    # Dynamic Grid Calculation
    waveform, sr = torchaudio.load(args.audio)
    duration = waveform.shape[-1] / sr
    # Density: 8.53 cols/sec
    grid_width = int(duration * 8.53)
    grid_width = max(64, (grid_width // 16) * 16)
    grid_height = 128
    print(f"Audio: {duration:.2f}s -> Grid: {grid_height}x{grid_width}")

    # Dispatch
    if args.algorithm == "adam":
        target_image = preprocess_image(args.image)  # Adam needs tensor
        encoder, X, F = run_adam(
            target_image, args.audio, grid_height, grid_width, args
        )
    else:
        # NSGA-II loads image internally inside Problem
        encoder, X, F = run_nsgaii(None, args.audio, grid_height, grid_width, args)

    print("Saving results...")
    np.save(output_dir / "results.npy", {"X": X, "F": F})
    plot_pareto_front(F, save_path=str(output_dir / "pareto_front.png"))

    render_morph_video(encoder, X, F, output_dir / "pareto_morph.gif")

    # Select solutions
    selected_F, selected_X, labels = select_diverse_solutions(F, X, n=5)

    for params, label, objs in zip(selected_X, labels, selected_F):
        safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        params_t = torch.tensor(
            params, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)
        with torch.no_grad():
            audio, mixed_mag = encoder(params_t)

        torchaudio.save(
            output_dir / f"{safe_label}.wav", audio.cpu(), config.SAMPLE_RATE
        )

        save_spectrogram_plot(
            mixed_mag[0],
            f"{label} (L_vis={objs[0]:.2f}, L_aud={objs[1]:.2f})",
            output_dir / f"{safe_label}_spec.png",
        )

        # Mask Viz
        grid = params_t.view(1, 1, grid_height, grid_width)
        mask = torch.nn.functional.interpolate(
            grid, size=(encoder.full_height, encoder.full_width), mode="bilinear"
        )[0, 0]
        plt.figure(figsize=(10, 4))
        plt.imshow(
            mask.cpu().numpy(),
            aspect="auto",
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        plt.title(f"Mask: {label}")
        plt.colorbar()
        plt.savefig(output_dir / f"{safe_label}_mask.png")
        plt.close()

    print(f"Done! Results in {output_dir}")


if __name__ == "__main__":
    main()
