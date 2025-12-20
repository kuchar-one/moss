#!/usr/bin/env python3
"""Run Mask-Based Image-Sound Encoding Optimization."""

import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling


class AnchorSampling(Sampling):
    """Custom sampling that injects extreme solutions (All-0 and All-1)."""

    def _do(self, problem, n_samples, **kwargs):
        # Generate random samples using standard FloatRandom
        X = np.random.random((n_samples, problem.n_var))

        # Inject "All Audio" (Mask = 0)
        X[0, :] = 0.0

        # Inject "All Image" (Mask = 1)
        if n_samples > 1:
            X[1, :] = 1.0

        return X


from pymoo.termination import get_termination

from src import config
from src.problem import MaskOptimizationProblem
from src.visualize import plot_pareto_front, plot_pareto_walk, select_diverse_solutions


def parse_args():
    parser = argparse.ArgumentParser(description="MOSS: Mask Optimization")
    parser.add_argument("--image", "-i", type=str, required=True)
    parser.add_argument("--audio", "-a", type=str, required=True)
    parser.add_argument("--generations", "-g", type=int, default=50)
    parser.add_argument("--pop-size", "-p", type=int, default=50)
    parser.add_argument("--output", "-o", type=str, default="data/output/mask_log_22k")
    parser.add_argument("--grid-height", type=int, default=64)
    parser.add_argument("--grid-width", type=int, default=128)
    return parser.parse_args()


def save_spectrogram_plot(spec, title, path):
    plt.figure(figsize=(10, 4))
    # spec is (F, T) magnitude
    # Convert to dB for plotting
    spec_db = 20 * torch.log10(spec + 1e-8)
    plt.imshow(spec_db.cpu().numpy(), aspect="auto", origin="lower", cmap="magma")
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def render_morph_video(encoder, X_pareto, F_pareto, output_path: Path):
    """Render a video morphing through the Pareto front."""
    print(f"Rendering morph video to {output_path}...")

    # Sort by Visual Loss (F[:, 0])
    sorted_idx = np.argsort(F_pareto[:, 0])
    X_sorted = X_pareto[sorted_idx]
    F_sorted = F_pareto[sorted_idx]

    # Generate frames
    fig, ax = plt.subplots(figsize=(10, 6))

    ims = []
    # To make it smoother, interpolate? For now just raw frames
    for i, params in enumerate(X_sorted):
        params_t = torch.tensor(
            params, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)
        with torch.no_grad():
            _, mixed_mag = encoder(params_t)

        # Log magnitude in dB
        spec_db = 20 * torch.log10(mixed_mag[0] + 1e-8).cpu().numpy()

        im = ax.imshow(
            spec_db, aspect="auto", origin="lower", cmap="magma", animated=True
        )
        title = ax.text(
            0.5,
            1.01,
            f"Sol {i}: VisL={F_sorted[i, 0]:.2f}, AudL={F_sorted[i, 1]:.2f}",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=12,
        )
        ims.append([im, title])

    ani = animation.ArtistAnimation(
        fig, ims, interval=200, blit=True, repeat_delay=1000
    )

    # Save as GIF (more portable than needing ffmpeg for mp4)
    ani.save(str(output_path), writer="pillow", fps=5)
    plt.close()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MOSS: Mask-Based Optimization")
    print(f"Image: {args.image}")
    print(f"Audio: {args.audio}")
    print("=" * 60)

    problem = MaskOptimizationProblem(
        target_image_path=args.image,
        target_audio_path=args.audio,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
    )

    algorithm = NSGA2(pop_size=args.pop_size, sampling=AnchorSampling())
    termination = get_termination("n_gen", args.generations)

    print("Starting optimization...")
    res = minimize(problem, algorithm, termination, seed=42, verbose=True)

    print("\noptimization complete.")

    # Analyze Results
    F = res.F
    X = res.X

    # Save raw results
    np.save(output_dir / "results.npy", {"X": X, "F": F})

    plot_pareto_front(F, save_path=str(output_dir / "pareto_front.png"))

    # Render Morph Video (Smooth Pareto Traversal)
    render_morph_video(problem.encoder, X, F, output_dir / "pareto_morph.gif")

    # Select solutions
    selected_F, selected_X, labels = select_diverse_solutions(F, X, n=5)

    specs_for_walk = []

    # Save Audio and Plots for selected solutions
    for params, label, objs in zip(selected_X, labels, selected_F):
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        print(f"Rendering {label} (Loss: {objs})")

        params_t = torch.tensor(
            params, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            audio, mixed_mag = problem.encoder(params_t)

        # Save Audio
        torchaudio.save(
            str(output_dir / f"{safe_label}.wav"), audio[0].cpu(), config.SAMPLE_RATE
        )

        # Save Spectrogram Plot
        spec = mixed_mag[0]
        specs_for_walk.append(spec)  # Keep raw mag for walk plot
        save_spectrogram_plot(
            spec,
            f"{label} (ImgL={objs[0]:.2f}, AudL={objs[1]:.2f})",
            output_dir / f"{safe_label}_spec.png",
        )

        # Save Mask visualization
        # Reconstruct mask from params
        grid = params_t.view(1, 1, args.grid_height, args.grid_width)
        mask = torch.nn.functional.interpolate(
            grid,
            size=(problem.encoder.full_height, problem.encoder.full_width),
            mode="bilinear",
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
        plt.title(f"Mask: {label} (White=Image, Black=Audio)")
        plt.colorbar()
        plt.savefig(output_dir / f"{safe_label}_mask.png")
        plt.close()

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
