#!/usr/bin/env python3
"""Main optimization script for Multi-Objective Spectral Synthesis.

Usage:
    python run_optimization.py --image data/input/target.jpg
    python run_optimization.py --image data/input/target.jpg --generations 100 --visual-only
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torchaudio

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src import config
from src.problem import SpectralOptimization, BatchSpectralOptimization
from src.audio_utils import audio_to_spectrogram, preprocess_image, spectrogram_to_image
from src.visualize import (
    plot_pareto_front,
    plot_spectrogram_comparison,
    plot_pareto_walk,
    select_diverse_solutions,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Objective Spectral Synthesis")
    parser.add_argument(
        "--image", "-i", type=str, required=True, help="Path to target image file"
    )
    parser.add_argument(
        "--generations",
        "-g",
        type=int,
        default=config.N_GEN,
        help=f"Number of generations (default: {config.N_GEN})",
    )
    parser.add_argument(
        "--pop-size",
        "-p",
        type=int,
        default=config.POP_SIZE,
        help=f"Population size (default: {config.POP_SIZE})",
    )
    parser.add_argument(
        "--visual-only",
        action="store_true",
        help="Only optimize visual loss (Phase 2 testing)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/output",
        help="Output directory (default: data/output)",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Use batch evaluation (faster on GPU)"
    )
    return parser.parse_args()


def render_and_save_audio(params: np.ndarray, output_path: str, synth):
    """Render audio from parameters and save to file."""
    params_tensor = torch.tensor(
        params, dtype=torch.float32, device=config.DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        audio = synth(params_tensor)

    audio = audio.cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    torchaudio.save(output_path, audio, config.SAMPLE_RATE)
    print(f"Saved audio to {output_path}")

    return audio


def main():
    args = parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Target image not found: {args.image}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Multi-Objective Spectral Synthesis (MOSS)")
    print("=" * 60)
    print(f"Target image: {args.image}")
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.pop_size}")
    print(f"Visual only: {args.visual_only}")
    print(f"Device: {config.DEVICE}")
    print("=" * 60)

    # Initialize problem (synth is now inside problem)
    if args.batch:
        problem = BatchSpectralOptimization(args.image, visual_only=args.visual_only)
    else:
        problem = SpectralOptimization(args.image, visual_only=args.visual_only)

    # Get synth from problem for later rendering
    synth = problem.synth

    # Setup NSGA-II algorithm
    algorithm = NSGA2(pop_size=args.pop_size)
    termination = get_termination("n_gen", args.generations)

    print("\nStarting optimization...")

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=42,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Optimization complete!")
    print(f"Solutions found: {len(result.F)}")
    print("=" * 60)

    F = result.F
    X = result.X

    target_tensor = preprocess_image(args.image)
    target_img = spectrogram_to_image(target_tensor)

    if args.visual_only:
        F = F.flatten() if F.ndim > 1 else F
        best_idx = np.argmin(F)
        best_params = X[best_idx] if X.ndim > 1 else X

        print(f"\nBest visual loss: {F[best_idx]:.4f}")

        render_and_save_audio(best_params, str(output_dir / "best_visual.wav"), synth)

        params_tensor = torch.tensor(
            best_params, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)
        with torch.no_grad():
            audio = synth(params_tensor)
            spec = audio_to_spectrogram(audio)

        spec_img = spectrogram_to_image(spec[0])
        plot_spectrogram_comparison(
            target_img,
            spec_img,
            save_path=str(output_dir / "comparison.png"),
            title="Visual-Only Optimization Result",
        )
    else:
        plot_pareto_front(F, save_path=str(output_dir / "pareto_front.png"))

        selected_F, selected_X, labels = select_diverse_solutions(F, X, n=5)

        print("\nSelected solutions:")
        for f, label in zip(selected_F, labels):
            print(f"  {label}: Visual={f[0]:.4f}, Musical={f[1]:.4f}")

        spectrograms = []

        for params, label in zip(selected_X, labels):
            safe_label = label.lower().replace(" ", "_")
            wav_path = str(output_dir / f"{safe_label}.wav")

            audio = render_and_save_audio(params, wav_path, synth)

            with torch.no_grad():
                spec = audio_to_spectrogram(audio.to(config.DEVICE))

            spec_img = spectrogram_to_image(spec[0])
            spectrograms.append(spec_img)

        plot_pareto_walk(
            spectrograms,
            labels,
            target_img,
            save_path=str(output_dir / "pareto_walk.png"),
        )

    print(f"\nResults saved to {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
