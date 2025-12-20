#!/usr/bin/env python3
"""Run Image-Sound Encoding Optimization.

Usage:
    python run_optimization.py --image monalisa.jpg --audio ambient.wav
    python run_optimization.py --image monalisa.jpg  # No target audio
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
from src.problem import ImageSoundProblem
from src.audio_utils import spectrogram_to_image
from src.visualize import plot_pareto_front, plot_pareto_walk, select_diverse_solutions


def parse_args():
    parser = argparse.ArgumentParser(description="Image-Sound Encoding via MOO")
    parser.add_argument("--image", "-i", type=str, required=True, help="Target image")
    parser.add_argument(
        "--audio", "-a", type=str, default=None, help="Target audio (optional)"
    )
    parser.add_argument("--generations", "-g", type=int, default=50)
    parser.add_argument("--pop-size", "-p", type=int, default=50)
    parser.add_argument("--output", "-o", type=str, default="data/output")
    parser.add_argument("--grid-height", type=int, default=32)
    parser.add_argument("--grid-width", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.audio and not Path(args.audio).exists():
        raise FileNotFoundError(f"Audio not found: {args.audio}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Image-Sound Encoding via Multi-Objective Optimization")
    print("=" * 60)
    print(f"Target image: {args.image}")
    print(f"Target audio: {args.audio or 'None (using image as sound target)'}")
    print(
        f"Grid: {args.grid_height}x{args.grid_width} = {args.grid_height * args.grid_width} params"
    )
    print(f"Generations: {args.generations}")
    print(f"Population: {args.pop_size}")
    print("=" * 60)

    # Create problem
    problem = ImageSoundProblem(
        target_image_path=args.image,
        target_audio_path=args.audio,
        grid_height=args.grid_height,
        grid_width=args.grid_width,
    )

    # Run optimization
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
    print(f"Pareto solutions: {len(result.F)}")
    print("=" * 60)

    F = result.F
    X = result.X

    # Plot Pareto front
    plot_pareto_front(F, save_path=str(output_dir / "pareto_front.png"))

    # Select diverse solutions
    selected_F, selected_X, labels = select_diverse_solutions(F, X, n=5)

    print("\nSelected solutions:")
    for f, label in zip(selected_F, labels):
        print(f"  {label}: Image={f[0]:.4f}, Sound={f[1]:.4f}")

    # Render and save selected solutions
    spectrograms = []
    encoder = problem.encoder

    for params, label in zip(selected_X, labels):
        safe_label = label.lower().replace(" ", "_")

        params_tensor = torch.tensor(
            params, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            audio, spec = encoder(params_tensor)

        # Save audio
        wav_path = str(output_dir / f"{safe_label}.wav")
        torchaudio.save(wav_path, audio.cpu(), config.SAMPLE_RATE)
        print(f"Saved {wav_path}")

        # Collect spectrogram for visualization
        spec_img = spectrogram_to_image(spec[0])
        spectrograms.append(spec_img)

    # Plot Pareto walk
    target_img = spectrogram_to_image(encoder.target_image)
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
