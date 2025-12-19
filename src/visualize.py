"""Visualization tools for optimization results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from . import config
from .audio_utils import spectrogram_to_image


def plot_pareto_front(
    F: np.ndarray,
    save_path: str = None,
    title: str = "Pareto Front: Visual vs Musical Loss",
) -> plt.Figure:
    """Plot the Pareto front from optimization results.

    Args:
        F: Objective values of shape (n_solutions, n_objectives)
        save_path: Optional path to save the figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(F[:, 0], F[:, 1], c="blue", alpha=0.6, s=50)
    ax.set_xlabel("Visual Loss (1 - SSIM)", fontsize=12)
    ax.set_ylabel("Musical Loss (Roughness)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark extreme points
    best_visual_idx = np.argmin(F[:, 0])
    best_musical_idx = np.argmin(F[:, 1])

    ax.scatter(
        F[best_visual_idx, 0],
        F[best_visual_idx, 1],
        c="green",
        s=200,
        marker="*",
        label="Best Visual",
        zorder=5,
    )
    ax.scatter(
        F[best_musical_idx, 0],
        F[best_musical_idx, 1],
        c="red",
        s=200,
        marker="*",
        label="Best Musical",
        zorder=5,
    )

    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved Pareto front to {save_path}")

    return fig


def plot_spectrogram_comparison(
    target: np.ndarray,
    generated: np.ndarray,
    save_path: str = None,
    title: str = "Spectrogram Comparison",
) -> plt.Figure:
    """Plot target and generated spectrograms side by side.

    Args:
        target: Target spectrogram image
        generated: Generated spectrogram image
        save_path: Optional path to save the figure
        title: Plot title

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(target, aspect="auto", cmap="magma", origin="lower")
    axes[0].set_title("Target Image", fontsize=12)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency (Mel)")

    axes[1].imshow(generated, aspect="auto", cmap="magma", origin="lower")
    axes[1].set_title("Generated Spectrogram", fontsize=12)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Frequency (Mel)")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison to {save_path}")

    return fig


def plot_pareto_walk(
    spectrograms: list,
    labels: list,
    target: np.ndarray,
    save_path: str = None,
) -> plt.Figure:
    """Plot multiple solutions along the Pareto front.

    Args:
        spectrograms: List of spectrogram arrays from different solutions
        labels: Labels for each solution (e.g., "Best Visual", "Balanced", "Best Audio")
        target: Target image spectrogram
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure
    """
    n = len(spectrograms) + 1  # +1 for target
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    # Plot target
    axes[0].imshow(target, aspect="auto", cmap="magma", origin="lower")
    axes[0].set_title("Target", fontsize=11)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Frequency")

    # Plot solutions
    for i, (spec, label) in enumerate(zip(spectrograms, labels)):
        axes[i + 1].imshow(spec, aspect="auto", cmap="magma", origin="lower")
        axes[i + 1].set_title(label, fontsize=11)
        axes[i + 1].set_xlabel("Time")

    plt.suptitle("Pareto Front Walk: Visual ← → Musical", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved Pareto walk to {save_path}")

    return fig


def select_diverse_solutions(F: np.ndarray, X: np.ndarray, n: int = 5) -> tuple:
    """Select diverse solutions along the Pareto front.

    Args:
        F: Objective values of shape (n_solutions, n_objectives)
        X: Decision variable values of shape (n_solutions, n_var)
        n: Number of solutions to select

    Returns:
        Tuple of (selected_F, selected_X, labels)
    """
    # Sort by visual loss (f1)
    sorted_indices = np.argsort(F[:, 0])

    # Select evenly spaced solutions
    step = max(1, len(sorted_indices) // (n - 1))
    selected_indices = [
        sorted_indices[min(i * step, len(sorted_indices) - 1)] for i in range(n)
    ]

    # Ensure we include extremes
    best_visual = np.argmin(F[:, 0])
    best_musical = np.argmin(F[:, 1])

    if best_visual not in selected_indices:
        selected_indices[0] = best_visual
    if best_musical not in selected_indices:
        selected_indices[-1] = best_musical

    selected_F = F[selected_indices]
    selected_X = X[selected_indices]

    labels = []
    for i, idx in enumerate(selected_indices):
        if idx == best_visual:
            labels.append("Best Visual")
        elif idx == best_musical:
            labels.append("Best Musical")
        else:
            labels.append(f"Balanced {i}")

    return selected_F, selected_X, labels
