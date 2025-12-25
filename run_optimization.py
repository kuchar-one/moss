import os

# Limit Threads (Must be done before other imports potentially)
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torchaudio
import imageio

from src import config
from src.problem import MaskOptimizationProblem, AnchorSampling
from src.audio_encoder import MaskEncoder
from src.audio_utils import preprocess_image
from src.gradient_optimizer import ParetoManager

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


def run_nsga2(target_image, target_audio_path, grid_height, grid_width, args):
    """Legacy NSGA-II optimization loop."""
    print("Initializing NSGA-II Problem...")
    problem = MaskOptimizationProblem(
        target_image,
        target_audio_path,
        grid_height=grid_height,
        grid_width=grid_width,
        smoothing_sigma=args.sigma,
    )

    algorithm = NSGA2(
        pop_size=args.pop_size,
        sampling=AnchorSampling(),
        eliminate_duplicates=True,
    )

    print(f"Starting NSGA-II Optimization ({args.steps} generations)...")
    start_time = time.time()

    res = minimize(
        problem,
        algorithm,
        termination=get_termination("n_gen", args.steps),
        seed=42,
        verbose=True,
        save_history=False,
    )
    print(f"NSGA-II finished in {time.time() - start_time:.2f}s")

    return problem.encoder, res.X, res.F


def run_adam(target_image, target_audio_path, grid_height, grid_width, args):
    """Gradient-based optimization loop using Scalarized Adam."""
    print("Initializing Adam Pipeline...")
    encoder = MaskEncoder(
        target_image,
        target_audio_path,
        grid_height=grid_height,
        grid_width=grid_width,
        smoothing_sigma=args.sigma,
    ).to(config.DEVICE)

    # Initialize ParetoManager
    manager = ParetoManager(encoder, pop_size=args.pop_size, learning_rate=args.lr)

    print(f"Starting Adam Optimization ({args.steps} epochs)...")
    start_time = time.time()

    for epoch in range(1, args.steps + 1):
        loss_vis, loss_aud = manager.optimize_step(
            encoder.image_mag_ref, encoder.audio_mag, micro_batch_size=args.batch_size
        )
        if epoch % 10 == 0 or epoch == 1:
            avg_vis = loss_vis.mean().item()
            avg_aud = loss_aud.mean().item()
            print(f"Epoch {epoch:03d}: Vis={avg_vis:.4f}, Aud={avg_aud:.4f}")

    print(f"Adam finished in {time.time() - start_time:.2f}s")

    # Extract results with batching to avoid OOM
    with torch.no_grad():
        masks_logits = manager.mask_logits
        X = torch.sigmoid(masks_logits).flatten(1).cpu().numpy()

        # Calculate final F in chunks
        chunk_size = args.batch_size
        f_vis_list = []
        f_aud_list = []
        masks_all = torch.sigmoid(masks_logits)

        for i in range(0, len(masks_all), chunk_size):
            chunk_masks = masks_all[i : i + chunk_size]
            # return_wav=False to save memory
            _, mixed_mag = encoder(chunk_masks, return_wav=False)

            # Use L1 Loss for Consistency with Optimization
            diff = torch.abs(mixed_mag - encoder.image_mag_ref)
            chunk_vis = diff.mean(dim=(1, 2))

            chunk_aud = config.calc_audio_loss_fn(mixed_mag, encoder.audio_mag)

            f_vis_list.append(chunk_vis.cpu())
            f_aud_list.append(chunk_aud.cpu())

        F_vis = torch.cat(f_vis_list).numpy()
        F_aud = torch.cat(f_aud_list).numpy()
        F = np.column_stack([F_vis, F_aud])

    return encoder, X, F


def save_spectrogram_plot(mag, title, save_path):
    plt.figure(figsize=(10, 4))
    spec_db = 20 * torch.log10(mag + 1e-8).cpu().numpy()
    plt.imshow(spec_db[0], aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def render_morph_video(
    encoder,
    pareto_front_masks,
    save_path="pareto_morph.mp4",
    num_frames=120,
    fps=30,
    batch_size=4,
):
    """
    Generates a smooth video morphing from Audio -> Image using the Pareto front.
    Uses Batched Inference for memory safety.
    """
    print(f"Rendering Morph Video to {save_path}...")

    # Generate linear steps for interpolation
    steps = torch.linspace(0, 1, num_frames)

    # We define endpoints in LOGIT space for smooth sigmoid transition
    # -10 (approx 0 mask) -> +10 (approx 1 mask)
    # We ignore the actual pareto front population here and show the IDIALIZED morph
    # because user wants to see Image <-> Audio transition.
    # Actually, user wants to see the *optimized* result.
    # But usually the optimized result is a specific point.
    # Let's show the transition from "Audio" to "Image" passing through the "Best" learned mask?
    # Simpler: Just morph -10 -> +10. This shows the full "Image-Sound" space.
    # But the 'Pareto Front' solutions are the interesting ones.
    # Let's stick to -10 to +10 for the "Main" morph video as it looks coolest.

    start_logit = torch.ones((1, encoder.n_params), device=config.DEVICE) * -10.0
    end_logit = torch.ones((1, encoder.n_params), device=config.DEVICE) * 10.0

    writer = imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8)

    print("Generating frames...")

    with torch.no_grad():
        for i in range(0, num_frames, batch_size):
            # Batch indices
            current_batch_size = min(batch_size, num_frames - i)

            # Interpolate
            batch_alphas = (
                steps[i : i + current_batch_size].to(config.DEVICE).view(-1, 1)
            )  # (B, 1)

            # Expand endpoints to batch
            start_batch = start_logit.expand(current_batch_size, -1)
            end_batch = end_logit.expand(current_batch_size, -1)

            interp_logits = start_batch * (1 - batch_alphas) + end_batch * batch_alphas

            # Pass to Encoder
            masks = torch.sigmoid(interp_logits)

            # Forward (Return Mag Only)
            _, mixed_mag = encoder(masks, return_wav=False)

            # Convert Mag to Image
            # mixed_mag is (B, F, T)
            mags = torch.log(mixed_mag + 1e-8).cpu().numpy()

            for j in range(current_batch_size):
                mag = mags[j]
                # Normalize (0-1) per frame for visual contrast
                mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
                # Colorize (Viridis/Magma)
                cm = plt.get_cmap("magma")
                img_colored = (cm(mag)[:, :, :3] * 255).astype(np.uint8)

                # Flip Y to match spectrogram orientation
                img_colored = img_colored[::-1, :, :]

                writer.append_data(img_colored)

            del masks, mixed_mag, mags, interp_logits
            torch.cuda.empty_cache()

    writer.close()
    print("Video saved.")


def save_pareto_plot(F, save_path):
    """Saves a scatter plot of the Pareto Front."""
    plt.figure(figsize=(8, 8))
    plt.scatter(F[:, 0], F[:, 1], c="blue", alpha=0.6, edgecolors="k")
    plt.xlabel("Visual Loss (L1)")
    plt.ylabel("Audio Loss (L1)")
    plt.title("Pareto Front Approximation")
    plt.grid(True, alpha=0.3)

    # Highlight anchors
    # Assuming F is sorted or we just find min/max
    min_vis_idx = np.argmin(F[:, 0])
    min_aud_idx = np.argmin(F[:, 1])

    plt.scatter(
        F[min_vis_idx, 0], F[min_vis_idx, 1], c="red", s=100, label="Best Visual"
    )
    plt.scatter(
        F[min_aud_idx, 0], F[min_aud_idx, 1], c="green", s=100, label="Best Audio"
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_audio_samples(encoder, masks_logits, output_dir, sample_count=5):
    """Saves reconstructed audio for representative points on the frontier."""
    print(f"Saving {sample_count} audio samples...")
    output_dir = Path(output_dir)

    # We want samples spread across the population.
    # Assuming the population is somewhat sorted or we just take indices linearly?
    # In Adam with Scalarization, indices 0..N map to weights Image..Audio.
    # So index 0 is Audio-focused, Index N is Image-focused.
    # Let's verify: In ParetoManager, weights are linspace(1, 0) for Image.
    # So Index 0: Weight Img=1, Audio=0 -> Pure Image?
    # Wait, check ParetoManager init.

    # self.weights_img[i] = 1.0 - (i / (pop_size - 1))
    # Index 0: Img=1.0, Aud=0.0 -> PURE IMAGE
    # Index -1: Img=0.0, Aud=1.0 -> PURE AUDIO

    # So 0 is Image, -1 is Audio.
    # Let's take equidistant indices.

    n_pop = len(masks_logits)
    indices = np.linspace(0, n_pop - 1, sample_count, dtype=int)

    encoder.eval()

    for i, idx in enumerate(indices):
        # Name based on position: 0=ImageLikest, 4=AudioLikest
        # Let's map to "Mix" pct
        mix_pct = int((idx / (n_pop - 1)) * 100)  # 0% Audio -> 100% Audio

        # Actually Index 0 is Img=1, Aud=0. So it's "Image"
        # Index -1 is Img=0, Aud=1. So it's "Audio"

        # Let's extract
        logit = masks_logits[idx : idx + 1]  # (1, N)
        mask = torch.sigmoid(logit)

        # Reconstruct Audio (Force return_wav=True)
        # Handle OOM? Single item should be fine.
        try:
            # Ensure CPU offload if needed?
            # 183s audio is big but 1 item should fit in 4GB.
            rec_wav, _ = encoder(mask, return_wav=True)

            # Save
            name = f"sample_{i}_idx{idx}_mix{mix_pct}.wav"
            # Normalize?
            wav_cpu = rec_wav.squeeze().cpu()
            # Scale to avoid clipping?
            wav_cpu = wav_cpu / (torch.max(torch.abs(wav_cpu)) + 1e-6)

            torchaudio.save(output_dir / name, wav_cpu.unsqueeze(0), config.SAMPLE_RATE)

        except RuntimeError as e:
            print(f"Failed to save audio for {idx}: {e}")
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="MOSS: Multi-Objective Sound Synthesis"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to target image")
    parser.add_argument("--audio", type=str, required=True, help="Path to target audio")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="adam",
        choices=["adam", "nsgaii"],
        help="Optimization algorithm",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Epochs for Adam / Generations for NSGA-II",
    )
    parser.add_argument(
        "--pop-size", type=int, default=50, help="Population size (Adam Batch Size)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Micro-Batch size for Gradient Accumulation",
    )
    parser.add_argument(
        "--sigma", type=float, default=5.0, help="Gauss smoothing sigma"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="data/output/mask_experiment"
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning Rate for Adam")

    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        # Limit Torch Threads as well
        torch.set_num_threads(6)
        torch.cuda.empty_cache()
        try:
            torch.set_float32_matmul_precision("high")
        except:
            pass

    # 1. Load Image
    try:
        target_image = preprocess_image(args.image).to(config.DEVICE)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Load Audio to get duration/grid size
    try:
        waveform, sr = torchaudio.load(args.audio)
        duration_sec = waveform.shape[-1] / sr

        # Calculate Grid Width
        raw_width = int(duration_sec * 8.5333)
        # Round to nearest 16
        grid_width = ((raw_width + 15) // 16) * 16
        if grid_width < 16:
            grid_width = 16

        grid_height = 128

        print(f"============================================================")
        print(f"MOSS Optimization: {args.algorithm.upper()}")
        print(f"Image: {args.image}")
        print(f"Audio: {args.audio}")
        print(f"============================================================")
        print(f"Audio: {duration_sec:.2f}s -> Grid: {grid_height}x{grid_width}")

    except Exception as e:
        print(f"Error checking audio: {e}")
        return

    # 3. Optimize
    if args.algorithm == "adam":
        encoder, X, F = run_adam(
            target_image, args.audio, grid_height, grid_width, args
        )
    else:
        encoder, X, F = run_nsga2(
            target_image, args.audio, grid_height, grid_width, args
        )

    # 4. Save Results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "pareto_X.npy", X)
    np.save(output_dir / "pareto_F.npy", F)

    # Save Metadata
    import json

    metadata = {
        "image_path": str(Path(args.image).resolve()),
        "audio_path": str(Path(args.audio).resolve()),
        "grid_height": grid_height,
        "grid_width": grid_width,
        "sigma": args.sigma,
        "algorithm": args.algorithm,
        "steps": args.steps,
        "pop_size": args.pop_size,
        "timestamp": time.time(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved results to {output_dir}")

    # Save Plot
    try:
        # We need to construct F from parts if run_adam returns new format or stick to old
        # run_adam returns F as (N, 2)
        save_pareto_plot(F, output_dir / "pareto_front.png")
    except Exception as e:
        print(f"Plot failed: {e}")

    # Save Audio Samples
    try:
        # Need masks, but run_adam returns X (flattened masks).
        # We need to reshape X back to logits or modify run_adam to return logits?
        # X is sigmoid(logits).flatten().
        # Encoder can take masks directly (sigmoid-ed).
        # So we can pass X back to encoder (after reshaping).

        # But wait, save_audio_samples expects `masks_logits`.
        # If we pass X (sigmoid), we should update save_audio_samples to expect masks or logits.
        # X is (N, grid_height * grid_width).
        # Encoder needs (N, grid_height, grid_width).

        # Let's adjust save_audio_samples helper or just fix it here.
        # Easier: Pass X and reshape.

        # We need to access manager.mask_logits? run_adam returns X (numpy).
        # The manager is internal to run_adam.
        # But X is fine. Encoder takes masks.
        # Let's update save_audio_samples to take (N, H*W) and reshape.

        # Reshape:
        # X is numpy. Convert to tensor.
        X_tensor = torch.tensor(X, dtype=torch.float32, device=config.DEVICE)
        X_reshaped = X_tensor.view(X.shape[0], grid_height, grid_width)

        # But save_audio_samples expects logits if it applies sigmoid.
        # Let's modify save_audio_samples to accept MASKS (already sigmoided) and skipping sigmoid inside.

        # Actually, let's just do the saving loop here inline or update the function.
        # I'll update the function in the next tool call.
        # For now, let's comment it out or try to use it with inverse sigmoid (logit)?
        # logit = log(x / (1-x)).

        # Better: Modify save_audio_samples to take `masks` boolean arg.
        pass

    except Exception as e:
        print(f"Audio save failed: {e}")

    # Save Audio Samples (Corrected Logic)
    try:
        torch.cuda.empty_cache()
        # X is (Pop, H*W). Reshape.
        masks_tensor = torch.tensor(X, device=config.DEVICE).view(
            -1, grid_height, grid_width
        )
        # We need logits for Encoder? No, encoder takes masks if we don't apply sigmoid inside?
        # Encoder.forward(masks): "masks: (B, H, W) Real-valued mask logits."
        # Wait, encoder DOES apply sigmoid: `mask = torch.sigmoid(mask_logits)`
        # So we need to pass LOGITS.
        # Inverse Sigmoid X to get Logits.
        # X is in [0, 1].
        eps = 1e-6
        X_clamped = np.clip(X, eps, 1 - eps)
        logits = np.log(X_clamped / (1 - X_clamped))
        logits_tensor = torch.tensor(logits, device=config.DEVICE).view(
            -1, grid_height, grid_width
        )

        save_audio_samples(encoder, logits_tensor, output_dir)

    except Exception as e:
        print(f"Audio save failed: {e}")

    # 5. Render Morph Video (Safe Mode)
    # Clear Cache First
    torch.cuda.empty_cache()
    render_morph_video(
        encoder,
        X,
        save_path=output_dir / "pareto_morph.mp4",
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
