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

from src import config
from src.audio_encoder import MaskEncoder
from src.audio_utils import preprocess_image
from src.objectives import calc_image_loss, calc_audio_mag_loss


def optimize_single_point(
    target_image, target_audio_path, grid_height, grid_width, args
):
    print(f"Initializing Metric Optimizer (Alpha={args.alpha})...")

    # Encoder
    encoder = MaskEncoder(
        target_image,
        target_audio_path,
        grid_height=grid_height,
        grid_width=grid_width,
        smoothing_sigma=args.sigma,
    ).to(config.DEVICE)

    # Parameters (Single Mask)
    # Init with random noise, or Audio/Image bias?
    # Let's use 0.0 (Audio bias) as start to match Adam init for index 0?
    # Or random for fairness. Random is better.
    mask_logits = nn.Parameter(torch.randn(1, encoder.n_params, device=config.DEVICE))

    optimizer = optim.Adam([mask_logits], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    print(f"Starting Optimization ({args.steps} steps)...")
    start_time = time.time()

    # Weighting
    w_vis = args.alpha
    w_aud = 1.0 - args.alpha

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            # Forward
            masks = torch.sigmoid(mask_logits)
            _, mixed_mag = encoder(masks, return_wav=False)

            # Loss
            # L1 Visual
            diff = torch.abs(mixed_mag - encoder.image_mag_ref)
            loss_vis = diff.mean()

            # Audio
            loss_aud = calc_audio_mag_loss(mixed_mag, encoder.audio_mag)

            # Weighted Sum
            total_loss = w_vis * loss_vis + w_aud * loss_aud

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 100 == 0 or step == 1:
            print(
                f"Step {step:04d}: Loss={total_loss.item():.4f} (Vis={loss_vis.item():.4f}, Aud={loss_aud.item():.4f})"
            )

    print(f"Optimization finished in {time.time() - start_time:.2f}s")

    return encoder, mask_logits


def parse_args():
    parser = argparse.ArgumentParser(description="MOSS: Direct Metric Optimization")
    parser.add_argument("--image", type=str, required=True, help="Path to target image")
    parser.add_argument("--audio", type=str, required=True, help="Path to target audio")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for Visual Loss (0.0=Audio, 1.0=Image)",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning Rate")
    parser.add_argument(
        "--sigma", type=float, default=5.0, help="Gauss smoothing sigma"
    )
    parser.add_argument("--output", "-o", type=str, default="data/output/direct_opt")
    return parser.parse_args()


def main():
    if torch.cuda.is_available():
        torch.set_num_threads(6)
        torch.cuda.empty_cache()
        try:
            torch.set_float32_matmul_precision("high")
        except:
            pass

    args = parse_args()

    # Load Data logic (Copy from run_optimization)
    try:
        target_image = preprocess_image(args.image).to(config.DEVICE)
        waveform, sr = torchaudio.load(args.audio)
        duration_sec = waveform.shape[-1] / sr
        raw_width = int(duration_sec * 8.5333)
        grid_width = ((raw_width + 15) // 16) * 16
        if grid_width < 16:
            grid_width = 16
        grid_height = 128
        print(f"Audio: {duration_sec:.2f}s -> Grid: {grid_height}x{grid_width}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    encoder, logits = optimize_single_point(
        target_image, args.audio, grid_height, grid_width, args
    )

    # Save Results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Wav
    with torch.no_grad():
        mask = torch.sigmoid(logits)
        rec_wav, mixed_mag = encoder(mask, return_wav=True)

        # Audio
        wav_cpu = rec_wav.squeeze().cpu()
        wav_cpu = wav_cpu / (torch.max(torch.abs(wav_cpu)) + 1e-6)
        torchaudio.save(
            output_dir / f"result_alpha{args.alpha}.wav",
            wav_cpu.unsqueeze(0),
            config.SAMPLE_RATE,
        )

        # Image Plot
        spec_db = 20 * torch.log10(mixed_mag[0] + 1e-8).cpu().numpy()
        plt.figure(figsize=(10, 4))
        plt.imshow(spec_db, aspect="auto", origin="lower", cmap="magma")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Direct Optimization Alpha={args.alpha}")
        plt.tight_layout()
        plt.savefig(output_dir / f"result_alpha{args.alpha}.png")
        plt.close()

    print(f"Saved artifacts to {output_dir}")


if __name__ == "__main__":
    main()
