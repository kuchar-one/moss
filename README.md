# MOSS: Multi-Objective Spectral Synthesis

**MOSS** is a system that creates "Hybrid Audio" that lies on the boundary between an **Image** and a **Sound**.

It uses **Multi-Objective Evolutionary Optimization (NSGA-II)** to optimize a spectral mask that blends a target image with a target ambient track.
The result is a set of audio files (the Pareto Front) ranging from "Pure Music" to "Pure Image Spectrogram", with smooth morphing between them.

## Key Features

- **Mask-Based Encoding**: Optimizes a 128x256 mask to blend visual and auditory content in the time-frequency domain.
- **Phase-Aware Reconstruction**: Uses the **phase** of the target audio for reconstruction, ensuring the output sounds like natural audio/drone rather than synthetic noise (Griffin-Lim artifacts).
- **Log-Domain Mixing**: Blends magnitudes in Decibels (dB), creating creating natural fade-ins/outs.
- **Musical Morphing**: Applies **Gaussian Smoothing (Sigma=3.0)** to the mask, forcing the AI to select "organic" spectral shapes rather than individual noisy pixels.
- **High-Resolution**: Maps the image to the **0-11kHz** audible range (22,050Hz SR) for maximum clarity and detail.
- **Real-Time Morphing Video**: Automatically generates a smooth GIF (`pareto_morph.gif`) interpolating between all solutions on the Pareto front.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchaudio pymoo pytorch-msssim pillow matplotlib numpy
# Ensure ffmpeg is installed for MP3 support
```

## Usage

```bash
# Basic run (defaults to High-Res 128x256)
python run_optimization.py --image data/input/monalisa.jpg --audio data/input/target_ambient.wav

# Customize generation
python run_optimization.py \
    --image data/input/monalisa.jpg \
    --audio data/input/target_ambient.wav \
    --generations 100 \
    --pop-size 50 \
    --grid-height 128 \
    --grid-width 256 \
    --output data/output/my_experiment
```

## How It Works

1.  **Inputs**: Target Image (Visual Goal) + Target Audio (Auditory Goal, Phase Source).
2.  **Genotype**: A 128x256 grid of values [0, 1].
3.  **Phenotype Construction**:
    *   Upsample grid to full spectrogram size (1025 x 1292).
    *   **Gaussian Blur** the mask to ensure smoothness.
    *   **Log-Blend**: `Mixed_Log = Mask * Image_Log + (1-Mask) * Audio_Log`.
    *   **ISTFT**: Reconstruct audio using `Mixed_Mag` and `Target_Audio_Phase`.
4.  **Objectives**:
    *   **Matches Image**: `1 - SSIM(Mixed_Log, Target_Image)`
    *   **Matches Audio**: `L1_Loss(Mixed_Log, Target_Audio_Log)`
5.  **Output**: A morphing video and a set of diverse audio files.

## Output Files

-   `pareto_morph.gif`: **The Hero Output**. A video showing the smooth transformation from Audio to Image.
-   `best_visual.wav`: The solution closest to the image (Visual Loss ≈ 0).
-   `best_musical.wav`: The solution closest to the audio (Audio Loss ≈ 0).
-   `pareto_front.png`: Plot of the trade-off curve.
-   `results.npy`: Raw data of the Pareto front.

## License

MIT
