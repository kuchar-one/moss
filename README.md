# MOSS: Multi-Objective Spectral Synthesis

Encode images into audio spectrograms using Multi-Objective Optimization.

## What It Does

MOSS finds audio waveforms where:
1. **Spectrogram looks like your image** (optimized via SSIM)
2. **Audio sounds like your target sound** (optimized via multi-scale spectral loss)

The optimization explores the trade-off between visual fidelity and sound similarity, producing a Pareto front of solutions.

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch torchaudio pymoo pytorch-msssim pillow matplotlib numpy
```

## Usage

```bash
# Basic: encode image with no sound target
python run_optimization.py --image data/input/monalisa.jpg

# With target sound: balance between image and sound
python run_optimization.py --image data/input/monalisa.jpg --audio data/input/ambient.wav

# Full options
python run_optimization.py \
    --image data/input/monalisa.jpg \
    --audio data/input/ambient.wav \
    --generations 50 \
    --pop-size 50 \
    --grid-height 32 \
    --grid-width 64 \
    --output data/output/results
```

## How It Works

1. **Decision Variables**: Low-resolution spectrogram grid (32×64 = 2,048 parameters)
2. **Upsampling**: Grid is upsampled to full spectrogram resolution
3. **Audio Synthesis**: Griffin-Lim phase reconstruction
4. **Objectives**:
   - Image loss: `1 - SSIM(generated_spec, target_image)`
   - Sound loss: Multi-scale spectral distance to target audio
5. **Optimization**: NSGA-II finds Pareto-optimal solutions

## Output

- `pareto_front.png`: Trade-off curve between image and sound objectives
- `pareto_walk.png`: Visual comparison of solutions along the Pareto front
- `best_visual.wav`, `balanced_*.wav`, `best_musical.wav`: Audio files

## Project Structure

```
moss/
├── run_optimization.py    # Main entry point
├── src/
│   ├── audio_encoder.py   # Spectrogram → audio conversion
│   ├── objectives.py      # SSIM and spectral loss functions
│   ├── problem.py         # MOO problem definition
│   ├── audio_utils.py     # Audio/image preprocessing
│   ├── visualize.py       # Plotting utilities
│   └── config.py          # Global settings
└── data/
    ├── input/             # Target images and audio
    └── output/            # Generated results
```

## License

MIT
