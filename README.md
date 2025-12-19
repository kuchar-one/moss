# MOSS: Multi-Objective Spectral Synthesis

Generate audio that is simultaneously **visually accurate** (its spectrogram resembles a target image) and **musically pleasing** (adheres to psychoacoustic constraints) using NSGA-II multi-objective optimization with a GPU-accelerated modular synthesizer.

## Overview

This project uses:
- **TorchSynth**: GPU-accelerated modular synthesizer for fast batch audio rendering
- **Pymoo**: NSGA-II multi-objective genetic algorithm
- **PyTorch/torchaudio**: Audio processing and mel spectrogram conversion

The optimization balances two objectives:
1. **Visual Loss** (SSIM): How well does the spectrogram match the target image?
2. **Musical Loss** (Roughness): How pleasing/smooth does the audio sound?

## Installation

```bash
# Clone the repository
git clone https://github.com/kuchar-one/moss.git
cd moss

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run with a target image
python run_optimization.py --image data/input/your_image.jpg

# Quick test with fewer generations
python run_optimization.py --image data/input/test.jpg --generations 50

# Visual-only optimization (Phase 2 - creates alien noise but accurate spectrogram)
python run_optimization.py --image data/input/test.jpg --visual-only
```

### Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--image` | `-i` | Path to target image (required) | - |
| `--generations` | `-g` | Number of GA generations | 500 |
| `--pop-size` | `-p` | Population size | 100 |
| `--visual-only` | - | Single objective (visual only) | False |
| `--output` | `-o` | Output directory | data/output |
| `--batch` | - | Use batch evaluation | False |

### Outputs

- `pareto_front.png`: Scatter plot of visual vs musical loss
- `pareto_walk.png`: Spectrograms from different Pareto solutions
- `best_visual.wav`: Most visually accurate solution
- `best_musical.wav`: Most musically pleasing solution
- `balanced_*.wav`: Solutions balancing both objectives

## Project Structure

```
moss/
├── data/
│   ├── input/          # Target images
│   ├── output/         # Generated .wav files and plots
│   └── cache/          # Pre-processed tensors
├── src/
│   ├── config.py       # Global constants
│   ├── synth.py        # AmbientDrone synthesizer
│   ├── audio_utils.py  # STFT, Mel mapping
│   ├── objectives.py   # SSIM & Roughness losses
│   ├── problem.py      # Pymoo problem wrapper
│   └── visualize.py    # Plotting tools
├── run_optimization.py # Main script
└── requirements.txt
```

## How It Works

1. **Genome Encoding**: Each solution is a vector of normalized synthesizer parameters (0-1)
2. **Synthesis**: The `AmbientDrone` synth renders audio using VCOs, LFOs, ADSR envelope, and low-pass filter
3. **Spectrogram**: Audio is converted to mel spectrogram for visual comparison
4. **Evaluation**: SSIM measures visual similarity; Plomp-Levelt model measures roughness
5. **Selection**: NSGA-II evolves population to find Pareto-optimal trade-offs

## Expected Results

- **Phase 1 (Visual Only)**: Spectrogram matches image but sounds like alien noise
- **Phase 2 (Multi-Objective)**: Pareto front shows trade-off between accuracy and musicality
- **Best Balanced**: Recognizable image with smoother, drone-like audio

## License

MIT License
