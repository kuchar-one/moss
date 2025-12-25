# MOSS: Multi-Objective Sound Synthesis

**MOSS** is a system that creates "Hybrid Audio" that lies on the boundary between an **Image** and a **Sound**.

It uses **Multi-Objective Gradient Optimization** to learn a spectral mask that blends a target image with a target ambient track.
The result is a set of audio files (the Pareto Front) ranging from "Pure Music" to "Pure Image Spectrogram", with smooth morphing between them.

![Pareto Morph](data/output/sample/pareto_morph.mp4)

## Key Features

- **Mask-Based Encoding**: Optimizes a 128x256 mask to blend visual and auditory content in the time-frequency domain.
- **Parametric Morphing**: Generates a smooth video transition between audio and image.
- **Log-Domain Mixing**: Blends magnitudes in Decibels (dB) for natural fade-ins/outs.
- **Interactive Dashboard**: Explore the trade-off curve and synthesize audio on-demand via Streamlit.
- **Direct Metric Optimization**: Target specific blend ratios (e.g., 50% Image, 50% Audio).

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# OR manually:
# pip install torch torchaudio pymoo pytorch-msssim pillow matplotlib numpy imageio[ffmpeg] streamlit plotly pandas
```

## Quick Start

### 1. Run Standard Optimization
Finds the entire trade-off curve (Pareto Front) and generates a morph video.

```bash
python run_optimization.py \
    --image data/input/monalisa.jpg \
    --audio data/input/06\ -\ III\ Allegro\ assai.flac \
    --algorithm adam \
    --steps 1000 \
    --pop-size 75 \
    --batch-size 4 \
    --lr 0.05
```

### 2. Launch Interactive Dashboard
Explore the results, click on points in the plot to hear specific solutions.

```bash
streamlit run dashboard.py
```

### 3. Run Single-Point Optimization
If you only want a specific blend (e.g., "70% Image"):

```bash
# Alpha 0.0 = Pure Audio, 1.0 = Pure Image
python run_metric_opt.py \
    --image data/input/monalisa.jpg \
    --audio data/input/06\ -\ III\ Allegro\ assai.flac \
    --alpha 0.7 \
    --steps 1000
```

## How It Works

1.  **Inputs**: Target Image (Visual Goal) + Target Audio (Phase Source).
2.  **Model**: A mask $M$ (sigmoid logits) is learned.
3.  **Synthesis**:
    $$ \text{Result} = M \cdot I + (1-M) \cdot A $$
4.  **Objectives**:
    *   **Visual Loss**: L1 distance between Result and Target Image.
    *   **Audio Loss**: L1 distance between Result and Target Audio.
5.  **Output**:
    *   `pareto_X.npy`: The learned mask parameters.
    *   `pareto_morph.mp4`: Morphing video.
    *   `pareto_front.png`: Pareto plot.

## License

MIT
