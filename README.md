# MOSS: Multi-Objective Sound Synthesis

**MOSS** is a system that creates "Hybrid Audio" lying on the boundary between an **Image** and a **Sound**.

It uses **Multi-Objective Evolutionary Optimization** (NSGA-II) to learn spectral masks that blend a target image with a target audio track. The result is a Pareto front of audio files representing different trade-offs between "Pure Music" and "Pure Image Spectrogram".

## Key Features

- **Mask-Based Encoding**: Optimizes masks to blend visual and auditory content in the time-frequency domain
- **Log-Domain Mixing**: Blends magnitudes in Decibels (dB) for natural fade-ins/outs
- **Web Interface**: Modern React/Vite frontend for interactive exploration
- **Real-Time Visualization**: Watch the optimization process live with loss plots and spectrogram updates
- **Pareto Front Generation**: Generate a diverse set of solutions representing image-audio trade-offs
- **Interactive Steering**: Guide the optimization towards "Better Image" or "Better Audio"

## Project Structure

```
moss/
├── backend/          # FastAPI backend (Python)
│   ├── main.py       # API endpoints
│   └── service.py    # Optimization logic
├── frontend/         # React + Vite + TailwindCSS
│   └── src/
├── src/              # Core modules
│   ├── audio_encoder.py
│   ├── evolutionary_optimizer.py
│   ├── fitness_objectives.py
│   └── ...
├── notebooks/        # Jupyter notebooks (including Czech explanation)
├── data/input/       # Sample images and audio files
├── requirements.txt
└── start.sh          # Startup script
```

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+
- FFmpeg (for audio/video processing)

### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
cd ..
```

## Running the Application

### Option 1: Using the Startup Script (Recommended)

```bash
./start.sh
```

This launches both backend and frontend with interactive controls for restart/quit.

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Access the Interface

Open your browser at: **http://localhost:5173**

## Usage

1. **Select Resources**: Choose a target image and target audio file from the grid
2. **Start Optimization**: Click the fire button to begin
3. **Explore**:
   - Watch the spectrogram evolve in real-time
   - Click "Better Image" or "Better Audio" to steer the result
   - Click "Pareto Front" to generate a diverse set of solutions
4. **Listen**: Play the resulting hybrid audio and explore the trade-offs

## Notebooks

The `notebooks/` directory contains Jupyter notebooks explaining the MOSS algorithm:

- **`MOSS_Explained_CZ.ipynb`** - Detailed explanation in Czech with interactive examples
- **`EXECUTED_MOSS_Explained_CZ.ipynb`** - Pre-executed version with all outputs

To run the notebooks:
```bash
cd notebooks
pip install -r requirements.txt
jupyter notebook
```

## Architecture

- **Backend**: FastAPI (Python) - Handles optimization logic, audio processing, and file serving
- **Frontend**: React + Vite + TailwindCSS - Provides the interactive user interface
- **Optimization**: PyTorch-based evolutionary optimization (NSGA-II via pymoo) for multi-objective spectral mask learning

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
