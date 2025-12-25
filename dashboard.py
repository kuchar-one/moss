import streamlit as st
import numpy as np
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import torchaudio
import io
import pandas as pd

from src import config
from src.audio_encoder import MaskEncoder
from src.audio_utils import preprocess_image

# Page Config
st.set_page_config(page_title="MOSS Dashboard", layout="wide")

st.title("MOSS: Interactive Results Dashboard")

# 1. Select Run
output_base = Path("data/output")
if not output_base.exists():
    st.error(f"Output directory {output_base} not found.")
    st.stop()

runs = [d for d in output_base.iterdir() if d.is_dir()]
runs = sorted(runs, key=lambda x: x.stat().st_mtime, reverse=True)
run_names = [r.name for r in runs]

selected_run_name = st.sidebar.selectbox("Select Optimization Run", run_names)
run_path = output_base / selected_run_name

# 2. Load Metadata & Results
try:
    with open(run_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Load Numpy Results
    # Load Numpy Results
    F = np.load(run_path / "pareto_F.npy")

    X = None
    if (run_path / "pareto_X.npy").exists():
        X = np.load(run_path / "pareto_X.npy")  # (Pop, Param)
    else:
        st.warning(
            "`pareto_X.npy` not found. Mask reconstruction and audio playback will be disabled."
        )

except Exception as e:
    st.error(f"Error loading run data: {e}")
    st.stop()

st.sidebar.write("Metadata:")
st.sidebar.json(metadata)

# 3. Pareto Plot (Interactive)
st.subheader("Pareto Front")

# Prepare DataFrame for Plotly
df = pd.DataFrame(F, columns=["Visual Loss", "Audio Loss"])
df["Index"] = range(len(F))

# Create Plot
fig = px.scatter(
    df,
    x="Visual Loss",
    y="Audio Loss",
    hover_data=["Index"],
    title="Pareto Front (Click point to analyze)",
    color="Visual Loss",  # Color by one objective
    color_continuous_scale="Viridis_r",
)
fig.update_traces(marker_size=10)
fig.update_layout(clickmode="event+select")

# Select Point using Streamlit selection approach or just Index Selector
# Streamlit's plotly_chart specific selection API is newer (st.plotly_chart(..., on_select="rerun"))
# Let's hope user has recent streamlit. If not, use Sidebar Index Selector.

selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
selected_points = selection.get("selection", {}).get("points", [])

selected_idx = 0
if selected_points:
    selected_idx = selected_points[0]["pointIndex"]
else:
    # Fallback / Default
    selected_idx = st.sidebar.number_input(
        "Select Point Index manually", 0, len(F) - 1, 0
    )

st.write(f"**Analyzing Point Index: {selected_idx}**")
st.write(f"Visual Loss: {F[selected_idx, 0]:.4f}, Audio Loss: {F[selected_idx, 1]:.4f}")


# 4. Generate & Visualize
# Initialize Encoder (Cached)?
@st.cache_resource
def load_resources(image_path, audio_path, height, width, sigma):
    device = config.DEVICE
    target_image = preprocess_image(image_path).to(device)
    encoder = MaskEncoder(
        target_image,
        audio_path,
        grid_height=height,
        grid_width=width,
        smoothing_sigma=sigma,
        device=device,
    ).to(device)
    return encoder


encoder = load_resources(
    metadata["image_path"],
    metadata["audio_path"],
    metadata["grid_height"],
    metadata["grid_width"],
    metadata["sigma"],
)

if X is not None:
    # Decode
    input_x = X[selected_idx]  # (PopParam)
    # The X was saved as "Flattened Sigmoid Mask" in run_optimization.
    # "X = torch.sigmoid(masks_logits).flatten(1).cpu().numpy()" -> This logic in run_optimization
    # SO X IS ALREADY SIGMOIDED!
    # But Encoder expects LOGITS? Or Masks?
    # "def forward(self, mask_logits, return_wav=True): mask = torch.sigmoid(mask_logits)"
    # So Encoder expects LOGITS.
    # We must Inverse Sigmoid X to get Logits.
    # config says X is [0, 1].

    # Careful: Inverse Sigmoid of 0 or 1 is +/- Inf.
    # But X is sigmoid(logits), so it can't be exactly 0 or 1 unless logits are inf.
    # Optimization might push them hard.
    x_clamped = np.clip(input_x, 1e-6, 1 - 1e-6)
    logits_np = np.log(x_clamped / (1 - x_clamped))

    logits_tensor = torch.tensor(logits_np, device=config.DEVICE).float()
    # Reshape
    logits_tensor = logits_tensor.view(
        1, metadata["grid_height"], metadata["grid_width"]
    )

    with torch.no_grad():
        rec_wav, mixed_mag = encoder(logits_tensor, return_wav=True)

    # 5. Display
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Spectrogram")
        # Convert DB
        mag_db = 20 * torch.log10(mixed_mag[0] + 1e-8).cpu().numpy()

        fig_spec, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(mag_db, aspect="auto", origin="lower", cmap="magma")
        ax.set_title("Reconstructed Spectrogram")
        ax.axis("off")
        st.pyplot(fig_spec)

    with col2:
        st.subheader("Audio")
        # Convert Wav to Bytes for Streamlit
        wav_cpu = rec_wav.squeeze().cpu()
        wav_cpu = wav_cpu / (torch.abs(wav_cpu).max() + 1e-6)

        # Save to buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, wav_cpu.unsqueeze(0), config.SAMPLE_RATE, format="wav")
        buffer.seek(0)

        st.audio(buffer, format="audio/wav")

        st.info(
            "Use the Audio Player capabilities to seek. The 'Moving Line' sync is currently not supported in static mode."
        )
else:
    st.info("Visualizations not available because `pareto_X.npy` is missing.")
