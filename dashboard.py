import streamlit as st
import numpy as np
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px
import io
import pandas as pd
import scipy.io.wavfile
import imageio.v2 as iio

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

selection = st.plotly_chart(fig, width="stretch", on_select="rerun")
selected_points = selection.get("selection", {}).get("points", [])

selected_idx = 0
if selected_points:
    # Handle different Plotly selection structures
    try:
        # Sometimes it's pointIndex (standard plotly.js), sometimes point_index (python wrapper)
        if "pointIndex" in selected_points[0]:
            selected_idx = selected_points[0]["pointIndex"]
        elif "point_index" in selected_points[0]:
            selected_idx = selected_points[0]["point_index"]
        # If hover_data was passed, customdata might be useful, but index is standard
        else:
            st.warning(f"Unknown selection format: {selected_points[0].keys()}")
    except Exception as e:
        st.error(f"Selection Error: {e}")

# Fallback / Default if selection failed or was empty
if not selected_points:
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
        # Standard Library Transformation
        import torchaudio.transforms as T

        # turn off internal max-based thresholding (top_db=None)
        # so we can handle outliers manually
        to_db = T.AmplitudeToDB(stype="magnitude", top_db=None)

        # mixed_mag is [1, F, T]
        mag_db_tensor = to_db(mixed_mag[0].cpu())
        mag_db = mag_db_tensor.numpy()

        # Robust Scaling: Ignore top 0.5% outliers (transients)
        # to prevent them from crushing the rest of the image to black.
        ref_max = np.percentile(mag_db, 99.5)
        vmin = ref_max - 80
        vmax = ref_max

        fig_spec, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(
            mag_db, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax
        )
        ax.set_title("Reconstructed Spectrogram (Robust Scaling)")
        ax.axis("off")
        st.pyplot(fig_spec)

    with col2:
        st.subheader("Audio")
        # Convert Wav to Bytes for Streamlit
        wav_cpu = rec_wav.squeeze().cpu()
        wav_cpu = wav_cpu / (torch.abs(wav_cpu).max() + 1e-6)
        wav_int16 = (wav_cpu.numpy() * 32767).astype(np.int16)

        sub_sampled = wav_cpu[::50].numpy()
        fig_aud = px.line(sub_sampled, title="Audio Waveform (Subsampled)")
        # FIX: Do NOT use 'selection' here, and do NOT rerun. Just display audio.
        st.plotly_chart(fig_aud, width="stretch")

        buf_wav = io.BytesIO()
        scipy.io.wavfile.write(buf_wav, config.SAMPLE_RATE, wav_int16)
        st.audio(buf_wav, format="audio/wav")

    # 6. Video Generation
    if selection and selected_points:
        st.info("Generating video preview...")

        # 6. Generate Video with Scrolling Line
        # 6. Generate Video with Scrolling Line
        # We need to save the spectrogram image first
        buf_img = io.BytesIO()
        # Use imsave to save the raw pixels exactly (no axes/padding)
        # origin='lower' is critical to match the standard display
        # FIX: Explicitly pass vmin/vmax to match the Robust Scaling in st.pyplot
        plt.imsave(
            buf_img,
            mag_db,
            cmap="magma",
            origin="lower",
            format="png",
            vmin=vmin,
            vmax=vmax,
        )
        buf_img.seek(0)

        bg_image = iio.imread(buf_img)

        # Optimize Video Resolution & Aspect Ratio
        import cv2
        from PIL import Image

        # 1. Get Original Image Aspect Ratio
        original_img_path = metadata.get("image_path", "")
        target_aspect = 1.0  # Default square

        try:
            if original_img_path and Path(original_img_path).exists():
                with Image.open(original_img_path) as img:
                    w_orig, h_orig = img.size
                    target_aspect = w_orig / h_orig
            else:
                st.warning(
                    "Original image not found for aspect ratio correction. Using 1:1."
                )
        except Exception as e:
            print(f"Aspect Ratio Error: {e}")

        # 2. Calculate New Dimensions
        # We want the final video to have the same aspect ratio as the original image.
        # But we also want to fit standard video players (e.g. max width/height 1920 or 4K)
        # Strategy:
        # - Keep Spectrogram content (it's wide).
        # - Stretch Height so that W/H = Aspect.
        # - If H becomes huge, scale both down to fit max_dim.

        # Current raw spectrogram dims
        raw_h, raw_w, _ = bg_image.shape

        # Target Dimensions
        # new_h = raw_w / target_aspect
        # This implies we keep width and stretch height.
        # Example: 3min song -> raw_w ~ 4000px. Image 1:1. new_h = 4000px.
        # 4000x4000 is 4K video. Acceptable? Yes, typically.
        # Let's cap at Max Dimension 2160 (4K standard) to be safe for browsers.

        target_h_calc = int(raw_w / target_aspect)
        target_w_calc = raw_w

        # Define paths early to avoid NameError
        temp_dir = Path("temp_dashboard")
        temp_dir.mkdir(exist_ok=True)
        aud_path = temp_dir / "temp_aud.wav"
        out_path = temp_dir / "playback.mp4"
        img_path = temp_dir / "temp_spec.png"

        MAX_DIM = 1920  # Limit to 1080p/HD for speed/compatibility
        # If either exceeds, scale down maintaining ratio
        if target_w_calc > MAX_DIM or target_h_calc > MAX_DIM:
            scale_factor = MAX_DIM / max(target_w_calc, target_h_calc)
            target_w_calc = int(target_w_calc * scale_factor)
            target_h_calc = int(target_h_calc * scale_factor)

        # Ensure even dimensions
        if target_w_calc % 2 != 0:
            target_w_calc -= 1
        if target_h_calc % 2 != 0:
            target_h_calc -= 1

        # Resize
        bg_image = cv2.resize(
            bg_image, (target_w_calc, target_h_calc), interpolation=cv2.INTER_AREA
        )

        # Video settings
        fps = 30  # Smoother
        duration_sec = len(wav_cpu) / config.SAMPLE_RATE
        total_frames = int(duration_sec * fps)

        if bg_image.shape[2] == 4:
            # Drop alpha for video
            bg_image = bg_image[:, :, :3]

        H, W, C = bg_image.shape
        # Force even dimensions for H.264
        if H % 2 != 0:
            H -= 1
        if W % 2 != 0:
            W -= 1
        bg_image = bg_image[:H, :W, :]

        # Write temporary files
        temp_dir = Path("temp_dashboard")
        temp_dir.mkdir(exist_ok=True)
        aud_path = temp_dir / "temp_aud.wav"
        out_path = temp_dir / "playback.mp4"

        # 6. Optimized Video Generation using FFmpeg Filters
        # Instead of generating frames in Python (SLOW), we use FFmpeg's 'drawbox' filter
        # to render the scrolling line natively. This is orders of magnitude faster.

        # Save Background Image
        img_path = temp_dir / "temp_spec.png"
        cv2.imwrite(str(img_path), cv2.cvtColor(bg_image, cv2.COLOR_RGB2BGR))

        # Save Audio
        scipy.io.wavfile.write(aud_path, config.SAMPLE_RATE, wav_int16)

        # GENERATE SLIDER IMAGE (Visual Polish)
        # We use an overlay image for the slider to guarantee visibility.
        # User requested WHITE slider.
        slider_w = 6
        slider_path = temp_dir / "slider.png"
        # Create White Slider (Height = target_h_calc)
        slider_img = np.zeros((target_h_calc, slider_w, 3), dtype=np.uint8)
        slider_img[:, :] = [255, 255, 255]  # White
        cv2.imwrite(str(slider_path), slider_img)

        # FFmpeg Command
        # -hwaccel auto: Use GPU if available (User Request)
        # Inputs: 0:Bg, 1:Audio, 2:Slider
        # Filter: [0][2]overlay=x='(t/D)*TARGET_W':y=0

        cmd = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "auto",
            "-loop",
            "1",
            "-i",
            str(img_path),
            "-i",
            str(aud_path),
            "-loop",
            "1",
            "-i",
            str(slider_path),
            "-filter_complex",
            f"[0][2]overlay=x='(t/{duration_sec})*{target_w_calc}':y=0:shortest=1",
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(out_path),
        ]

        import subprocess

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        st.subheader("Video Playback")
        st.caption(f"Visualizing Pareto Solution Index: **{selected_idx}**")
        st.caption(
            "✅ Audio, Spectrogram, and Video are perfectly synchronized from the same source."
        )
        st.video(str(out_path))
else:
    st.info("Visualizations not available because `pareto_X.npy` is missing.")
