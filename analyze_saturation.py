import numpy as np
from PIL import Image
import os

# Specific uploaded files based on user metadata
img_paths = [
    "/home/vojtech/.gemini/antigravity/brain/336c1258-d34c-48f5-a77e-68e5dfdf5018/uploaded_image_0_1766862537230.png",
    "/home/vojtech/.gemini/antigravity/brain/336c1258-d34c-48f5-a77e-68e5dfdf5018/uploaded_image_1_1766862537230.png",
]

for p in img_paths:
    if not os.path.exists(p):
        print(f"File not found: {p}")
        continue

    print(f"--- Analyzing {os.path.basename(p)} ---")
    img = Image.open(p)
    data = np.array(img)

    # Check for saturation (255 or near max)
    # data is likely RGBA or RGB
    if data.shape[-1] >= 3:
        # Luminance
        lum = 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]
    else:
        lum = data

    print(f"Max Value: {lum.max()}")
    print(f"Mean Value: {lum.mean()}")
    print(
        f"Saturated Pixels (>250): {np.sum(lum > 250)} / {lum.size} ({np.sum(lum > 250) / lum.size:.2%})"
    )

    # Check if there are large blocks of exact 255
    unique, counts = np.unique(data, return_counts=True)
    # Print top 5 colors
    top_indices = np.argsort(counts)[-5:]
    print("Top 5 pixel values:")
    for i in top_indices:
        print(f"  Val: {unique[i]}, Count: {counts[i]}")
