import numpy as np
from PIL import Image
import os

img_path = "/home/vojtech/.gemini/antigravity/brain/336c1258-d34c-48f5-a77e-68e5dfdf5018/uploaded_image_1766687667340.jpg"

if not os.path.exists(img_path):
    print("Image not found path is wrong")
    exit()

img = Image.open(img_path)
img_np = np.array(img)

print(f"Image Shape: {img_np.shape}")
H, W, C = img_np.shape

# Analyze "Silence" (Assuming Right Edge or Top Left as user said)
# User said "silence at the end (top left)".
# This is confusing. "End" is usually right. "Top Left" is start high freq.
# Let's check both Corners.

print("\n--- Region Analysis ---")


def analyze_region(name, region):
    mean_color = region.mean(axis=(0, 1))
    std_color = region.std(axis=(0, 1))
    print(f"{name}: Mean RGB {mean_color}, Std {std_color}")
    return mean_color


# Top Left (High Freq, Start)
top_left = img_np[0:50, 0:50]
analyze_region("Top Left", top_left)

# Top Right (High Freq, End)
top_right = img_np[0:50, W - 50 : W]
analyze_region("Top Right", top_right)

# Bottom Left (Low Freq, Start)
bottom_left = img_np[H - 50 : H, 0:50]
analyze_region("Bottom Left", bottom_left)

# Bottom Right (Low Freq, End)
bottom_right = img_np[H - 50 : H, W - 50 : W]
analyze_region("Bottom Right", bottom_right)

# Middle (Signal)
middle = img_np[H // 2 - 50 : H // 2 + 50, W // 2 - 50 : W // 2 + 50]
analyze_region("Middle", middle)
