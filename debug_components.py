import torch
from src.audio_utils import preprocess_image
from src.audio_encoder import MaskEncoder
from src import config


def debug_system(image_path, audio_path):
    print("=== DEBUGGING COMPONENT ISOLATION ===")

    # 1. Verify Image Loading
    print(f"\n[1] Checking Image Loading: {image_path}")
    img_tensor = preprocess_image(image_path)
    print(f"  Shape: {img_tensor.shape}")
    print(f"  Min: {img_tensor.min():.4f}")
    print(f"  Max: {img_tensor.max():.4f}")
    print(f"  Mean: {img_tensor.mean():.4f}")

    if img_tensor.max() < 0.01:
        print("  CRITICAL: Image is effectively BLACK/EMPTY.")
    else:
        print("  Image loading looks OK (not black).")

    # 2. Verify Audio Loading & Encoder Logic
    print(f"\n[2] Checking MaskEncoder Gain Logic: {audio_path}")
    # Mock config
    config.DEVICE = "cpu"

    # Instantiate properly: requires TENSOR, then PATH
    encoder = MaskEncoder(img_tensor, audio_path, device="cpu")

    # Inspect internal state
    print(f"  Audio Log Max (State): {encoder.audio_log.max():.4f}")

    # We can't easily access the local variable 'headroom_nat' from __init__,
    # but we can deduce it from the mapping.
    # image_log = img_01 * (ceil - floor) + floor
    # We can reverse engineer where 'White' (1.0) maps to.

    # Let's create a fake white image and see where it maps
    dummy_white = torch.ones_like(encoder.image_log)
    # The encoder stored 'image_log' which IS the mapped image.
    # So max(image_log) tells us exactly how loud the white pixels are.

    mapped_white_level = encoder.image_log.max()
    print(f"  Mapped Image White Level: {mapped_white_level:.4f}")

    # Compare with Audio Max
    gap = encoder.audio_log.max() - mapped_white_level
    print(f"  Audio Peak vs Image White Gap: {gap:.4f} nat")

    if gap > 2.0:
        print(
            "  CRITICAL: Image is >2.0 nat (~17dB) quieter than Audio Peak. GAIN BOOST FAILED."
        )
    elif gap < 0:
        print("  WARNING: Image is louder than Audio Peak (Risk of Clipping).")
    else:
        print(
            "  SUCCESS: Image is within 2.0 nat of Audio Peak. Gain Boost is WORKING."
        )


if __name__ == "__main__":
    debug_system("data/input/monalisa.jpg", "data/input/06 - III Allegro assai.flac")
