import cv2
import numpy as np
from PIL import Image, ImageFilter

# ============== VARIABLES AJUSTABLES (Correction couleurs subtile) ==============

SATURATION_THRESHOLD = 0.4
MAX_SATURATION_FACTOR = 1.3
ENABLE_WHITE_BALANCE = False
WHITE_BALANCE_SCALE_MIN = 0.97
WHITE_BALANCE_SCALE_MAX = 1.03

# =================== VARIABLES AJUSTABLES (Sharpening adaptatif) ===================

ENABLE_ADAPTIVE_SHARPENING = True
SHARPENING_INTENSITY = 0.8
SHARPENING_RADIUS = 1.2

# ===================================================================================

def measure_saturation(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls[:, :, 2].mean() / 255.0

def apply_sharpening(img: Image.Image) -> Image.Image:
    print("[Sharpening] Sharpening applied.")
    blurred = img.filter(ImageFilter.GaussianBlur(radius=SHARPENING_RADIUS))
    sharpened = Image.blend(img, blurred, alpha=-SHARPENING_INTENSITY)
    return sharpened

def apply_color_correction(img: Image.Image) -> Image.Image:
    img_array = np.array(img.convert("RGB"))
    saturation = measure_saturation(img_array)
    print(f"[ColorCorrection] Measured saturation: {saturation:.4f}")

    if saturation >= SATURATION_THRESHOLD:
        print("[ColorCorrection] Saturation above threshold, no color correction applied.")
        corrected_img = img
    else:
        print("[ColorCorrection] Saturation below threshold, applying color correction.")
        corrected_img = img_array.copy()

        if ENABLE_WHITE_BALANCE:
            print("[ColorCorrection] White balance enabled, applying correction.")
            avg_per_channel = corrected_img.reshape(-1, 3).mean(axis=0)
            avg_intensity = avg_per_channel.mean()
            scale = avg_intensity / (avg_per_channel + 1e-6)
            scale = np.clip(scale, WHITE_BALANCE_SCALE_MIN, WHITE_BALANCE_SCALE_MAX)
            corrected_img = np.clip(corrected_img * scale, 0, 255).astype(np.uint8)

        sat_factor = min(SATURATION_THRESHOLD / (saturation + 1e-6), MAX_SATURATION_FACTOR)
        print(f"[ColorCorrection] Applying saturation factor: {sat_factor:.3f}")
        hls = cv2.cvtColor(corrected_img, cv2.COLOR_RGB2HLS)
        H, L_chan, S = cv2.split(hls)
        S_boosted = np.clip(S * sat_factor, 0, 255).astype(np.uint8)
        hls_boosted = cv2.merge([H, L_chan, S_boosted])
        corrected_img = cv2.cvtColor(hls_boosted, cv2.COLOR_HLS2RGB)

        corrected_img = Image.fromarray(corrected_img)

    if ENABLE_ADAPTIVE_SHARPENING:
        corrected_img = apply_sharpening(corrected_img)

    return corrected_img
