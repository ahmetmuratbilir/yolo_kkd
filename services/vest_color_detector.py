# ============================================================
#  services/vest_color_detector.py  –  HSV yelek renk fallback'i
# ============================================================
import cv2
import numpy as np

import config
from services.glove_color_detector import crop_region


def vest_detected_by_color(frame, torso_box: list) -> bool:
    """
    Torso bölgesinde yüksek görünürlüklü yelek renklerini arar.
    Model yeleği kaçırdığında pratik bir VAR fallback'i sağlar.
    """
    region = crop_region(frame, torso_box)
    if region is None:
        return False

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    combined_mask = None

    for lower, upper in getattr(config, "VEST_COLOR_RANGES", []):
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

    if combined_mask is None:
        return False

    color_pixels = cv2.countNonZero(combined_mask)
    total_pixels = region.shape[0] * region.shape[1]
    ratio = color_pixels / total_pixels if total_pixels > 0 else 0.0
    return ratio >= getattr(config, "VEST_COLOR_RATIO", 0.10)
