# ============================================================
#  services/helmet_color_verifier.py
#  Model "kask var" dese bile, kafa bölgesinde gerçek kask rengi
#  (sarı, kırmızı, turuncu, beyaz, mavi) yoksa kararı reddeder.
#  Böylece saç, cilt, arka plan gibi nesneler kask sayılmaz.
# ============================================================
import cv2
import numpy as np
import config
from services.glove_color_detector import crop_region


def helmet_color_present(frame, head_box: list) -> tuple[bool, float, str]:
    """
    Verilen kafa bölgesinde İSG kaskı rengi var mı kontrol eder.

    Dönüş:
        (dogrulandi, max_oran, renk_adi)
        - dogrulandi: True ise kask rengi yeterli
        - max_oran  : bulunan en yüksek renk oranı (0–1)
        - renk_adi  : hangi renk aralığında bulunduğu
    """
    if not getattr(config, "ENABLE_HELMET_COLOR_VERIFY", True):
        return True, 1.0, "verify_disabled"

    region = crop_region(frame, head_box)
    if region is None:
        return False, 0.0, "no_region"

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    total = region.shape[0] * region.shape[1]
    if total == 0:
        return False, 0.0, "empty_region"

    min_ratio = getattr(config, "HELMET_COLOR_MIN_RATIO", 0.12)
    ranges = getattr(config, "HELMET_COLOR_RANGES", [])
    color_names = ["yellow", "red1", "red2", "orange", "white", "blue"]

    best_ratio = 0.0
    best_name = "none"

    for i, (lower, upper) in enumerate(ranges):
        lo = np.array(lower, dtype=np.uint8)
        hi = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo, hi)
        ratio = cv2.countNonZero(mask) / total
        if ratio > best_ratio:
            best_ratio = ratio
            best_name = color_names[i] if i < len(color_names) else f"range_{i}"

    verified = best_ratio >= min_ratio
    return verified, best_ratio, best_name
