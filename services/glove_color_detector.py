# ============================================================
#  services/glove_color_detector.py  –  HSV eldiven renk analizi
# ============================================================
import cv2
import numpy as np
import config


def crop_region(frame, box: list) -> np.ndarray | None:
    """
    Verilen [x1, y1, x2, y2] kutusunu görüntüden kırpar.
    Sınır dışına çıkmamak için clip uygular.
    Çok küçük bölge ise None döner.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(box[0])),
        max(0, int(box[1])),
        min(w, int(box[2])),
        min(h, int(box[3])),
    )
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return frame[y1:y2, x1:x2]


def color_ratio(region: np.ndarray, lower: list, upper: list) -> float:
    """
    BGR bölgeyi HSV'ye çevirir.
    [lower, upper] aralığındaki piksellerin toplam piksele oranını döner (0–1).
    """
    if region is None or region.size == 0:
        return 0.0

    hsv  = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    lo   = np.array(lower, dtype=np.uint8)
    hi   = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    colored = cv2.countNonZero(mask)
    total   = region.shape[0] * region.shape[1]
    return colored / total if total > 0 else 0.0


def glove_detected_by_color(frame, hand_box: list) -> bool:
    """
    El/bilek bölgesini kırpar ve tüm tanımlı eldiven renk aralıklarını dener.
    Herhangi biri GLOVE_COLOR_RATIO eşiğini geçerse True döner.
    """
    region = crop_region(frame, hand_box)
    if region is None:
        return False

    for lower, upper in config.GLOVE_COLOR_RANGES:
        ratio = color_ratio(region, lower, upper)
        if ratio >= config.GLOVE_COLOR_RATIO:
            return True
    return False


def make_hand_boxes_from_person(person_box: list) -> tuple[list, list]:
    """
    Kişi bounding box'ından yaklaşık sol ve sağ el bölgelerini çıkarır.
    Pose modeli yokken kullanılan fallback yöntem.

    Dönüş: (sol_el_kutusu, sağ_el_kutusu)
    """
    x1, y1, x2, y2 = person_box
    pw = x2 - x1   # kişi genişliği
    ph = y2 - y1   # kişi yüksekliği

    hand_top    = y1 + int(ph * config.HAND_TOP_RATIO)
    hand_bottom = y1 + int(ph * config.HAND_BOTTOM_RATIO)
    hand_width  = int(pw * config.HAND_WIDTH_RATIO)

    left_hand  = [x1,            hand_top, x1 + hand_width, hand_bottom]
    right_hand = [x2 - hand_width, hand_top, x2,            hand_bottom]

    return left_hand, right_hand


def make_wrist_box(wrist_x: float, wrist_y: float, size: int = 60) -> list:
    """
    Bilek merkezi etrafında kare bir kutu oluşturur.
    Pose modeli keypoint koordinatı verildiğinde kullanılır.
    """
    return [
        wrist_x - size,
        wrist_y - size,
        wrist_x + size,
        wrist_y + size,
    ]
