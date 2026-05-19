# ============================================================
#  services/detection_merge.py - Coklu model kutularini birlestirme
# ============================================================
import config


def _area(box: list) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return _area([x1, y1, x2, y2])


def _iou(a: list, b: list) -> float:
    inter = _intersection_area(a, b)
    union = _area(a) + _area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def merge_detections(*detection_groups: list[dict], iou_threshold: float | None = None) -> list[dict]:
    """
    Ayni siniftaki ust uste binen kutulari tek kayda indirir.
    Birden fazla model ayni kaski/yelegi buldugunda en guvenli kutu tutulur.
    """
    threshold = getattr(config, "DETECTION_MERGE_IOU", 0.65) if iou_threshold is None else iou_threshold
    merged: list[dict] = []

    for group in detection_groups:
        for detection in group:
            class_name = detection.get("class_name")
            box = detection.get("box")
            if not class_name or not box:
                continue

            duplicate_idx = None
            for idx, existing in enumerate(merged):
                if existing.get("class_name") != class_name:
                    continue
                if _iou(existing.get("box", []), box) >= threshold:
                    duplicate_idx = idx
                    break

            if duplicate_idx is None:
                merged.append(detection)
                continue

            existing = merged[duplicate_idx]
            if detection.get("confidence", 0.0) > existing.get("confidence", 0.0):
                merged[duplicate_idx] = detection

    return merged
