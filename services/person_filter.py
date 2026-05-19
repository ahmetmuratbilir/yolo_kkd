# ============================================================
#  services/person_filter.py  –  Hatalı/parça kişi kutularını eleme
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


def _intersection_over_area(small_box: list, large_box: list) -> float:
    small_area = _area(small_box)
    if small_area <= 0:
        return 0.0
    return _intersection_area(small_box, large_box) / small_area


def filter_person_boxes(persons: list[dict], frame_shape) -> list[dict]:
    """
    Kol/el gibi parçaların ayrı insan sayılmasını azaltır.
    İnsan kutusu genellikle yeterli alana, yüksekliğe ve makul h/w oranına sahiptir.
    """
    if not getattr(config, "FILTER_PERSON_BOXES", True):
        return persons

    frame_h, frame_w = frame_shape[:2]
    frame_area = max(1, frame_h * frame_w)
    min_area = frame_area * getattr(config, "PERSON_MIN_AREA_RATIO", 0.015)
    min_height = frame_h * getattr(config, "PERSON_MIN_HEIGHT_RATIO", 0.22)
    min_aspect = getattr(config, "PERSON_MIN_ASPECT_RATIO", 0.65)
    max_aspect = getattr(config, "PERSON_MAX_ASPECT_RATIO", 4.50)
    duplicate_ioa = getattr(config, "PERSON_DUPLICATE_IOA", 0.70)

    candidates = []
    for person in persons:
        x1, y1, x2, y2 = person["box"]
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        aspect = height / width
        area = width * height

        if area < min_area:
            continue
        if height < min_height:
            continue
        if aspect < min_aspect or aspect > max_aspect:
            continue

        candidates.append(person)

    candidates.sort(key=lambda p: _area(p["box"]), reverse=True)
    if not candidates and persons:
        return persons

    kept = []
    for person in candidates:
        box = person["box"]
        is_duplicate_part = any(
            _intersection_over_area(box, kept_person["box"]) >= duplicate_ioa
            for kept_person in kept
        )
        if not is_duplicate_part:
            kept.append(person)

    return kept
