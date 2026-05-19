# ============================================================
#  services/rule_engine.py  –  Ekipman-kişi eşleştirme & uyarı motoru
# ============================================================
import config
from services.glove_color_detector import (
    glove_detected_by_color,
    make_hand_boxes_from_person,
    make_wrist_box,
)


# ── Geometri yardımcıları ────────────────────────────────────────────── #

def _center(box: list) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _inside(point: tuple, box: list) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _sub_region(person_box: list, top_ratio: float, bottom_ratio: float) -> list:
    """Kişi kutusunun dikey dilimini döner."""
    x1, y1, x2, y2 = person_box
    ph = y2 - y1
    return [x1, y1 + int(ph * top_ratio), x2, y1 + int(ph * bottom_ratio)]


# ── İskelet durum nesnesi ────────────────────────────────────────────── #

def empty_status() -> dict:
    return {
        "helmet":      False,
        "vest":        False,
        "mask":        False,
        "left_glove":  False,
        "right_glove": False,
        "warnings":    [],
        "safe":        False,
    }


# ── Ana eşleştirme fonksiyonu ────────────────────────────────────────── #

def assign_equipment_to_persons(
    frame,
    persons:  list[dict],
    helmets:  list[dict],
    vests:    list[dict],
    masks:    list[dict],
    gloves:   list[dict],
    wrists:   list[dict] = None,
) -> list[dict]:
    """
    Her kişi için KKD durumunu belirler.
    """
    results = []

    for idx, person in enumerate(persons):
        pbox   = person["box"]
        status = empty_status()

        # Bölge tanımları
        head_region  = _sub_region(pbox, 0.0, config.HEAD_REGION_RATIO)
        torso_region = _sub_region(pbox, config.TORSO_TOP_RATIO, config.TORSO_BOTTOM_RATIO)
        face_region  = _sub_region(pbox, 0.0, 0.35)

        # ── Kask ────────────────────────────────────────────────────── #
        for h in helmets:
            if _inside(_center(h["box"]), head_region):
                status["helmet"] = True
                break

        # ── Yelek ───────────────────────────────────────────────────── #
        for v in vests:
            if _inside(_center(v["box"]), torso_region):
                status["vest"] = True
                break

        # ── Maske ───────────────────────────────────────────────────── #
        for m in masks:
            if _inside(_center(m["box"]), face_region):
                status["mask"] = True
                break

        # ── Eldiven: YOLO tespiti ────────────────────────────────────── #
        yolo_gloves_in_person = [
            g for g in gloves if _inside(_center(g["box"]), pbox)
        ]

        # YOLO sol/sağ ayrımı: kişi orta x değerine göre
        person_mid_x = (pbox[0] + pbox[2]) / 2
        for g in yolo_gloves_in_person:
            gc_x = _center(g["box"])[0]
            if gc_x <= person_mid_x:
                status["left_glove"]  = True
            else:
                status["right_glove"] = True

        # ── Eldiven: renk analizi (fallback / Pose ile hassas kontrol) ── #
        # Heuristics (tahmini kutular) varsayılan olarak alınır
        left_box, right_box = make_hand_boxes_from_person(pbox)

        # Eğer MediaPipe iskelet koordinatları varsa, onları kullan
        if wrists:
            for w in wrists:
                # Bilek koordinatı bu kişinin kutusu içindeyse
                if w.get("left_wrist") and _inside(w["left_wrist"], pbox):
                    left_box = make_wrist_box(w["left_wrist"][0], w["left_wrist"][1], size=int((pbox[3] - pbox[1]) * 0.08))
                if w.get("right_wrist") and _inside(w["right_wrist"], pbox):
                    right_box = make_wrist_box(w["right_wrist"][0], w["right_wrist"][1], size=int((pbox[3] - pbox[1]) * 0.08))

        if not status["left_glove"]:
            status["left_glove"]  = glove_detected_by_color(frame, left_box)

        if not status["right_glove"]:
            status["right_glove"] = glove_detected_by_color(frame, right_box)

        # ── Uyarılar ────────────────────────────────────────────────── #
        status["warnings"] = build_warnings(status)
        status["safe"]     = len(status["warnings"]) == 0

        results.append({
            "person_id":   idx + 1,
            "box":         pbox,
            **{k: v for k, v in status.items()},
        })

    return results


# ── Uyarı mesajları ──────────────────────────────────────────────────── #

def build_warnings(status: dict) -> list[str]:
    msgs = []
    if not status.get("helmet"):      msgs.append("Kask eksik")
    if not status.get("vest"):        msgs.append("Yelek eksik")
    if not status.get("mask"):        msgs.append("Maske eksik")
    if not status.get("left_glove"):  msgs.append("Sol eldiven eksik")
    if not status.get("right_glove"): msgs.append("Sag eldiven eksik")
    return msgs
