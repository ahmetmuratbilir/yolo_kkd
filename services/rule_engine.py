# ============================================================
#  services/rule_engine.py  –  Ekipman-kişi eşleştirme & uyarı motoru
# ============================================================
import config
from typing import Optional
from services.glove_color_detector import (
    glove_detected_by_color,
    make_hand_boxes_from_person,
    make_wrist_box,
)
from services.vest_color_detector import vest_detected_by_color, vest_color_ratio
from services.helmet_color_verifier import helmet_color_present


WARNING_LABELS = {
    "helmet":      "Kask eksik",
    "vest":        "Yelek eksik",
    "mask":        "Maske eksik",
    "glasses":     "Gozluk eksik",
    "left_glove":  "Sol eldiven eksik",
    "right_glove": "Sag eldiven eksik",
    "smoking":     "!!! SIGARA ICIYOR !!!",
}


# ── Geometri yardımcıları ────────────────────────────────────────────── #

def _center(box: list) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _inside(point: tuple, box: list) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def _area(box: list) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_area(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    return _area([x1, y1, x2, y2])


def _overlap_ratio(item_box: list, region_box: list) -> float:
    item_area = _area(item_box)
    if item_area <= 0:
        return 0.0
    return _intersection_area(item_box, region_box) / item_area


def _sub_region(person_box: list, top_ratio: float, bottom_ratio: float) -> list:
    """Kişi kutusunun dikey dilimini döner."""
    x1, y1, x2, y2 = person_box
    ph = y2 - y1
    return [x1, y1 + int(ph * top_ratio), x2, y1 + int(ph * bottom_ratio)]


def _region_match_score(item_box: list, region_box: list, min_overlap: float) -> float:
    if not region_box or region_box == [0, 0, 0, 0]:
        return 0.0
    overlap = _overlap_ratio(item_box, region_box)
    if _inside(_center(item_box), region_box):
        return 1.0 + overlap
    if overlap >= min_overlap:
        return overlap
    return 0.0


# ── Durum yardımcıları ───────────────────────────────────────────────── #

def _is_required(equipment: str) -> bool:
    return getattr(config, "REQUIRED_EQUIPMENTS", {}).get(equipment, True)


def _required_map() -> dict:
    return {
        "helmet": _is_required("helmet"),
        "vest": _is_required("vest"),
        "mask": _is_required("mask"),
        "glasses": _is_required("glasses"),
        "left_glove": _is_required("left_glove"),
        "right_glove": _is_required("right_glove"),
    }


def empty_status() -> dict:
    return {
        "helmet": False,
        "vest": False,
        "mask": False,
        "glasses": False,
        "left_glove": False,
        "right_glove": False,
        "smoking": False,
        "warnings": [],
        "safe": False,
        "required": _required_map(),
        "fallback_sources": {},
    }


def _assign_items_to_regions(items: list[dict], regions: list[list], min_overlap: float) -> dict[int, list[dict]]:
    """
    Her ekipman kutusunu en iyi eşleşen kişiye atar.
    Dönüş: person_idx -> [atanan öğeler listesi]
    """
    assigned_persons = {}

    for item in items:
        best_idx = None
        best_score = 0.0
        for idx, region in enumerate(regions):
            score = _region_match_score(item["box"], region, min_overlap)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None and best_score > 0:
            assigned_persons.setdefault(best_idx, []).append(item)

    return assigned_persons


def _hand_boxes_for_person(person_box: list, wrists: Optional[list[dict]]) -> tuple[list, list]:
    use_mediapipe = getattr(config, "ENABLE_MEDIAPIPE", False)
    
    if use_mediapipe:
        left_box = [0, 0, 0, 0]
        right_box = [0, 0, 0, 0]
        if not wrists:
            return left_box, right_box
    else:
        left_box, right_box = make_hand_boxes_from_person(person_box)
        if not wrists:
            return left_box, right_box

    wrist_box_size = max(24, int((person_box[3] - person_box[1]) * 0.08))
    for wrist in wrists:
        if wrist.get("left_wrist") and _inside(wrist["left_wrist"], person_box):
            left_box = make_wrist_box(
                wrist["left_wrist"][0],
                wrist["left_wrist"][1],
                size=wrist_box_size,
            )
        if wrist.get("right_wrist") and _inside(wrist["right_wrist"], person_box):
            right_box = make_wrist_box(
                wrist["right_wrist"][0],
                wrist["right_wrist"][1],
                size=wrist_box_size,
            )

    return left_box, right_box


def _assign_gloves_to_hands(gloves: list[dict], hand_targets: list[tuple[int, str, list]]) -> dict[tuple[int, str], list[dict]]:
    assigned_hands = {}
    min_overlap = getattr(config, "MIN_GLOVE_OVERLAP", 0.05)

    for glove in gloves:
        best_target = None
        best_score = 0.0
        for target in hand_targets:
            _, _, hand_box = target
            score = _region_match_score(glove["box"], hand_box, min_overlap)
            if score > best_score:
                best_score = score
                best_target = target

        if best_target is not None and best_score > 0:
            person_idx, hand_key, _ = best_target
            assigned_hands.setdefault((person_idx, hand_key), []).append(glove)

    return assigned_hands


# ── Karar Algoritmaları (Önceliklendirme) ────────────────────────────── #

def _resolve_status_helmet_vest(pos_items: list[dict], neg_items: list[dict]) -> tuple[bool, str]:
    if not pos_items and not neg_items:
        return False, "default_missing"
    
    # ppe_model her ikisini de görebilir (hem kask hem no-kask).
    ppe_pos = [item for item in pos_items if "ppe_model" in item.get("model_path", "")]
    ppe_neg = [item for item in neg_items if "ppe_model" in item.get("model_path", "")]
    
    # Eger ppe_model "Kask Yok (no-hardhat)" dendiyse, false-positive kaski ezsin.
    # Güvenlik önceliklidir: Model şüpheye düşüp "yok" dediyse YOK kabul et.
    if ppe_neg:
        # Eğer Kask VAR'ın confidence'ı olağanüstü yüksek değilse (örn. >0.80), YOK kararını dinle
        max_pos_conf = max((item.get("confidence", 0) for item in ppe_pos), default=0)
        max_neg_conf = max((item.get("confidence", 0) for item in ppe_neg), default=0)
        
        if max_neg_conf >= max_pos_conf * 0.7:  # Negatif güveni, pozitifin %70'ine ulaşıyorsa reddet
            return False, f"ppe_model_negative({max_neg_conf:.2f})"
            
    if ppe_pos:
        return True, "ppe_model"
        
    has_vyra_pos = any("vyra" in item.get("model_path", "") for item in pos_items)
    if has_vyra_pos:
        return True, "vyra"
        
    has_vyra_neg = any("vyra" in item.get("model_path", "") for item in neg_items)
    if has_vyra_neg:
        return False, "vyra_negative"
        
    # Varsayılan
    return True, "other_positive" if pos_items else "default_missing"

def _resolve_status_gloves(pos_items: list[dict], neg_items: list[dict], hsv_positive: bool) -> tuple[bool, str]:
    has_tanish_pos = any("tanish" in item.get("model_path", "") for item in pos_items)
    has_vyra_pos = any("vyra" in item.get("model_path", "") for item in pos_items)
    
    if has_tanish_pos: return True, "tanish"
    if has_vyra_pos: return True, "vyra"
    
    if hsv_positive:
        return True, "hsv_color"
        
    has_vyra_neg = any("vyra" in item.get("model_path", "") for item in neg_items)
    if has_vyra_neg:
        return False, "vyra_negative"
        
    if pos_items: return True, "other_positive"
    return False, "default_missing"

def _resolve_status_goggles(pos_items: list[dict], neg_items: list[dict]) -> tuple[bool, str]:
    has_vyra_pos = any("vyra" in item.get("model_path", "") for item in pos_items)
    has_tanish_pos = any("tanish" in item.get("model_path", "") for item in pos_items)
    
    if has_vyra_pos: return True, "vyra"
    if has_tanish_pos: return True, "tanish"
    
    has_vyra_neg = any("vyra" in item.get("model_path", "") for item in neg_items)
    if has_vyra_neg:
        return False, "vyra_negative"
        
    if pos_items: return True, "other_positive"
    return False, "default_missing"

def _resolve_status_mask(pos_items: list[dict], neg_items: list[dict]) -> tuple[bool, str]:
    if pos_items: return True, pos_items[0].get("model_path", "unknown").split("/")[-1].split("\\")[-1]
    if neg_items: return False, neg_items[0].get("model_path", "unknown").split("/")[-1].split("\\")[-1] + "_negative"
    return False, "default_missing"


# ── Ana eşleştirme fonksiyonu ────────────────────────────────────────── #

def assign_equipment_to_persons(
    frame,
    persons: list[dict],
    helmets_pos: list[dict], helmets_neg: list[dict],
    vests_pos: list[dict], vests_neg: list[dict],
    masks_pos: list[dict], masks_neg: list[dict],
    gloves_pos: list[dict], gloves_neg: list[dict],
    glasses_pos: list[dict] = None, glasses_neg: list[dict] = None,
    wrists: list[dict] = None,
    smokings: list[dict] = None,
) -> list[dict]:
    """Her kişi için KKD durumunu belirler."""
    glasses_pos = glasses_pos or []
    glasses_neg = glasses_neg or []
    smokings    = smokings or []
    min_overlap = getattr(config, "MIN_EQUIPMENT_OVERLAP", 0.08)

    head_regions = [_sub_region(p["box"], 0.0, config.HEAD_REGION_RATIO) for p in persons]
    torso_regions = [_sub_region(p["box"], config.TORSO_TOP_RATIO, config.TORSO_BOTTOM_RATIO) for p in persons]
    face_regions = [_sub_region(p["box"], 0.0, 0.35) for p in persons]
    eye_regions = [_sub_region(p["box"], 0.05, 0.28) for p in persons]

    helmet_pos_hits = _assign_items_to_regions(helmets_pos, head_regions, min_overlap)
    helmet_neg_hits = _assign_items_to_regions(helmets_neg, head_regions, min_overlap)
    
    vest_pos_hits = _assign_items_to_regions(vests_pos, torso_regions, min_overlap)
    vest_neg_hits = _assign_items_to_regions(vests_neg, torso_regions, min_overlap)
    
    mask_pos_hits = _assign_items_to_regions(masks_pos, face_regions, min_overlap)
    mask_neg_hits = _assign_items_to_regions(masks_neg, face_regions, min_overlap)
    
    glasses_pos_hits = _assign_items_to_regions(glasses_pos, eye_regions, min_overlap)
    glasses_neg_hits = _assign_items_to_regions(glasses_neg, eye_regions, min_overlap)

    # Sigara: kisi kutusunun tamami ile eslestirilir
    smoking_hits = _assign_items_to_regions(smokings, [p["box"] for p in persons], 0.05)

    hand_boxes = [_hand_boxes_for_person(p["box"], wrists) for p in persons]
    hand_targets = []
    for idx, (left_box, right_box) in enumerate(hand_boxes):
        hand_targets.append((idx, "left_glove", left_box))
        hand_targets.append((idx, "right_glove", right_box))

    glove_pos_hits = _assign_gloves_to_hands(gloves_pos, hand_targets)
    glove_neg_hits = _assign_gloves_to_hands(gloves_neg, hand_targets)

    results = []
    for idx, person in enumerate(persons):
        pbox = person["box"]
        status = empty_status()
        left_box, right_box = hand_boxes[idx]
        
        # Kararları çöz
        status["helmet"], status["fallback_sources"]["helmet"] = _resolve_status_helmet_vest(
            helmet_pos_hits.get(idx, []), helmet_neg_hits.get(idx, [])
        )

        # Kask renk doğrulanması: Model "kask var" dese bile,
        # kafa bölgesinde gerçek İSG kaski rengi yoksa (saç, cilt vs.) reddet.
        if status["helmet"] and getattr(config, "ENABLE_HELMET_COLOR_VERIFY", True):
            verified, color_ratio_val, color_name = helmet_color_present(frame, head_regions[idx])
            if not verified:
                status["helmet"] = False
                prev_src = status["fallback_sources"]["helmet"]
                status["fallback_sources"]["helmet"] = (
                    f"hair_rejected({prev_src},best_color={color_name},{color_ratio_val:.2f})"
                )
            else:
                status["fallback_sources"]["helmet"] += f"+color_ok({color_name},{color_ratio_val:.2f})"
        
        status["vest"], status["fallback_sources"]["vest"] = _resolve_status_helmet_vest(
            vest_pos_hits.get(idx, []), vest_neg_hits.get(idx, [])
        )
        
        status["mask"], status["fallback_sources"]["mask"] = _resolve_status_mask(
            mask_pos_hits.get(idx, []), mask_neg_hits.get(idx, [])
        )
        
        status["glasses"], status["fallback_sources"]["glasses"] = _resolve_status_goggles(
            glasses_pos_hits.get(idx, []), glasses_neg_hits.get(idx, [])
        )

        # Vest renk kontrolü – Önemli kural:
        # VEST_COLOR_REQUIRES_MODEL_CONFIRM = True ise renk tek başına yelek VAR yapmaz.
        # Sadece model daha önce vest_pos görmediyse ve renk de yoksa reddedilir.
        if getattr(config, "ENABLE_VEST_COLOR_FALLBACK", True) and _is_required("vest"):
            color_r = vest_color_ratio(frame, torso_regions[idx])
            vest_threshold = getattr(config, "VEST_COLOR_RATIO", 0.10)
            color_detected = color_r >= vest_threshold
            requires_model = getattr(config, "VEST_COLOR_REQUIRES_MODEL_CONFIRM", True)

            if color_detected and not status["vest"]:
                if requires_model:
                    # Renk var ama model görmedı → reddedildi, logla
                    status["fallback_sources"]["vest"] = f"color_only_rejected(ratio={color_r:.2f})"
                else:
                    # Eski davranış: renk yeterli → VAR
                    status["vest"] = True
                    status["fallback_sources"]["vest"] = f"color_fallback(ratio={color_r:.2f})"

        # Eldiven kararları (Renk analizi HSV de dahil edilir)
        use_color_fallback = getattr(config, "ENABLE_GLOVE_COLOR_FALLBACK", False)
        
        # Sol eldiven HSV
        left_hsv = False
        if use_color_fallback and _is_required("left_glove"):
            left_hsv = glove_detected_by_color(frame, left_box)
            
        status["left_glove"], status["fallback_sources"]["left_glove"] = _resolve_status_gloves(
            glove_pos_hits.get((idx, "left_glove"), []), 
            glove_neg_hits.get((idx, "left_glove"), []),
            left_hsv
        )

        # Sağ eldiven HSV
        right_hsv = False
        if use_color_fallback and _is_required("right_glove"):
            right_hsv = glove_detected_by_color(frame, right_box)
            
        status["right_glove"], status["fallback_sources"]["right_glove"] = _resolve_status_gloves(
            glove_pos_hits.get((idx, "right_glove"), []), 
            glove_neg_hits.get((idx, "right_glove"), []),
            right_hsv
        )

        # Sigara tespiti
        if smoking_hits.get(idx):
            status["smoking"] = True

        status["warnings"] = build_warnings(status)
        status["safe"] = len(status["warnings"]) == 0

        person_id = person.get("track_id", idx + 1)
        results.append({
            "person_id": person_id,
            "track_id": person.get("track_id"),
            "box": pbox,
            "confidence": person.get("confidence"),
            "hand_boxes": {
                "left_glove": left_box,
                "right_glove": right_box,
            },
            **status,
        })

    return results


# ── Uyarı mesajları ──────────────────────────────────────────────────── #

def build_warnings(status: dict) -> list[str]:
    msgs = []
    for equipment, label in WARNING_LABELS.items():
        if equipment == "smoking":
            if status.get("smoking"):
                msgs.append(label)
        elif _is_required(equipment) and not status.get(equipment):
            msgs.append(label)
    return msgs
