# ============================================================
#  image_indexer.py  –  Test görsellerini otomatik indeksler
#  Her görsel için: kişi sayısı, tespit edilen ekipmanlar,
#  kask rengi bilgisi ve fallback_sources tablosunu üretir.
# ============================================================
import json
import os
import shutil
from pathlib import Path
import cv2

import config
from services.detector import PPEDetector, normalize_class_name
from services.person_filter import filter_person_boxes
from services.rule_engine import assign_equipment_to_persons
from services.detection_merge import merge_detections
from services.helmet_color_verifier import helmet_color_present

INPUT_DIR  = Path("test_images")
OUTPUT_DIR = Path("image_index_output")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Kask rengine göre alt klasörler
HELMET_COLOR_DIRS = {
    "yellow":  "kask_sari",
    "orange":  "kask_turuncu",
    "red1":    "kask_kirmizi",
    "red2":    "kask_kirmizi",
    "white":   "kask_beyaz",
    "blue":    "kask_mavi",
    "none":    "kask_tespit_yok",
    "hair_rejected": "kask_sac_reddedildi",
}

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for d in set(HELMET_COLOR_DIRS.values()):
        (OUTPUT_DIR / d).mkdir(exist_ok=True)
    (OUTPUT_DIR / "kask_yok").mkdir(exist_ok=True)

    # Modelleri yükle
    detector = PPEDetector(config.MODEL_PATH)
    vyra_detector = None
    tanish_detector = None
    for aux_path in getattr(config, "AUX_PPE_MODEL_PATHS", []):
        if not os.path.exists(aux_path):
            continue
        if "vyra" in aux_path.lower():
            vyra_detector = PPEDetector(aux_path)
        elif "tanish" in aux_path.lower():
            tanish_detector = PPEDetector(aux_path)

    image_paths = sorted(p for p in INPUT_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        print(f"[indexer] Gorsel bulunamadi: {INPUT_DIR.resolve()}")
        return

    index_records = []

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        # Tespit
        dets = detector.filter_by_class_conf(detector.detect(frame))
        aux_dets = []
        if vyra_detector:
            aux_dets.extend(vyra_detector.filter_by_class_conf(
                vyra_detector.detect(frame, conf=getattr(config, "AUX_PPE_MODEL_CONF", 0.15))
            ))
        if tanish_detector:
            aux_dets.extend(tanish_detector.filter_by_class_conf(
                tanish_detector.detect(frame, conf=getattr(config, "AUX_PPE_MODEL_CONF", 0.15))
            ))
        if aux_dets:
            dets = merge_detections(dets, aux_dets)

        persons = [d for d in dets if d["class_name"] == "person"]
        persons = filter_person_boxes(persons, frame.shape)

        helmets_pos = [d for d in dets if d["class_name"] == "helmet_pos"]
        helmets_neg = [d for d in dets if d["class_name"] == "helmet_neg"]
        vests_pos   = [d for d in dets if d["class_name"] == "vest_pos"]
        vests_neg   = [d for d in dets if d["class_name"] == "vest_neg"]
        masks_pos   = [d for d in dets if d["class_name"] == "mask_pos"]
        masks_neg   = [d for d in dets if d["class_name"] == "mask_neg"]
        glasses_pos = [d for d in dets if d["class_name"] == "goggles_pos"]
        glasses_neg = [d for d in dets if d["class_name"] == "goggles_neg"]
        gloves_pos  = [d for d in dets if d["class_name"] == "glove_pos"]
        gloves_neg  = [d for d in dets if d["class_name"] == "glove_neg"]

        results = assign_equipment_to_persons(
            frame, persons,
            helmets_pos, helmets_neg,
            vests_pos, vests_neg,
            masks_pos, masks_neg,
            gloves_pos, gloves_neg,
            glasses_pos=glasses_pos, glasses_neg=glasses_neg,
            wrists=None
        )

        # Kask rengini belirle ve sırala
        helmet_color_found = "none"
        helmet_color_ratio = 0.0

        from services.rule_engine import _sub_region
        for person in persons:
            head_box = _sub_region(person["box"], 0.0, config.HEAD_REGION_RATIO)
            _, ratio, color = helmet_color_present(frame, head_box)
            if ratio > helmet_color_ratio:
                helmet_color_ratio = ratio
                helmet_color_found = color

        # Kaynak klasörü belirle
        has_helmet_result = any(r.get("helmet") for r in results)
        has_hair_rejected = any(
            "hair_rejected" in r.get("fallback_sources", {}).get("helmet", "")
            for r in results
        )

        if has_hair_rejected:
            dest_folder = "kask_sac_reddedildi"
        elif has_helmet_result:
            dest_folder = HELMET_COLOR_DIRS.get(helmet_color_found, "kask_tespit_yok")
        else:
            dest_folder = "kask_yok"

        dest_path = OUTPUT_DIR / dest_folder / image_path.name
        shutil.copy2(str(image_path), str(dest_path))

        # İndeks kaydı
        record = {
            "image": image_path.name,
            "kisi_sayisi": len(persons),
            "kask": has_helmet_result,
            "kask_rengi": helmet_color_found,
            "kask_renk_oran": round(helmet_color_ratio, 3),
            "kask_sac_reddedildi": has_hair_rejected,
            "yelek": any(r.get("vest") for r in results),
            "sol_eldiven": any(r.get("left_glove") for r in results),
            "sag_eldiven": any(r.get("right_glove") for r in results),
            "klasor": dest_folder,
            "fallback_sources": [r.get("fallback_sources", {}) for r in results],
        }
        index_records.append(record)

        print(f"[{image_path.name}] Kisi:{len(persons)} | "
              f"Kask:{'VAR' if has_helmet_result else 'YOK'} ({helmet_color_found}/{helmet_color_ratio:.2f}) | "
              f"Yelek:{'VAR' if record['yelek'] else 'YOK'} | "
              f"-> {dest_folder}")

    # JSON indeks kaydet
    with open(OUTPUT_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)

    # Özet istatistik
    print("\n" + "="*60)
    print("=== INDEKSLEME OZETI ===")
    print(f"Toplam gorsel: {len(index_records)}")
    for folder in sorted(set(r["klasor"] for r in index_records)):
        count = sum(1 for r in index_records if r["klasor"] == folder)
        print(f"  {folder}: {count} gorsel")
    print(f"\nSonuclar: {OUTPUT_DIR / 'index.json'}")
    print(f"Siniflandirilmis gorseller: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
