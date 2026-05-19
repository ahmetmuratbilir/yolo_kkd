import json
import os
from pathlib import Path

import cv2

import config
from services.detector import PPEDetector
from services.person_filter import filter_person_boxes
from services.rule_engine import assign_equipment_to_persons
from services.drawing import draw_header, draw_person_status
from services.detection_merge import merge_detections


INPUT_DIR = Path("test_images")
OUTPUT_DIR = Path("benchmark_output")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "annotated").mkdir(exist_ok=True)

    detector = PPEDetector(config.MODEL_PATH)
    aux_detectors = []
    if getattr(config, "ENABLE_AUX_PPE_MODELS", False):
        for aux_model_path in getattr(config, "AUX_PPE_MODEL_PATHS", []):
            if aux_model_path == config.MODEL_PATH:
                continue
            if os.path.exists(aux_model_path):
                aux_detectors.append(PPEDetector(aux_model_path))

    person_detector = None
    if getattr(config, "ENABLE_PERSON_FALLBACK_MODEL", True) and os.path.exists(config.PERSON_MODEL_PATH):
        person_detector = PPEDetector(config.PERSON_MODEL_PATH)

    summaries = []
    image_paths = sorted(p for p in INPUT_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        print(f"[benchmark] Gorsel bulunamadi: {INPUT_DIR.resolve()}")
        return

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        dets = detector.filter_by_class_conf(detector.detect(frame))
        aux_dets = []
        for aux_detector in aux_detectors:
            aux_raw = aux_detector.detect(frame, conf=getattr(config, "AUX_PPE_MODEL_CONF", None))
            aux_dets.extend(aux_detector.filter_by_class_conf(aux_raw))
        if aux_dets:
            dets = merge_detections(dets, aux_dets)

        fallback_persons = []
        if person_detector:
            person_raw = person_detector.detect(frame, conf=getattr(config, "PERSON_FALLBACK_CONF", config.PERSON_CONF))
            person_dets = person_detector.filter_by_class_conf(person_raw)
            fallback_persons = [d for d in person_dets if d["class_name"] == "person"]

        persons = [d for d in dets if d["class_name"] == "person"]
        persons = [*persons, *fallback_persons]
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

        annotated = frame.copy()
        unsafe_count = sum(1 for result in results if not result["safe"])
        debug_text = (
            f"Det:{len(dets)} Aux:{len(aux_dets)} P:{len(persons)} Kask:{len(helmets_pos)} Yelek:{len(vests_pos)} "
            f"Maske:{len(masks_pos)} Gozluk:{len(glasses_pos)} Eldiven:{len(gloves_pos)}"
        )
        draw_header(annotated, len(results), unsafe_count, debug_text=debug_text)
        for result in results:
            draw_person_status(annotated, result)

        out_path = OUTPUT_DIR / "annotated" / f"{image_path.stem}_annotated.jpg"
        cv2.imwrite(str(out_path), annotated)

        summary = {
            "image": str(image_path),
            "annotated": str(out_path),
            "detections": dets,
            "aux_detections": aux_dets,
            "fallback_persons": fallback_persons,
            "results": results,
        }
        summaries.append(summary)
        print(f"[benchmark] {image_path.name}: kisi={len(results)}, yelek={len(vests_pos)} -> {out_path}")

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"[benchmark] Tamamlandi: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
