import json
import os
import shutil
from pathlib import Path

import cv2

import config
from services.detector import PPEDetector
from services.person_filter import filter_person_boxes
from services.detection_merge import merge_detections
from services.rule_engine import (
    _assign_items_to_regions,
    _sub_region,
)
from services.vest_color_detector import vest_detected_by_color


INPUT_DIR = Path("test_images")
DATASET_DIR = Path("datasets") / "auto_ppe"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

CLASSES = ["helmet", "vest", "glove", "glasses"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def box_to_yolo(box, image_w, image_h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(image_w, float(x1)))
    y1 = max(0, min(image_h, float(y1)))
    x2 = max(0, min(image_w, float(x2)))
    y2 = max(0, min(image_h, float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    cx = ((x1 + x2) / 2) / image_w
    cy = ((y1 + y2) / 2) / image_h
    bw = (x2 - x1) / image_w
    bh = (y2 - y1) / image_h
    return cx, cy, bw, bh


def normalized_line(class_name, box, image_w, image_h):
    normalized = box_to_yolo(box, image_w, image_h)
    if normalized is None:
        return None
    cx, cy, bw, bh = normalized
    return f"{CLASS_TO_ID[class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    image_out = DATASET_DIR / "images" / "train"
    label_out = DATASET_DIR / "labels" / "train"
    meta_out = DATASET_DIR / "metadata"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

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

    image_paths = sorted(p for p in INPUT_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    summaries = []

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        h, w = frame.shape[:2]
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
        persons = filter_person_boxes([*persons, *fallback_persons], frame.shape)
        torso_regions = [_sub_region(p["box"], config.TORSO_TOP_RATIO, config.TORSO_BOTTOM_RATIO) for p in persons]

        labels = []
        label_sources = []
        for detection in dets:
            class_name = detection["class_name"]
            if class_name not in CLASS_TO_ID:
                continue
            line = normalized_line(class_name, detection["box"], w, h)
            if line:
                labels.append(line)
                label_sources.append({
                    "class_name": class_name,
                    "source": "model",
                    "box": detection["box"],
                    "confidence": detection.get("confidence"),
                })

        # Model yeleği kaçırdıysa torso bölgesinde renk fallback'i ile pseudo-label üret.
        vest_regions = _assign_items_to_regions(
            [d for d in dets if d["class_name"] == "vest"],
            torso_regions,
            getattr(config, "MIN_EQUIPMENT_OVERLAP", 0.08),
        )
        for idx, torso_box in enumerate(torso_regions):
            if idx in vest_regions:
                continue
            if vest_detected_by_color(frame, torso_box):
                line = normalized_line("vest", torso_box, w, h)
                if line:
                    labels.append(line)
                    label_sources.append({
                        "class_name": "vest",
                        "source": "color_fallback_torso",
                        "box": torso_box,
                        "confidence": None,
                    })

        out_image_path = image_out / f"{image_path.stem}.jpg"
        out_label_path = label_out / f"{image_path.stem}.txt"
        out_meta_path = meta_out / f"{image_path.stem}.json"

        shutil.copyfile(image_path, out_image_path)
        out_label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        out_meta_path.write_text(
            json.dumps({
                "image": str(image_path),
                "labels": label_sources,
                "note": "Pseudo-labels. Review manually before serious training.",
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summaries.append({
            "image": str(image_path),
            "label_count": len(labels),
            "labels": label_sources,
        })
        print(f"[auto_index] {image_path.name}: {len(labels)} etiket")

    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/train\n"
        f"names: {CLASSES}\n",
        encoding="utf-8",
    )

    (DATASET_DIR / "index_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[auto_index] Dataset hazir: {DATASET_DIR}")


if __name__ == "__main__":
    main()
