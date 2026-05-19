# ============================================================
#  services/dataset_collector.py  –  Kendi veri havuzu için örnek toplama
# ============================================================
import json
import os
import time
from datetime import datetime

import cv2

import config


class DatasetCollector:
    """
    Yanlış/şüpheli örnekleri ham görüntü + metadata olarak kaydeder.
    Bu klasör sonradan LabelImg/Roboflow/CVAT ile etiketlenip fine-tune için kullanılır.
    """

    def __init__(self):
        self.root = getattr(config, "DATASET_REVIEW_DIR", "dataset_review")
        self.images_dir = os.path.join(self.root, "images")
        self.meta_dir = os.path.join(self.root, "metadata")
        self.auto_enabled = getattr(config, "AUTO_SAVE_HARD_EXAMPLES", True)
        self.cooldown = getattr(config, "HARD_EXAMPLE_COOLDOWN", 8.0)
        self.last_saved = {}

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    def save_manual(self, frame, results: list[dict], detections: list[dict], reason: str = "manual") -> str:
        return self._save(frame, results, detections, reason)

    # ------------------------------------------------------------------ #
    def maybe_save_hard_example(self, frame, results: list[dict], detections: list[dict]) -> str | None:
        if not self.auto_enabled:
            return None

        reasons = []
        for result in results:
            fallback_sources = result.get("fallback_sources", {})
            if fallback_sources:
                reasons.append(f"fallback:{','.join(fallback_sources.keys())}")

        if not reasons:
            return None

        reason = "+".join(sorted(set(reasons)))
        now = time.time()
        last = self.last_saved.get(reason, 0)
        if now - last < self.cooldown:
            return None

        self.last_saved[reason] = now
        return self._save(frame, results, detections, reason)

    # ------------------------------------------------------------------ #
    def _save(self, frame, results: list[dict], detections: list[dict], reason: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in reason)
        stem = f"{timestamp}_{safe_reason}"
        image_path = os.path.join(self.images_dir, f"{stem}.jpg")
        meta_path = os.path.join(self.meta_dir, f"{stem}.json")

        cv2.imwrite(image_path, frame)

        metadata = {
            "timestamp": timestamp,
            "reason": reason,
            "detections": detections,
            "results": results,
            "note": "Review and label this image before training. Auto metadata is not a ground-truth label.",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"[DatasetCollector] Ornek kaydedildi: {image_path}")
        return image_path
