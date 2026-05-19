# ============================================================
#  services/logger.py  –  Guvenlik Ihlali Gunlukleyici (Logger)
# ============================================================
import os
import time
import json
import cv2
from datetime import datetime
from typing import Optional
import config


class ViolationLogger:
    """
    Guvenlik ihlallerini tespit aninda loglar, fotograf ceker ve
    dosya sistemine kaydeder. Cooldown (soguma suresi) mekanizmasina sahiptir.
    """

    def __init__(self):
        self.enabled = getattr(config, "SAVE_ALERTS", True)
        self.alert_dir = getattr(config, "ALERT_DIR", "alerts")
        self.cooldown_duration = getattr(config, "ALERT_COOLDOWN", 5.0) # saniye
        self.last_logged = {} # {person_id: timestamp}

        if self.enabled:
            os.makedirs(self.alert_dir, exist_ok=True)
            self.log_file = os.path.join(self.alert_dir, "violations_log.json")
            print(f"[ViolationLogger] Ihlal kayitlari aktif. Klasor: {self.alert_dir}")

    # ------------------------------------------------------------------ #
    def log_violation(self, frame, person_id: int, warnings: list[str], result: Optional[dict] = None) -> bool:
        """
        Ihlal durumunda resmi kaydeder ve log dosyasina yazar.
        Cooldown suresi dolmadiysa kaydetmez.
        """
        if not self.enabled or not warnings:
            return False

        now = time.time()
        # Bu kisi icin en son ne zaman log girildigini kontrol et
        if person_id in self.last_logged:
            elapsed = now - self.last_logged[person_id]
            if elapsed < self.cooldown_duration:
                return False # Cooldown dolmamis, atla

        # Cooldown guncelle
        self.last_logged[person_id] = now

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        filename = f"violation_p{person_id}_{timestamp_str}.jpg"
        filepath = os.path.join(self.alert_dir, filename)
        crop_path = None

        # Resmi kaydet
        cv2.imwrite(filepath, frame)

        if result and result.get("box"):
            crop = self._crop_box(frame, result["box"])
            if crop is not None:
                crop_filename = f"violation_p{person_id}_{timestamp_str}_crop.jpg"
                crop_path = os.path.join(self.alert_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

        # Log kaydini hazirla
        log_entry = {
            "timestamp": date_str,
            "person_id": person_id,
            "violations": warnings,
            "raw_violations": result.get("raw_warnings", warnings) if result else warnings,
            "box": result.get("box") if result else None,
            "equipment_status": self._equipment_status(result),
            "confirmed_missing": result.get("confirmed_missing") if result else None,
            "pending_missing": result.get("pending_missing") if result else None,
            "fallback_sources": result.get("fallback_sources") if result else None,
            "image_path": filepath,
            "person_crop_path": crop_path,
        }

        # JSON dosyasina ekle
        self._write_to_json(log_entry)

        print(f"[ViolationLogger] Ihlal Kaydedildi! Kisi: {person_id}, Neden: {', '.join(warnings)} -> {filename}")
        return True

    # ------------------------------------------------------------------ #
    @staticmethod
    def _crop_box(frame, box: list):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (
            max(0, int(box[0])),
            max(0, int(box[1])),
            min(w, int(box[2])),
            min(h, int(box[3])),
        )
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _equipment_status(result: Optional[dict]) -> Optional[dict]:
        if not result:
            return None

        keys = ("helmet", "vest", "mask", "glasses", "left_glove", "right_glove")
        return {key: bool(result.get(key)) for key in keys}

    # ------------------------------------------------------------------ #
    def _write_to_json(self, entry: dict):
        """JSON log dosyasini guvenli bir sekilde gunceller."""
        data = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = []

        data.append(entry)

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
