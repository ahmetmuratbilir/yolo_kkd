# ============================================================
#  services/logger.py  –  Guvenlik Ihlali Gunlukleyici (Logger)
# ============================================================
import os
import time
import json
import cv2
from datetime import datetime
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
    def log_violation(self, frame, person_id: int, warnings: list[str]) -> bool:
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

        # Resmi kaydet
        cv2.imwrite(filepath, frame)

        # Log kaydini hazirla
        log_entry = {
            "timestamp": date_str,
            "person_id": person_id,
            "violations": warnings,
            "image_path": filepath
        }

        # JSON dosyasina ekle
        self._write_to_json(log_entry)

        print(f"[ViolationLogger] Ihlal Kaydedildi! Kisi: {person_id}, Neden: {', '.join(warnings)} -> {filename}")
        return True

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
