# ============================================================
#  services/detector.py  –  YOLO tabanlı PPE tespiti
# ============================================================
from ultralytics import YOLO
import config


class PPEDetector:
    """
    YOLO modelini yükler ve bir kare üzerinde nesne tespiti yapar.

    Dönen yapı (her eleman):
        {
            "class_name": str,   # örn. "helmet", "person", "vest" …
            "confidence": float,
            "box": [x1, y1, x2, y2]  # piksel koordinatları
        }
    """

    def __init__(self, model_path: str = config.MODEL_PATH):
        print(f"[PPEDetector] Model yükleniyor: {model_path}")
        self.model = YOLO(model_path)
        print(f"[PPEDetector] Sınıflar: {self.model.names}")

    # ------------------------------------------------------------------ #
    def detect(self, frame) -> list[dict]:
        """
        Verilen BGR kare üzerinde çıkarım yapar.
        Güven eşiğini genel olarak en düşük eşiğe (GLOVE_CONF) ayarla;
        sınıfa özel eşikler assign_equipment_to_persons içinde uygulanır.
        """
        results = self.model(frame, verbose=False, conf=config.GLOVE_CONF)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                xyxy       = box.xyxy[0].tolist()          # [x1,y1,x2,y2]
                class_name = self.model.names[cls_id].lower()

                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "box": xyxy,
                })

        return detections

    # ------------------------------------------------------------------ #
    @staticmethod
    def filter_by_class_conf(detections: list[dict]) -> list[dict]:
        """
        Sınıfa özel güven eşiği uygular.
        Model 'no-helmet', 'no-vest' gibi negatif sınıflar içerebilir;
        bunlar burada ayrıca işlenmez – rule_engine üstlenir.
        """
        thresholds = {
            "person": config.PERSON_CONF,
            "helmet": config.HELMET_CONF,
            "hardhat": config.HELMET_CONF,
            "vest":   config.VEST_CONF,
            "safety vest": config.VEST_CONF,
            "mask":   config.MASK_CONF,
            "glove":  config.GLOVE_CONF,
        }
        filtered = []
        for d in detections:
            cls  = d["class_name"]
            conf = d["confidence"]
            # Eşik tablosunda yoksa (bilinmeyen sınıf) genel GLOVE_CONF uygula
            thr = thresholds.get(cls, config.GLOVE_CONF)
            if conf >= thr:
                filtered.append(d)
        return filtered
