# ============================================================
#  services/detector.py  –  YOLO tabanlı PPE tespiti
# ============================================================
import re

from ultralytics import YOLO
import config


def _clean_class_name(class_name: str) -> str:
    """Modelden gelen sınıf adını karşılaştırma için sadeleştirir."""
    cleaned = class_name.lower().replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_class_name(class_name: str) -> str:
    """
    Farklı veri setlerinden gelebilecek sınıf adlarını ortak isimlere çevirir.
    Örn: "NO-Hardhat" -> "helmet_neg", "helmet" -> "helmet_pos".
    """
    class_map = getattr(config, "CLASS_MAP", {})
    
    # 1. Tam eşleşme kontrolü (küçük harf ile)
    raw = class_name.lower()
    if raw in class_map:
        return class_map[raw]
        
    # 2. Temizlenmiş (boşluklu) eşleşme kontrolü
    cleaned = _clean_class_name(class_name)
    if cleaned in class_map:
        return class_map[cleaned]
        
    # 3. Bulunamadıysa tireleri altçizgi yap
    return cleaned.replace(" ", "_")


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
        self.model_path = model_path
        print(f"[PPEDetector] Model yükleniyor: {model_path}")
        self.model = YOLO(model_path)
        print(f"[PPEDetector] Sınıflar: {self.model.names}")

    # ------------------------------------------------------------------ #
    def detect(self, frame, conf: float = None) -> list[dict]:
        """
        Verilen BGR kare üzerinde çıkarım yapar.
        Güven eşiğini genel olarak en düşük eşiğe (GLOVE_CONF) ayarla;
        sınıfa özel eşikler assign_equipment_to_persons içinde uygulanır.
        """
        inference_conf = config.GLOVE_CONF if conf is None else conf
        results = self.model(frame, verbose=False, conf=inference_conf)
        detections = []

        for result in results:
            for box in result.boxes:
                cls_id     = int(box.cls[0])
                conf       = float(box.conf[0])
                xyxy       = box.xyxy[0].tolist()          # [x1,y1,x2,y2]
                raw_name    = self.model.names[cls_id].lower()
                class_name  = normalize_class_name(raw_name)

                detections.append({
                    "class_name": class_name,
                    "raw_class_name": raw_name,
                    "confidence": conf,
                    "box": xyxy,
                    "model_path": self.model_path,
                })

        return detections

    # ------------------------------------------------------------------ #
    @staticmethod
    def filter_by_class_conf(detections: list[dict]) -> list[dict]:
        """
        Sınıfa özel güven eşiği uygular.
        """
        thresholds = {
            "person": config.PERSON_CONF,
            "helmet": config.HELMET_CONF,
            "vest":   config.VEST_CONF,
            "mask":   config.MASK_CONF,
            "glove":  config.GLOVE_CONF,
            "goggles": getattr(config, "GLASSES_CONF", config.MASK_CONF),
        }
        filtered = []
        for d in detections:
            cls  = d["class_name"]
            conf = d["confidence"]
            
            # Örneğin "helmet_pos" veya "helmet_neg" -> "helmet"
            base_cls = cls.split("_")[0] if "_" in cls else cls
            
            # Eşik tablosunda yoksa (bilinmeyen sınıf) genel GLOVE_CONF uygula
            thr = thresholds.get(base_cls, config.GLOVE_CONF)
            if conf >= thr:
                filtered.append(d)
        return filtered
