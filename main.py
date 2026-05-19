# ============================================================
#  main.py  –  ISG KKD Algılama Sistemi  |  Ana döngü
# ============================================================
import sys
import time
import cv2
import threading

import config
from services.detector         import PPEDetector
from services.rule_engine      import assign_equipment_to_persons
from services.drawing          import draw_person_status, draw_fps, draw_header
from services.pose_detector    import PoseDetector
from services.logger           import ViolationLogger


# ── Servisleri ve Modeli yükle ───────────────────────────── #
detector = PPEDetector(config.MODEL_PATH)
pose_detector = PoseDetector()
violation_logger = ViolationLogger()

# Modelin hangi sınıfları bildiğini al
MODEL_CLASSES = set(detector.model.names[i].lower()
                    for i in detector.model.names)

print("\n[main] Model sınıfları:", MODEL_CLASSES)
print("[main] 'q' tuşu ile çıkabilirsiniz.\n")


# ── Yardımcı Fonksiyonlar ────────────────────────────────── #
def play_alert_sound():
    """Arka planda uyarı bip sesi çalar."""
    try:
        import winsound
        winsound.Beep(1200, 250)  # 1200 Hz frekans, 250 ms süre
    except Exception:
        pass


# ── Kamera aç ────────────────────────────────────────────── #
cap = cv2.VideoCapture(config.CAMERA_SOURCE,
                       cv2.CAP_DSHOW if sys.platform == "win32" else 0)

if not cap.isOpened():
    print("[HATA] Kamera açılamadı. CAMERA_SOURCE değerini kontrol edin.")
    sys.exit(1)


# ── FPS sayacı ───────────────────────────────────────────── #
prev_time = time.time()
fps = 0.0

# ── Ana döngü ────────────────────────────────────────────── #
while True:
    ret, frame = cap.read()
    if not ret:
        print("[HATA] Kare okunamadı.")
        break

    # Kopya kare al (çizimsiz temiz kareyi loglamak için)
    clean_frame = frame.copy()

    # ── İskelet Takibi (MediaPipe) ──────────────────────── #
    wrists = None
    if getattr(config, "ENABLE_MEDIAPIPE", True):
        wrists = pose_detector.find_wrists(frame)

    # ── Nesne Tespiti (YOLO) ────────────────────────────── #
    raw = detector.detect(frame)
    dets = detector.filter_by_class_conf(raw)

    # Sınıflara göre ayır
    persons = [d for d in dets if d["class_name"] == "person"]
    helmets = [d for d in dets if ("helmet" in d["class_name"] or "hardhat" in d["class_name"])
               and "no" not in d["class_name"]]
    vests   = [d for d in dets if "vest"   in d["class_name"]
               and "no" not in d["class_name"]]
    masks   = [d for d in dets if "mask"   in d["class_name"]
               and "no" not in d["class_name"]]
    gloves  = [d for d in dets if "glove"  in d["class_name"]
               and "no" not in d["class_name"]]

    # ── Kural motoru ────────────────────────────────────── #
    results = assign_equipment_to_persons(
        frame, persons, helmets, vests, masks, gloves, wrists=wrists
    )

    # ── Çizim ───────────────────────────────────────────── #
    unsafe_count = sum(1 for r in results if not r["safe"])
    draw_header(frame, len(results), unsafe_count)

    for r in results:
        draw_person_status(frame, r)

        # ── İhlal Kaydı ve Alarm Tetikleme ───────────────── #
        if not r["safe"]:
            logged = violation_logger.log_violation(clean_frame, r["person_id"], r["warnings"])
            if logged and getattr(config, "PLAY_SOUND", True):
                # Ayrı bir thread ile beklemesiz ses çal
                threading.Thread(target=play_alert_sound, daemon=True).start()

    # ── FPS ─────────────────────────────────────────────── #
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    draw_fps(frame, fps)

    # ── Göster ──────────────────────────────────────────── #
    cv2.imshow("ISG KKD Algilama", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[main] Sistem kapatıldı.")
