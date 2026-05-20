# ============================================================
#  main.py  –  ISG KKD Algılama Sistemi  |  Ana döngü
# ============================================================
import sys
import time
import os
import cv2
import threading

import config
from services.detector         import PPEDetector, normalize_class_name
from services.rule_engine      import assign_equipment_to_persons
from services.drawing          import draw_person_status, draw_fps, draw_header
from services.pose_detector    import PoseDetector
from services.logger           import ViolationLogger
from services.person_tracker   import PersonTracker
from services.status_smoother  import StatusSmoother
from services.person_filter    import filter_person_boxes
from services.camera           import open_camera, reopen_camera
from services.dataset_collector import DatasetCollector
from services.detection_merge  import merge_detections


# ── Servisleri ve Modeli yükle ────────────────────────── #
RUN_MODE = getattr(config, "RUN_MODE", "realtime").lower()
print(f"\n[main] Çalışma modu: {RUN_MODE.upper()}")

# RUN_MODE'a göre bazı ayarları otomatik olarak override et
if RUN_MODE == "benchmark":
    config.ENABLE_PERSON_FALLBACK_MODEL = True
    config.ENABLE_MEDIAPIPE = True
    config.ENABLE_TANISH_MODEL = True
    config.ENABLE_VYRA_MODEL = True
    config.POSE_EVERY_N_FRAMES = 1   # benchmark: her frame pose
    print("[main] Benchmark modu: Tüm modeller ve MediaPipe ACIK")
else:  # realtime
    config.ENABLE_PERSON_FALLBACK_MODEL = False
    config.ENABLE_MEDIAPIPE = True   # Eldiven tespiti için iskelet takibi ŞART, geri açıldı.
    config.ENABLE_TANISH_MODEL = False
    config.ENABLE_VYRA_MODEL = False # Vyra kapalı (çok fazla false-positive kask üretiyor)
    print("[main] Realtime modu: Ana Model (ppe_model.pt) + MediaPipe (Eldiven için) çalışacak")

model_path = config.MODEL_PATH
if not os.path.exists(model_path) and getattr(config, "FALLBACK_MODEL_PATH", None):
    fallback_model_path = config.FALLBACK_MODEL_PATH
    if os.path.exists(fallback_model_path):
        print(f"[main] UYARI: {model_path} bulunamadi, fallback model kullaniliyor: {fallback_model_path}")
        model_path = fallback_model_path

detector = PPEDetector(model_path)

# Yardımcı modelleri RUN_MODE + ENABLE_ bayraklarına göre yükle
vyra_detector = None
tanish_detector = None

if getattr(config, "ENABLE_AUX_PPE_MODELS", False):
    aux_paths = getattr(config, "AUX_PPE_MODEL_PATHS", [])
    for aux_model_path in aux_paths:
        if aux_model_path == model_path:
            continue
        if not os.path.exists(aux_model_path):
            print(f"[main] UYARI: Yardimci PPE modeli bulunamadi: {aux_model_path}")
            continue
        is_vyra  = "vyra"   in aux_model_path.lower()
        is_tanish = "tanish" in aux_model_path.lower()
        if is_vyra and config.ENABLE_VYRA_MODEL:
            print(f"[main] Vyra modeli yukleniyor: {aux_model_path}")
            vyra_detector = PPEDetector(aux_model_path)
        elif is_tanish and config.ENABLE_TANISH_MODEL:
            print(f"[main] Tanish modeli yukleniyor: {aux_model_path}")
            tanish_detector = PPEDetector(aux_model_path)

person_detector = None
if config.ENABLE_PERSON_FALLBACK_MODEL:
    person_model_path = getattr(config, "PERSON_MODEL_PATH", "")
    if person_model_path and os.path.exists(person_model_path):
        print(f"[main] Kisi fallback modeli yukleniyor: {person_model_path}")
        person_detector = PPEDetector(person_model_path)
    else:
        print(f"[main] UYARI: Kisi fallback modeli bulunamadi: {person_model_path}")

pose_detector = PoseDetector()
violation_logger = ViolationLogger()
person_tracker = PersonTracker()
status_smoother = StatusSmoother()
dataset_collector = DatasetCollector()

# Frame-skip aralıkları
VYRA_SKIP  = getattr(config, "VYRA_FRAME_SKIP",   3)
TANISH_SKIP = getattr(config, "TANISH_FRAME_SKIP", 5)
POSE_SKIP  = getattr(config, "POSE_EVERY_N_FRAMES", 5)

# Cache: ara frame'lerde son sonuçlar kullanılır
_cached_vyra_dets   = []    # son vyra tespitleri
_cached_tanish_dets = []    # son tanish tespitleri
_cached_wrists      = None  # son bilek/pose bilgisi

# Modelin hangi sınıfları bildiğini al
model_name_values = (
    detector.model.names.values()
    if hasattr(detector.model.names, "values")
    else detector.model.names
)
MODEL_CLASSES = {str(name).lower() for name in model_name_values}
NORMALIZED_MODEL_CLASSES = {normalize_class_name(name) for name in MODEL_CLASSES}

print("[main] Model sınıfları:", MODEL_CLASSES)
print("[main] Normalize sınıflar:", NORMALIZED_MODEL_CLASSES)
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
cap, camera_backend = open_camera()

if cap is None or not cap.isOpened():
    print("[HATA] Kamera açılamadı. CAMERA_SOURCE değerini kontrol edin.")
    sys.exit(1)


# ── FPS sayacı ───────────────────────────────────────────── #
prev_time = time.time()
fps = 0.0
failed_reads = 0
frame_count = 0

# ── Ana döngü ────────────────────────────────────────────── #
while True:
    ret, frame = cap.read()
    if not ret:
        failed_reads += 1
        print(f"[HATA] Kare okunamadi. Deneme: {failed_reads}")
        if failed_reads >= getattr(config, "CAMERA_REOPEN_AFTER_FAILURES", 20):
            print("[Camera] Kamera yeniden aciliyor...")
            cap, camera_backend = reopen_camera(cap)
            failed_reads = 0
            if cap is None:
                break
        continue
    failed_reads = 0

    # Kopya kare al (çizimsiz temiz kareyi loglamak için)
    clean_frame = frame.copy()

    # ── İskelet Takibi (MediaPipe) ──────────────────── #
    # Pose her frame değil, POSE_SKIP karede bir güncellenir.
    # Arada kalan frame'lerde son bilinen bilek konumu kullanılır (cache).
    if getattr(config, "ENABLE_MEDIAPIPE", False):
        if frame_count % POSE_SKIP == 0:
            _cached_wrists = pose_detector.find_wrists(frame)
        wrists = _cached_wrists
    else:
        wrists = None

    # ── Nesne Tespiti (YOLO) ──────────────────────────── #
    frame_count += 1

    raw = detector.detect(frame)
    dets = detector.filter_by_class_conf(raw)
    aux_dets = []

    # Vyra: her VYRA_SKIP frame'de bir çalışır, arada cache kullanılır
    if vyra_detector is not None:
        if frame_count % VYRA_SKIP == 0:
            vyra_raw = vyra_detector.detect(frame, conf=getattr(config, "AUX_PPE_MODEL_CONF", None))
            _cached_vyra_dets = vyra_detector.filter_by_class_conf(vyra_raw)
        aux_dets.extend(_cached_vyra_dets)

    # Tanish: her TANISH_SKIP frame'de bir çalışır, arada cache kullanılır (sadece benchmark)
    if tanish_detector is not None:
        if frame_count % TANISH_SKIP == 0:
            tanish_raw = tanish_detector.detect(frame, conf=getattr(config, "AUX_PPE_MODEL_CONF", None))
            _cached_tanish_dets = tanish_detector.filter_by_class_conf(tanish_raw)
        aux_dets.extend(_cached_tanish_dets)

    if aux_dets:
        dets = merge_detections(dets, aux_dets)
    fallback_persons = []
    if person_detector is not None:
        person_raw = person_detector.detect(
            frame,
            conf=getattr(config, "PERSON_FALLBACK_CONF", config.PERSON_CONF),
        )
        person_dets = person_detector.filter_by_class_conf(person_raw)
        fallback_persons = [d for d in person_dets if d["class_name"] == "person"]

    # Sınıflara göre ayır
    raw_persons = [d for d in dets if d["class_name"] == "person"]
    if fallback_persons:
        raw_persons = [*raw_persons, *fallback_persons]
    persons = raw_persons
    persons = filter_person_boxes(persons, frame.shape)
    persons = person_tracker.update(persons)
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

    # ── Kural motoru ────────────────────────────────────── #
    results = assign_equipment_to_persons(
        frame, persons, 
        helmets_pos, helmets_neg, 
        vests_pos, vests_neg, 
        masks_pos, masks_neg, 
        gloves_pos, gloves_neg, 
        glasses_pos=glasses_pos, glasses_neg=glasses_neg, 
        wrists=wrists
    )
    results = status_smoother.update(results)

    # ── Çizim ───────────────────────────────────────────── #
    unsafe_count = sum(1 for r in results if not r["safe"])
    debug_text = ""
    if getattr(config, "SHOW_DEBUG_COUNTS", False):
        debug_text = (
            f"Cam:{camera_backend} | Det: {len(dets)} | Ham P: {len(raw_persons)} | "
            f"Aux:{len(aux_dets)} | PF:{len(fallback_persons)} | "
            f"P: {len(persons)} | Kask:{len(helmets_pos)} Yelek:{len(vests_pos)} "
            f"Maske:{len(masks_pos)} Gozluk:{len(glasses_pos)} Eldiven:{len(gloves_pos)}"
        )
    draw_header(frame, len(results), unsafe_count, debug_text=debug_text)

    for r in results:
        draw_person_status(frame, r)

        # ── İhlal Kaydı ve Alarm Tetikleme ───────────────── #
        if not r["safe"]:
            logged = violation_logger.log_violation(
                clean_frame,
                r["person_id"],
                r["warnings"],
                result=r,
            )
            if logged and getattr(config, "PLAY_SOUND", True):
                # Ayrı bir thread ile beklemesiz ses çal
                threading.Thread(target=play_alert_sound, daemon=True).start()

    dataset_collector.maybe_save_hard_example(clean_frame, results, dets)

    # ── FPS ─────────────────────────────────────────────── #
    now = time.time()
    fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
    prev_time = now
    draw_fps(frame, fps)

    # ── Göster ──────────────────────────────────────────── #
    cv2.imshow("ISG KKD Algilama", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        dataset_collector.save_manual(clean_frame, results, dets, reason="manual_capture")
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[main] Sistem kapatıldı.")
