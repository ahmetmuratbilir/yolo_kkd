# ============================================================
#  config.py  –  ISG KKD Algılama Sistemi Yapılandırması
# ============================================================

# ── Çalışma Modu ────────────────────────────────────────────
# "realtime"  = canlı kamera, hafif, yüksek FPS
# "benchmark" = tüm modeller açık, detaylı analiz
RUN_MODE = "realtime"

# ── Kamera ──────────────────────────────────────────────────
CAMERA_SOURCE = 0          # 0 = varsayılan webcam | "rtsp://..." = IP kamera
CAMERA_BACKENDS = ["DSHOW", "ANY", "MSMF"]  # Windows'ta bozuk/parazitli görüntü için sırayla denenir
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_FOURCC = ""         # bazı kameralar MJPG isteyince siyah/parazitli görüntü verir
CAMERA_REOPEN_AFTER_FAILURES = 20

# ── Model yolu ──────────────────────────────────────────────
# Ana model: testlerde kask/yelek ve kisi kutusunda en guclu sonuc veren yerel PPE modeli.
MODEL_PATH = "models/ppe_model.pt"
FALLBACK_MODEL_PATH = "models/ppe_model.pt"

# Yardimci PPE modelleri: ana modelin kacirdigi eldiven/gozluk/yelek/kask adaylarini ekler.
ENABLE_AUX_PPE_MODELS = True
AUX_PPE_MODEL_PATHS = [
    "models/vyra_yolo_ppe_best.pt",
    "models/tanish_yolov8n_ppe_6class.pt",
]
AUX_PPE_MODEL_CONF = 0.15
DETECTION_MERGE_IOU = 0.65

# ── Kask Renk Doğrulaması (Saçı kask sanmayı önler) ──────────────
# Model başlık bölgesinde kask görse bile, o bölgede gerçek İSG kaski rengi
# (sarı, kırmızı, turuncu, beyaz, mavi) yoksa kask kararı reddedilir.
ENABLE_HELMET_COLOR_VERIFY = True
HELMET_COLOR_MIN_RATIO = 0.20   # Agresif filtre: kask bölgesinin en az %20'si saf kask rengi olmak zorunda. Saç/cilt yansımalarını eler.

# Sarı kask (en yaygın iş güvenliği rengi)
LOWER_HELMET_YELLOW = [15,  80,  80]
UPPER_HELMET_YELLOW = [40, 255, 255]

# Kırmızı kask
LOWER_HELMET_RED1   = [0,   100, 80]
UPPER_HELMET_RED1   = [10,  255, 255]
LOWER_HELMET_RED2   = [165, 100, 80]
UPPER_HELMET_RED2   = [180, 255, 255]

# Turuncu kask
LOWER_HELMET_ORANGE = [8,   100, 80]
UPPER_HELMET_ORANGE = [18,  255, 255]

# Beyaz kask (düşük doygunluk, yüksek parlaklık)
LOWER_HELMET_WHITE  = [0,   0,   180]
UPPER_HELMET_WHITE  = [180, 55,  255]

# Mavi kask
LOWER_HELMET_BLUE   = [95,  80,  80]
UPPER_HELMET_BLUE   = [135, 255, 255]

HELMET_COLOR_RANGES = [
    (LOWER_HELMET_YELLOW, UPPER_HELMET_YELLOW),
    (LOWER_HELMET_RED1,   UPPER_HELMET_RED1),
    (LOWER_HELMET_RED2,   UPPER_HELMET_RED2),
    (LOWER_HELMET_ORANGE, UPPER_HELMET_ORANGE),
    (LOWER_HELMET_WHITE,  UPPER_HELMET_WHITE),
    (LOWER_HELMET_BLUE,   UPPER_HELMET_BLUE),
]

# ── Model Aktiflik Kontrolü (RUN_MODE'a göre override edilir) ─
# Canlı modda FPS'i düşüren modelleri tek tek kapatabilirsiniz.
ENABLE_VYRA_MODEL  = True   # vyra canlı modda da açık kalsın
ENABLE_TANISH_MODEL = False  # tanish sadece benchmark modda açılır

# ── Frame-Skip (Yardımcı modeller her karede çalışmasın) ────
# Ana model (ppe_model) her frame çalışır.
# Yardımcı modeller kaç frame'de bir çalışsın? (1 = her frame)
VYRA_FRAME_SKIP   = 3   # vyra 3 frame'de bir
TANISH_FRAME_SKIP = 5   # tanish 5 frame'de bir (sadece benchmark)
POSE_EVERY_N_FRAMES = 5  # realtime modda pose kaç frame'de bir çalışsın
                          # benchmark modda her frame çalışır

# Kapsamlı PPE modeli bazı kamera açılarında kişiyi kaçırırsa,
# kişi kutusunu COCO tabanlı genel modelden al.
# Realtime modda kapalı (FPS kazanımı için), benchmark modda açık.
ENABLE_PERSON_FALLBACK_MODEL = False  # RUN_MODE="benchmark" ise True yapın
PERSON_MODEL_PATH = "yolov8n.pt"
PERSON_FALLBACK_CONF = 0.25

# ── Güven eşikleri ──────────────────────────────────────────
PERSON_CONF  = 0.30
HELMET_CONF  = 0.50
VEST_CONF    = 0.50  # Yüksek eşik: sadece reflektif/fosforlu yelek kabul edilir
MASK_CONF    = 0.35
GLOVE_CONF   = 0.15
GLASSES_CONF = 0.15

# ── Model sınıf adı eşleştirmeleri ──────────────────────────
# Farklı PPE modelleri aynı nesneleri farklı isimlerle döndürebilir.
CLASS_MAP = {
    "person": "person",
    "worker": "person",
    "human": "person",
    
    "hardhat": "helmet_pos",
    "hard hat": "helmet_pos",
    "hard-hat": "helmet_pos",
    "safety helmet": "helmet_pos",
    "safety-helmet": "helmet_pos",
    "helmet": "helmet_pos",
    "no-hardhat": "helmet_neg",
    "no_hardhat": "helmet_neg",
    "no-helmet": "helmet_neg",
    "no_helmet": "helmet_neg",

    "safety vest": "vest_pos",
    "safety-vest": "vest_pos",
    "reflective vest": "vest_pos",
    "hi-vis": "vest_pos",
    "hivis": "vest_pos",
    "vest": "vest_pos",
    "no-safety vest": "vest_neg",
    "no_vest": "vest_neg",

    "glove": "glove_pos",
    "gloves": "glove_pos",
    "safety glove": "glove_pos",
    "safety gloves": "glove_pos",
    "no-glove": "glove_neg",
    "no-gloves": "glove_neg",

    "goggles": "goggles_pos",
    "goggle": "goggles_pos",
    "safety glasses": "goggles_pos",
    "safety-glasses": "goggles_pos",
    "safety goggles": "goggles_pos",
    "eye protection": "goggles_pos",
    "eye wear": "goggles_pos",
    "eyewear": "goggles_pos",
    "glasses": "goggles_pos",
    "no-goggles": "goggles_neg",
    "no_glasses": "goggles_neg",
    
    "mask": "mask_pos",
    "face mask": "mask_pos",
    "face-mask": "mask_pos",
    "respirator": "mask_pos",
    "gas mask": "mask_pos",
}

# ── Bölge oranları (kişi kutusuna göre 0-1 arası) ───────────
HEAD_REGION_RATIO   = 0.30   # kask / maske için üst %30
TORSO_TOP_RATIO     = 0.25   # yelek başlangıcı
TORSO_BOTTOM_RATIO  = 0.75   # yelek bitişi
HAND_TOP_RATIO      = 0.45   # el bölgesi başlangıcı
HAND_BOTTOM_RATIO   = 0.90   # el bölgesi bitişi
HAND_WIDTH_RATIO    = 0.35   # el bölgesi genişliği (kişi genişliğine oran)

# ── Ekipman-kişi eşleştirme hassasiyeti ─────────────────────
MIN_EQUIPMENT_OVERLAP = 0.08  # ekipman kutusunun ilgili kişi bölgesiyle en az örtüşme oranı
MIN_GLOVE_OVERLAP     = 0.05  # eldiven kutusunun el/bilek bölgesiyle en az örtüşme oranı

# ── Kişi kutusu temizleme ───────────────────────────────────
# Model bazen kol/el gibi parçaları ayrı "person" kutusu sanabilir.
FILTER_PERSON_BOXES        = True
PERSON_MIN_AREA_RATIO      = 0.006  # kare alanının en az %0.6'sı
PERSON_MIN_HEIGHT_RATIO    = 0.12   # kare yüksekliğinin en az %12'si
PERSON_MIN_ASPECT_RATIO    = 0.35   # h / w; çok yatay kutuları ele
PERSON_MAX_ASPECT_RATIO    = 4.50   # aşırı ince/dikey hatalı kutuları ele
PERSON_DUPLICATE_IOA       = 0.70   # küçük kutu büyük kutunun parçasıysa at
SHOW_DEBUG_COUNTS          = True   # üst barda ham/filtrelenmiş kişi sayısını göster

# ── Eldiven renk analizi ─────────────────────────────────────
# OpenCV HSV: H=0-179, S=0-255, V=0-255
GLOVE_COLOR_RATIO = 0.25     # bilek bölgesinde bu orandan fazla renk görünürse eldiven var
ENABLE_GLOVE_COLOR_FALLBACK = True  # HSV analizi eldiven kuralında destek olarak kullanılacak

# Mavi eldiven
LOWER_GLOVE_BLUE  = [90,  50,  50]
UPPER_GLOVE_BLUE  = [130, 255, 255]

# Sarı / turuncu eldiven (ikincil kontrol)
LOWER_GLOVE_YELLOW = [15,  80,  80]
UPPER_GLOVE_YELLOW = [40, 255, 255]

# Siyah eldiven (değer kanalı çok düşük)
LOWER_GLOVE_BLACK = [0,   0,   0]
UPPER_GLOVE_BLACK = [180, 255, 60]

# Beyaz / gri eldiven (düşük doygunluk, yüksek parlaklık)
# SADECE bilek/el ROI içinde çalışır, eşik kasıtlı olarak yüksek tutulmuştur.
LOWER_GLOVE_WHITE = [0,   0,  150]
UPPER_GLOVE_WHITE = [180, 60, 255]
WHITE_GLOVE_MIN_RATIO = 0.45   # Bilek bölgesinin %45'i beyaz/gri olmalı
MIN_WRIST_REGION_AREA = 800    # piksel²; çok küçük bölgede beyaz aramayı atla

# Sarı eldiven için özel oran eşiği (sadece el/bilek ROI'de geçerli)
YELLOW_GLOVE_MIN_RATIO = 0.30   # bilek bölgesinin %30'u sarıysa eldiven VAR

# Aktif olarak kullanılacak renk aralıkları listesi
GLOVE_COLOR_RANGES = [
    (LOWER_GLOVE_BLUE,   UPPER_GLOVE_BLUE,   GLOVE_COLOR_RATIO),
    (LOWER_GLOVE_YELLOW, UPPER_GLOVE_YELLOW, YELLOW_GLOVE_MIN_RATIO),  # sarı için özel eşik
    (LOWER_GLOVE_BLACK,  UPPER_GLOVE_BLACK,  GLOVE_COLOR_RATIO),
    (LOWER_GLOVE_WHITE,  UPPER_GLOVE_WHITE,  WHITE_GLOVE_MIN_RATIO),  # yüksek eşik
]

# ── Yelek renk fallback'i ───────────────────────────────────
# Model yeleği kaçırırsa torso bölgesinde fosforlu sarı/yeşil/turuncu aranır.
ENABLE_VEST_COLOR_FALLBACK = True
VEST_COLOR_RATIO = 0.10
# Kritik kural: Renk tek başına yelek VAR kararı veremez.
# Model (ppe_model veya vyra) gövdede vest_pos görmedikçe
# renk analizi sonucu sadece fallback_sources'a 'color_only_rejected' olarak loglanır.
VEST_COLOR_REQUIRES_MODEL_CONFIRM = True
VEST_SAT_MIN = 70
VEST_VAL_MIN = 80
VEST_COLOR_RANGES = [
    ([12, VEST_SAT_MIN, VEST_VAL_MIN], [28, 255, 255]),   # turuncu
    ([28, VEST_SAT_MIN, VEST_VAL_MIN], [45, 255, 255]),   # sarı
    ([45, VEST_SAT_MIN, VEST_VAL_MIN], [85, 255, 255]),   # fosforlu yeşil
]

# ── Kendi veri havuzu / hard-example toplama ────────────────
DATASET_REVIEW_DIR = "dataset_review"
AUTO_SAVE_HARD_EXAMPLES = True
HARD_EXAMPLE_COOLDOWN = 8.0

# ── Gerekli ekipmanlar ───────────────────────────────────────
REQUIRED_EQUIPMENTS = {
    "helmet":      True,
    "vest":        True,
    "mask":        False,   # Maske bu ortamda zorunlu degil
    "glasses":     False,   # Gozluk modeli yetersiz; benchmark sonrasi eklenecek
    "left_glove":  True,
    "right_glove": True,
}

# ── Güvenlik İhlal & Alarm Ayarları ─────────────────────────
SAVE_ALERTS    = True       # İhlal anlarında resim ve log dosyası kaydeder
ALERT_DIR      = "alerts"   # Kaydedilecek klasör adı
ALERT_COOLDOWN = 5.0        # İki ihlal resmi kaydı arasındaki minimum saniye (kişi başı)
PLAY_SOUND     = False      # İhlal anında bip sesi çalar

# ── Kişi takibi ─────────────────────────────────────────────
ENABLE_TRACKING      = True
TRACK_MAX_MISSED     = 12    # kişi birkaç kare görünmezse takibi ne kadar koruyalım?
TRACK_MIN_IOU        = 0.05
TRACK_MAX_DISTANCE   = 140   # piksel; kameraya göre artırılıp azaltılabilir

# ── Alarm doğrulama / kısa süreli hata filtresi ─────────────
ENABLE_STATUS_SMOOTHING = True
STATUS_HISTORY_SIZE_HELMET_VEST = 5
STATUS_CONFIRM_FRAMES_HELMET_VEST = 2
STATUS_HISTORY_SIZE_GLOVES_GOGGLES = 7
STATUS_CONFIRM_FRAMES_GLOVES_GOGGLES = 2
STATUS_MISSING_TIMEOUT_FRAMES = 3   # 3 kare peş peşe pozitif yoksa anında YOK (missing) kabul edilir

# ── İskelet Takibi (MediaPipe Pose) ─────────────────────────
# Realtime modda kapalı (FPS kazanımı). Benchmark modda açılabilir.
ENABLE_MEDIAPIPE = False    # True = hassas bilek tespiti, False = kişi kutusundan ROI tahmini

# ── Görsel ayarlar ───────────────────────────────────────────
FONT            = 0   # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.55
FONT_THICKNESS  = 2
COLOR_SAFE      = (50, 200, 50)    # BGR – yeşil
COLOR_WARN      = (0,  50,  220)   # BGR – kırmızı
COLOR_BOX       = (200, 200, 0)    # BGR – sarı çerçeve
