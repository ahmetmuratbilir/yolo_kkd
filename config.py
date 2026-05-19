# ============================================================
#  config.py  –  ISG KKD Algılama Sistemi Yapılandırması
# ============================================================

# ── Kamera ──────────────────────────────────────────────────
CAMERA_SOURCE = 0          # 0 = varsayılan webcam | "rtsp://..." = IP kamera

# ── Model yolu ──────────────────────────────────────────────
MODEL_PATH = "models/ppe_model.pt"

# ── Güven eşikleri ──────────────────────────────────────────
PERSON_CONF  = 0.50
HELMET_CONF  = 0.50
VEST_CONF    = 0.50
MASK_CONF    = 0.45
GLOVE_CONF   = 0.45

# ── Bölge oranları (kişi kutusuna göre 0-1 arası) ───────────
HEAD_REGION_RATIO   = 0.30   # kask / maske için üst %30
TORSO_TOP_RATIO     = 0.25   # yelek başlangıcı
TORSO_BOTTOM_RATIO  = 0.75   # yelek bitişi
HAND_TOP_RATIO      = 0.45   # el bölgesi başlangıcı
HAND_BOTTOM_RATIO   = 0.90   # el bölgesi bitişi
HAND_WIDTH_RATIO    = 0.35   # el bölgesi genişliği (kişi genişliğine oran)

# ── Eldiven renk analizi ─────────────────────────────────────
# OpenCV HSV: H=0-179, S=0-255, V=0-255
GLOVE_COLOR_RATIO = 0.25     # bilek bölgesinde bu orandan fazla renk görünürse eldiven var

# Mavi eldiven
LOWER_GLOVE_BLUE  = [90,  50,  50]
UPPER_GLOVE_BLUE  = [130, 255, 255]

# Sarı / turuncu eldiven (ikincil kontrol)
LOWER_GLOVE_YELLOW = [15,  80,  80]
UPPER_GLOVE_YELLOW = [40, 255, 255]

# Siyah eldiven (değer kanalı çok düşük)
LOWER_GLOVE_BLACK = [0,   0,   0]
UPPER_GLOVE_BLACK = [180, 255, 60]

# Aktif olarak kullanılacak renk aralıkları listesi
GLOVE_COLOR_RANGES = [
    (LOWER_GLOVE_BLUE,   UPPER_GLOVE_BLUE),
    (LOWER_GLOVE_YELLOW, UPPER_GLOVE_YELLOW),
    (LOWER_GLOVE_BLACK,  UPPER_GLOVE_BLACK),
]

# ── Gerekli ekipmanlar ───────────────────────────────────────
REQUIRED_EQUIPMENTS = {
    "helmet":      True,
    "vest":        True,
    "mask":        True,
    "left_glove":  True,
    "right_glove": True,
}

# ── Güvenlik İhlal & Alarm Ayarları ─────────────────────────
SAVE_ALERTS    = True       # İhlal anlarında resim ve log dosyası kaydeder
ALERT_DIR      = "alerts"   # Kaydedilecek klasör adı
ALERT_COOLDOWN = 5.0        # İki ihlal resmi kaydı arasındaki minimum saniye (kişi başı)
PLAY_SOUND     = True       # İhlal anında bip sesi çalar

# ── İskelet Takibi (MediaPipe Pose) ─────────────────────────
ENABLE_MEDIAPIPE = True     # Eldiven/el bölgesi analizi için MediaPipe aktif eder

# ── Görsel ayarlar ───────────────────────────────────────────
FONT            = 0   # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.55
FONT_THICKNESS  = 2
COLOR_SAFE      = (50, 200, 50)    # BGR – yeşil
COLOR_WARN      = (0,  50,  220)   # BGR – kırmızı
COLOR_BOX       = (200, 200, 0)    # BGR – sarı çerçeve
