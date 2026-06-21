"""
train_v4_clean.py
=================
Temiz eğitim — zincirleme fine-tune problemi çözüldü.

Önceki sorunlar:
  - yolov8s → gpu → phase2 → phase2-2 → phase2-3 → v3  (5 nesil zincirleme)
  - dropout=0.0, cos_lr=False → overfitting platosu ~epoch 15'te
  - gloves veri eksikliği → %52/%55 doğruluk

Bu versiyonda:
  - Temiz yolo11m.pt pretrained'den başlar (sıfırdan değil, sadece zincir kesilir)
  - dropout=0.1  → overfitting baskılar
  - cos_lr=True  → learning rate akıllıca azalır
  - patience=25  → gerçek plato gelince durur (~60-80 epoch bekleniyor)
  - mixup=0.15   → veri çeşitliliği artırır
  - Gloves sınıfı artık datasette mevcut (merge_gloves_to_combined.py çalıştırıldı)

Kullanım:
  Kaggle / Colab (GPU):  python train_v4_clean.py
  Yerel (CPU):           YAPMA — günler sürer
"""

from ultralytics import YOLO
from pathlib import Path
import yaml, os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
DATA_YAML = BASE / "data.yaml"
MODEL_OUT = BASE / "runs" / "detect" / "custom_ppe_v4_clean"

# ─── Kontrol: data.yaml sınıf sayısı ─────────────────────────────────────────
with open(DATA_YAML, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
assert cfg["nc"] == 10, f"Beklenen 10 sinif, data.yaml'da {cfg['nc']} var!"
print(f"[OK] data.yaml: {cfg['nc']} sinif -> {list(cfg['names'].values())}")

# ─── Model: TEMIZ pretrained (zincirleme yok) ─────────────────────────────────
# yolo11m.pt yoksa ultralytics otomatik indirir (~40 MB)
model = YOLO("yolo11m.pt")
print("[OK] Model: yolo11m.pt (temiz pretrained, zincirleme yok)")

# ─── Eğitim ──────────────────────────────────────────────────────────────────
results = model.train(
    data       = str(DATA_YAML),
    epochs     = 150,          # üst limit yüksek — patience erken durdurur
    patience   = 25,           # 25 epoch iyileşme yoksa dur (önceki: 15)
    batch      = 16,
    imgsz      = 640,
    device     = 0,            # GPU 0 — Kaggle'da T4/P100 için

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer  = "AdamW",
    lr0        = 0.001,        # önceki: 0.0005 — daha hızlı başlar
    lrf        = 0.01,         # son lr = lr0 * lrf
    cos_lr     = True,         # cosine schedule (önceki: False)
    momentum   = 0.937,
    weight_decay = 0.0005,

    # ── Overfitting Önleme ────────────────────────────────────────────────────
    dropout    = 0.1,          # önceki: 0.0

    # ── Veri Artırma (Augmentation) ───────────────────────────────────────────
    mixup      = 0.15,         # önceki: 0.0  — gloves gibi az sınıflar için kritik
    copy_paste = 0.1,          # küçük nesneleri yapıştırarak çoğaltır
    mosaic     = 1.0,
    hsv_h      = 0.015,
    hsv_s      = 0.7,
    hsv_v      = 0.4,
    fliplr     = 0.5,
    scale      = 0.5,
    erasing    = 0.4,

    # ── Kayıt / Log ───────────────────────────────────────────────────────────
    project    = str(BASE / "runs" / "detect"),
    name       = "custom_ppe_v4_clean",
    exist_ok   = True,
    save       = True,
    plots      = True,
    verbose    = True,

    # ── Performans ────────────────────────────────────────────────────────────
    workers    = 4,
    cache      = True,         # RAM'e önbellekle — ilk epoch yavaş, sonrası hızlı
    amp        = True,         # mixed precision — GPU belleği korur
)

print("\n" + "="*55)
print("EGITIM TAMAMLANDI")
print(f"  En iyi model: {MODEL_OUT}/weights/best.pt")
print(f"  mAP50  : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
print(f"  mAP5095: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
print()
print("Sonraki adim: config.py'de MODEL_PATH'i guncelle")
print("  MODEL_PATH = 'runs/detect/custom_ppe_v4_clean/weights/best.pt'")
