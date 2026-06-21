"""
Kaggle Eğitim Scripti — KKD Tespit Modeli V4
=============================================
Seçenek 2 implementasyonu:
  - /kaggle/input/combined-ppe-v2/ → mevcut büyük dataset (extract edilecek)
  - /kaggle/input/gloves-ppe-zips-cleaned/ → 5 küçük gloves ZIP (~120 MB)
  - Notebook içinde merge yapılır, sonra yolo11m.pt ile eğitim başlar

Parametreler (V3'e göre değişiklikler):
  - Model     : yolo11m.pt (temiz pretrained, zincirleme yok)
  - dropout   : 0.1   (önceki: 0.0)
  - cos_lr    : True  (önceki: False)
  - patience  : 25    (önceki: 15/20)
  - lr0       : 0.001 (önceki: 0.0005)
  - mixup     : 0.15  (önceki: 0.0)
  - copy_paste: 0.1   (önceki: 0.0)
  - epochs    : 150   (patience erken durdurur, ~60-80 epoch bekleniyor)
"""

import os
import zipfile
import random
import subprocess
from pathlib import Path
from collections import defaultdict

# ─── Kurulum ────────────────────────────────────────────────────────────────
subprocess.run(["pip", "install", "-q", "ultralytics"], check=True)
from ultralytics import YOLO

# ─── Sabit Yollar (Kaggle ortamına özel) ────────────────────────────────────
KAGGLE_INPUT   = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
DATASET_DIR    = KAGGLE_WORKING / "datasets" / "combined_ppe"
DATA_YAML      = KAGGLE_WORKING / "data.yaml"

# Mevcut combined-ppe dataset kaynağı
COMBINED_ZIP   = KAGGLE_INPUT / "combined-ppe-v2" / "combined_ppe.bin"

# Gloves ZIP'lerinin Kaggle'daki klasörü
# (dataset adını aşağıdaki GLOVES_INPUT_DIR ile eşleştir)
GLOVES_INPUT_DIR = KAGGLE_INPUT / "gloves-ppe-zips-cleaned"

# Sınıf ID'leri (data.yaml ile birebir aynı)
GLOVES_POS_ID = 5
GLOVES_NEG_ID = 6

# Gloves ZIP eşleştirmeleri
ZIP_CONFIGS = [
    ("gloves_v1i_yolov8.zip",
     {0: GLOVES_NEG_ID, 1: GLOVES_POS_ID},
     "GLOVES.v1i (no_gloves/with_gloves)"),
    ("gloves_v1i_yolov8_2.zip",
     {0: GLOVES_POS_ID, 1: GLOVES_NEG_ID},
     "gloves.v1i(2) (Glove/No-Glove)"),
    ("gloves_detection_v1i_yolov8.zip",
     {0: GLOVES_POS_ID},
     "Gloves Detection.v1i (sadece pos)"),
    ("gloves_detection_v4i_yolov8.zip",
     {0: GLOVES_POS_ID},
     "gloves detection.v4i"),
    ("gloves_v1i_yolov8_1.zip",
     {0: GLOVES_POS_ID},
     "gloves.v1i(1)"),
]

VAL_RATIO  = 0.15
TEST_RATIO = 0.05


# ─── Adım 1: combined_ppe'yi çıkart ─────────────────────────────────────────
def extract_combined():
    if (DATASET_DIR / "train" / "images").exists():
        imgs = len(list((DATASET_DIR / "train" / "images").iterdir()))
        print(f"[OK] combined_ppe zaten mevcut — train: {imgs:,} resim")
        return

    print(f"Extracting {COMBINED_ZIP} ...")
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(COMBINED_ZIP, "r") as zf:
        for member in zf.infolist():
            # Windows ters slash'larını Linux uyumlu düz slash'lara çevir
            clean_name = member.filename.replace("\\", "/")
            
            # Hedef yol
            dest_path = DATASET_DIR.parent / clean_name
            
            # Klasör ise oluştur
            if member.is_dir() or clean_name.endswith("/"):
                dest_path.mkdir(parents=True, exist_ok=True)
            else:
                # Üst klasörün varlığından emin ol
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                # Dosyayı yaz
                dest_path.write_bytes(zf.read(member.filename))
                
    print("[OK] Extraction tamamlandi")


# ─── Adım 2: Gloves merge ────────────────────────────────────────────────────
def convert_label(lines, class_map):
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        if old_id not in class_map:
            continue
        out.append(f"{class_map[old_id]} " + " ".join(parts[1:]))
    return out


def choose_split():
    r = random.random()
    if r < TEST_RATIO:
        return "test"
    elif r < TEST_RATIO + VAL_RATIO:
        return "valid"
    else:
        return "train"


def safe_name(zip_name, orig_name):
    prefix = Path(zip_name).stem.replace(" ", "_").replace(".", "_")
    return f"{prefix}__{Path(orig_name).name}"


def merge_gloves():
    random.seed(42)
    total_added = 0
    total_skipped = 0

    for zip_filename, class_map, desc in ZIP_CONFIGS:
        zip_path = GLOVES_INPUT_DIR / zip_filename
        folder_name = zip_filename.replace(".zip", "")
        folder_path = GLOVES_INPUT_DIR / folder_name

        if folder_path.exists() and folder_path.is_dir():
            print(f"\n[DIR] {desc} (Klasor olarak bulundu: {folder_name})")
            added = 0
            skipped = 0

            # Find all image files in the folder (we look for images inside any subdirectory)
            img_files = []
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                img_files.extend(folder_path.glob(f"**/images/*{ext}"))

            print(f"  Resim sayisi: {len(img_files)}")

            for img_p in img_files:
                # Find corresponding label file
                parts = list(img_p.parts)
                try:
                    idx = len(parts) - parts[::-1].index("images") - 1
                    parts[idx] = "labels"
                    lbl_p = Path(*parts).with_suffix(".txt")
                except ValueError:
                    skipped += 1
                    continue

                if not lbl_p.exists():
                    skipped += 1
                    continue

                try:
                    raw = lbl_p.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    skipped += 1
                    continue

                converted = convert_label(raw.splitlines(), class_map)
                if not converted:
                    skipped += 1
                    continue

                split = choose_split()
                img_dir = DATASET_DIR / split / "images"
                lbl_dir = DATASET_DIR / split / "labels"
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)

                new_img = safe_name(zip_filename, img_p.name)
                new_lbl = Path(new_img).with_suffix(".txt").name

                dst_img = img_dir / new_img
                dst_lbl = lbl_dir / new_lbl

                if dst_img.exists():
                    skipped += 1
                    continue

                import shutil
                shutil.copy2(img_p, dst_img)
                dst_lbl.write_text("\n".join(converted) + "\n", encoding="utf-8")
                added += 1

            print(f"  [OK] Eklendi: {added}  |  Atlandi: {skipped}")
            total_added += added
            total_skipped += skipped

        elif zip_path.exists():
            print(f"\n[ZIP] {desc}")
            added = 0
            skipped = 0

            with zipfile.ZipFile(zip_path, "r") as zf:
                all_files = zf.namelist()
                img_files = [
                    f for f in all_files
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    and "/images/" in f
                ]
                print(f"  Resim sayisi: {len(img_files)}")

                for img_path in img_files:
                    lbl_path = img_path.replace("/images/", "/labels/")
                    dot = lbl_path.rfind(".")
                    lbl_path = lbl_path[:dot] + ".txt"

                    if lbl_path not in all_files:
                        skipped += 1
                        continue

                    raw = zf.read(lbl_path).decode("utf-8", errors="ignore")
                    converted = convert_label(raw.splitlines(), class_map)

                    if not converted:
                        skipped += 1
                        continue

                    split = choose_split()
                    img_dir = DATASET_DIR / split / "images"
                    lbl_dir = DATASET_DIR / split / "labels"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    lbl_dir.mkdir(parents=True, exist_ok=True)

                    new_img = safe_name(zip_filename, Path(img_path).name)
                    new_lbl = Path(new_img).with_suffix(".txt").name

                    dst_img = img_dir / new_img
                    dst_lbl = lbl_dir / new_lbl

                    if dst_img.exists():
                        skipped += 1
                        continue

                    dst_img.write_bytes(zf.read(img_path))
                    dst_lbl.write_text("\n".join(converted) + "\n", encoding="utf-8")
                    added += 1

            print(f"  [OK] Eklendi: {added}  |  Atlandi: {skipped}")
            total_added += added
            total_skipped += skipped
        else:
            print(f"  [!] Bulunamadi: {zip_filename} veya klasoru — atlaniyor")
            total_skipped += 1

    print(f"\nMerge tamamlandi: {total_added} resim eklendi, {total_skipped} atlandi")
    for split in ["train", "valid", "test"]:
        d = DATASET_DIR / split / "images"
        if d.exists():
            print(f"  {split}: {len(list(d.iterdir())):,} resim")


# ─── Adım 3: data.yaml yaz ───────────────────────────────────────────────────
def write_yaml():
    content = f"""path: {DATASET_DIR}
train: train/images
val: valid/images
test: test/images

nc: 10
names:
  0: person
  1: helmet_pos
  2: helmet_neg
  3: vest_pos
  4: vest_neg
  5: gloves_pos
  6: gloves_neg
  7: goggles_pos
  8: goggles_neg
  9: smoking
"""
    DATA_YAML.write_text(content, encoding="utf-8")
    print(f"[OK] data.yaml yazildi: {DATA_YAML}")


# ─── Adım 4: Eğitim ──────────────────────────────────────────────────────────
def train():
    print("\n" + "=" * 55)
    print("EGITIM BASLIYOR — yolo11m.pt (temiz pretrained)")
    print("=" * 55)

    model = YOLO("yolo11m.pt")

    results = model.train(
        data        = str(DATA_YAML),
        epochs      = 150,
        patience    = 25,           # val loss 25 epoch iyilesmezse dur
        batch       = 16,
        imgsz       = 640,
        device      = 0,

        # Optimizer
        optimizer   = "AdamW",
        lr0         = 0.001,        # temiz baslangiç — yüksek lr dogru
        lrf         = 0.01,
        cos_lr      = True,         # cosine schedule
        momentum    = 0.937,
        weight_decay= 0.0005,

        # Overfitting onleme
        dropout     = 0.1,

        # Augmentation
        mixup       = 0.15,
        copy_paste  = 0.1,
        mosaic      = 1.0,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        fliplr      = 0.5,
        scale       = 0.5,
        erasing     = 0.4,

        # Kayit
        project     = str(KAGGLE_WORKING / "runs" / "detect"),
        name        = "custom_ppe_v4",
        exist_ok    = True,
        save        = True,
        plots       = True,
        verbose     = True,

        # Performans
        workers     = 4,
        cache       = "ram",        # Kaggle RAM'i bol, cache kullan
        amp         = True,
    )

    best = KAGGLE_WORKING / "runs" / "detect" / "custom_ppe_v4" / "weights" / "best.pt"
    print("\n" + "=" * 55)
    print("EGITIM TAMAMLANDI")
    print(f"  mAP50    : {results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
    print(f"  mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 0):.3f}")
    print(f"  Best model: {best}")
    print("=" * 55)


# ─── Ana Akış ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== KKD V4 Egitim Pipeline ===\n")
    extract_combined()   # Adım 1
    merge_gloves()       # Adım 2
    write_yaml()         # Adım 3
    train()              # Adım 4
