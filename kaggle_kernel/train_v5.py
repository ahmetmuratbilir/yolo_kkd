"""
Kaggle Eğitim Scripti — KKD Tespit Modeli V4  [SEANS 5 - Geliştirilmiş Sürüm V15]
========================================================================
Yeni Özellikler ve İyileştirmeler:
  - Çözünürlük  : imgsz = 960 (Küçük gözlük/eldiven nesneleri için 2.5 kat daha fazla piksel)
  - Batch Size  : batch = 8 (960px çözünürlükte VRAM taşmasını önlemek için)
  - Augmentation: copy_paste = 0.3 (Gözlük ve eldivenlerin yapay yapıştırılma oranı artırıldı)
  - Kayıp Ağırlı: cls = 1.0 (Sınıflandırma hata cezası 2 katına çıkarıldı, goggles/gloves odaklı)
  - Dinamik Veri Temizleme: Eğitim başlamadan önce etiket hataları (insansız ppe vb.) otomatik elenir.
"""

import os
import zipfile
import random
import subprocess
import shutil
from pathlib import Path

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
GLOVES_INPUT_DIR = KAGGLE_INPUT / "gloves-ppe-zips-cleaned"

# Sınıf ID'leri
CLASS_PERSON = 0
CLASS_HELMET_POS = 1
CLASS_HELMET_NEG = 2
CLASS_VEST_POS = 3
CLASS_VEST_NEG = 4
CLASS_GLOVES_POS = 5
CLASS_GLOVES_NEG = 6
CLASS_GOGGLES_POS = 7
CLASS_GOGGLES_NEG = 8
CLASS_SMOKING = 9

# Gloves & Goggles ZIP eşleştirmeleri
ZIP_CONFIGS = [
    ("gloves_v1i_yolov8.zip", {0: CLASS_GLOVES_NEG, 1: CLASS_GLOVES_POS}, "GLOVES.v1i"),
    ("gloves_v1i_yolov8_2.zip", {0: CLASS_GLOVES_POS, 1: CLASS_GLOVES_NEG}, "gloves.v1i(2)"),
    ("gloves_detection_v1i_yolov8.zip", {0: CLASS_GLOVES_POS}, "Gloves Detection (pos)"),
    ("gloves_detection_v4i_yolov8.zip", {0: CLASS_GLOVES_POS}, "gloves detection v4i"),
    ("gloves_v1i_yolov8_1.zip", {0: CLASS_GLOVES_POS}, "gloves.v1i(1)"),
    ("safety_glasses_v5.zip", {0: CLASS_GOGGLES_POS, 1: CLASS_GOGGLES_NEG}, "Safety Glasses v5"),
]

VAL_RATIO  = 0.15
TEST_RATIO = 0.05

# ─── Adım 1: combined_ppe'yi çıkart ─────────────────────────────────────────
def extract_combined():
    if (DATASET_DIR / "train" / "images").exists():
        print(f"[OK] combined_ppe zaten mevcut.")
        return

    print(f"Extracting {COMBINED_ZIP} ...")
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(COMBINED_ZIP, "r") as zf:
        for member in zf.infolist():
            clean_name = member.filename.replace("\\", "/")
            dest_path = DATASET_DIR.parent / clean_name
            if member.is_dir() or clean_name.endswith("/"):
                dest_path.mkdir(parents=True, exist_ok=True)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
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
    for zip_filename, class_map, desc in ZIP_CONFIGS:
        zip_path = GLOVES_INPUT_DIR / zip_filename
        folder_name = zip_filename.replace(".zip", "")
        folder_path = GLOVES_INPUT_DIR / folder_name

        if folder_path.exists() and folder_path.is_dir():
            print(f"\n[DIR] {desc}")
            img_files = []
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                img_files.extend(folder_path.glob(f"**/images/*{ext}"))

            for img_p in img_files:
                parts = list(img_p.parts)
                try:
                    idx = len(parts) - parts[::-1].index("images") - 1
                    parts[idx] = "labels"
                    lbl_p = Path(*parts).with_suffix(".txt")
                except ValueError:
                    continue

                if not lbl_p.exists():
                    continue

                raw = lbl_p.read_text(encoding="utf-8", errors="ignore")
                converted = convert_label(raw.splitlines(), class_map)
                if not converted:
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
                    continue

                shutil.copy2(img_p, dst_img)
                dst_lbl.write_text("\n".join(converted) + "\n", encoding="utf-8")

        elif zip_path.exists():
            print(f"\n[ZIP] {desc}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                all_files = zf.namelist()
                img_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png")) and "/images/" in f]

                for img_path in img_files:
                    lbl_path = img_path.replace("/images/", "/labels/")
                    dot = lbl_path.rfind(".")
                    lbl_path = lbl_path[:dot] + ".txt"

                    if lbl_path not in all_files:
                        continue

                    raw = zf.read(lbl_path).decode("utf-8", errors="ignore")
                    converted = convert_label(raw.splitlines(), class_map)
                    if not converted:
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
                        continue

                    dst_img.write_bytes(zf.read(img_path))
                    dst_lbl.write_text("\n".join(converted) + "\n", encoding="utf-8")
    print("[OK] Merge tamamlandi")

# ─── Adım 3: Dinamik Veri Temizleme (Dynamic Label Cleaning) ──────────────────
def clean_dataset_labels():
    print("\n" + "=" * 55)
    print("DINAMIK VERI ETIKET TEMIZLEME PIPELINE TETIKLENDI")
    print("=" * 55)
    
    stats = {
        "total_files": 0,
        "cleaned_files": 0,
        "deleted_files": 0,
        "out_of_bounds_fixed": 0,
        "invalid_size_removed": 0,
        "duplicates_removed": 0,
        "missing_person_context": 0
    }
    
    label_files = list(DATASET_DIR.glob("**/labels/*.txt"))
    stats["total_files"] = len(label_files)
    
    for file_path in label_files:
        if not file_path.exists():
            continue
            
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        valid_boxes = []
        has_person = False
        has_body_ppe = False  # Sadece baret veya yelek var mı? (Gözlük/Eldiven yakın çekimlerinde insan kutusu aranmaz)
        seen_boxes = set()
        file_changed = False
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
                
            try:
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                file_changed = True
                continue
                
            # Sınır kontrolü (clip)
            orig_coords = (x, y, w, h)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            if (x, y, w, h) != orig_coords:
                stats["out_of_bounds_fixed"] += 1
                file_changed = True
                
            # Sıfır boyut kontrolü
            if w <= 0.001 or h <= 0.001:
                stats["invalid_size_removed"] += 1
                file_changed = True
                continue
                
            # Mükerrer kontrolü
            box_key = (cls_id, round(x, 4), round(y, 4), round(w, 4), round(h, 4))
            if box_key in seen_boxes:
                stats["duplicates_removed"] += 1
                file_changed = True
                continue
            seen_boxes.add(box_key)
            
            if cls_id == CLASS_PERSON:
                has_person = True
            elif cls_id in [CLASS_HELMET_POS, CLASS_HELMET_NEG, CLASS_VEST_POS, CLASS_VEST_NEG]:
                has_body_ppe = True
                
            valid_boxes.append((cls_id, x, y, w, h))
            
        # Kural: Baret veya yelek (gövde İSG) var ama insan kutusu yoksa -> Dosyayı temizle
        if has_body_ppe and not has_person:
            stats["missing_person_context"] += 1
            valid_boxes = []
            file_changed = True
            
        if file_changed:
            stats["cleaned_files"] += 1
            if not valid_boxes:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")
                stats["deleted_files"] += 1
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    for box in valid_boxes:
                        f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                        
    print(f"  Taranan Toplam Dosya               : {stats['total_files']:,}")
    print(f"  Temizlenen/Guncellenen Dosya       : {stats['cleaned_files']:,}")
    print(f"  Sinir Disi Koordinat Duzeltme     : {stats['out_of_bounds_fixed']:,}")
    print(f"  Mukerrer Kutu Eleme                : {stats['duplicates_removed']:,}")
    print(f"  Insansiz PPE Temizleme (Eleme)     : {stats['missing_person_context']:,}")
    print("[OK] Veri temizleme adimi tamamlandi.")

# ─── Adım 3.5: Dinamik Sınıf Dengeleme (Dynamic Stratification) ───────────────
def stratify_dataset_labels():
    print("\n" + "=" * 55)
    print("DINAMIK SINIF DENGELEME (STRATIFICATION) PIPELINE")
    print("=" * 55)
    
    import random
    import shutil
    
    # Target classes to check and balance: {class_id: (min_valid_files, min_test_files)}
    TARGET_CONFIGS = {
        8: (100, 50),  # goggles_neg için en az 100 valid, 50 test
        7: (200, 100), # goggles_pos için en az 200 valid, 100 test
    }
    
    train_labels_dir = DATASET_DIR / "train" / "labels"
    train_images_dir = DATASET_DIR / "train" / "images"
    
    for cid, (min_valid, min_test) in TARGET_CONFIGS.items():
        # Mevcut valid/test sayılarını sayalım
        valid_labels_dir = DATASET_DIR / "valid" / "labels"
        test_labels_dir = DATASET_DIR / "test" / "labels"
        
        valid_existing = 0
        if valid_labels_dir.exists():
            for f in valid_labels_dir.glob("*.txt"):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as file_read:
                        if any(int(line.split()[0]) == cid for line in file_read if line.strip().split()):
                            valid_existing += 1
                except:
                    pass
                    
        test_existing = 0
        if test_labels_dir.exists():
            for f in test_labels_dir.glob("*.txt"):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as file_read:
                        if any(int(line.split()[0]) == cid for line in file_read if line.strip().split()):
                            test_existing += 1
                except:
                    pass
        
        needed_valid = max(0, min_valid - valid_existing)
        needed_test = max(0, min_test - test_existing)
        
        if needed_valid == 0 and needed_test == 0:
            print(f"  [OK] Sınıf {cid} zaten dengeli. Valid: {valid_existing}, Test: {test_existing}")
            continue
            
        print(f"  Sınıf {cid} için eksik bulundu -> Validasyon İhtiyacı: {needed_valid}, Test İhtiyacı: {needed_test}")
        
        # Train içindeki eşleşen dosyaları bul
        matching_files = []
        for file_path in train_labels_dir.glob("*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    if any(int(line.strip().split()[0]) == cid for line in f if line.strip().split()):
                        matching_files.append(file_path)
            except:
                pass
                
        total_found = len(matching_files)
        if total_found < (needed_valid + needed_test):
            print(f"  [!] Uyarı: Sınıf {cid} için yeterli kaynak dosya yok. Bulunan: {total_found}")
            needed_valid = int(total_found * 0.7)
            needed_test = total_found - needed_valid
            
        random.seed(42)
        random.shuffle(matching_files)
        
        to_valid = matching_files[:needed_valid]
        to_test = matching_files[needed_valid:needed_valid + needed_test]
        
        def move_pair(label_path, target_split):
            dest_lbl = DATASET_DIR / target_split / "labels" / label_path.name
            dest_lbl.parent.mkdir(parents=True, exist_ok=True)
            
            img_name = label_path.stem + ".jpg"
            src_img = train_images_dir / img_name
            if not src_img.exists():
                for ext in [".jpeg", ".png", ".JPG", ".PNG"]:
                    if (train_images_dir / (label_path.stem + ext)).exists():
                        src_img = train_images_dir / (label_path.stem + ext)
                        img_name = label_path.stem + ext
                        break
                        
            if not src_img.exists():
                return False
                
            dest_img = DATASET_DIR / target_split / "images" / img_name
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(label_path), str(dest_lbl))
            shutil.move(str(src_img), str(dest_img))
            return True

        moved_valid = sum(1 for lbl in to_valid if move_pair(lbl, "valid"))
        moved_test = sum(1 for lbl in to_test if move_pair(lbl, "test"))
        
        print(f"  Sınıf {cid} için validasyon setine {moved_valid}, test setine {moved_test} adet taşındı.")
    print("[OK] Sınıf dengeleme adımı tamamlandı.")

# ─── Adım 4: data.yaml yaz ───────────────────────────────────────────────────
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

# ─── Adım 5: Eğitim ──────────────────────────────────────────────────────────
def train():
    print("\n" + "=" * 55)
    print("EGITIM BASLIYOR — SEANS 5 (Seans 4 best.pt'den fine-tune)")
    print("=" * 55)

    CHECKPOINT_PT = KAGGLE_INPUT / "kkd-v4-checkpoint" / "best.pt"
    if not CHECKPOINT_PT.exists():
        raise FileNotFoundError(f"[HATA] Checkpoint bulunamadi: {CHECKPOINT_PT}")
        
    print(f"[OK] Checkpoint yukleniyor: {CHECKPOINT_PT}  ({CHECKPOINT_PT.stat().st_size/1e6:.1f} MB)")
    model = YOLO(str(CHECKPOINT_PT))

    LAST_DST = KAGGLE_WORKING / "last_checkpoint.pt"

    def on_train_epoch_end(trainer):
        last_src = Path(trainer.save_dir) / "weights" / "last.pt"
        if last_src.exists():
            shutil.copy2(str(last_src), str(LAST_DST))
            print(f"  [checkpoint] last.pt kaydedildi -> {LAST_DST}  (epoch {trainer.epoch+1})")

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    results = model.train(
        data        = str(DATA_YAML),
        epochs      = 150,
        patience    = 25,           # 25 epoch iyilesmezse dur
        batch       = 8,            # 960px icin batch 8'e dusuruldu (OOM engellemek icin)
        imgsz       = 960,          # Çözünürlük 640 -> 960 yapıldı (Küçük nesne hassasiyeti)
        device      = 0,

        # Optimizer
        optimizer   = "AdamW",
        lr0         = 0.00005,      # Fine-tune kararlılığı için düşük lr
        lrf         = 0.001,
        cos_lr      = True,
        momentum    = 0.937,
        weight_decay= 0.0005,

        # Overfitting engelleme
        dropout     = 0.1,

        # Augmentation
        mixup       = 0.15,
        copy_paste  = 0.3,          # Gözlük/Eldiven yapıştırma oranı 0.1 -> 0.3 yükseltildi
        mosaic      = 1.0,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        fliplr      = 0.5,
        scale       = 0.5,
        erasing     = 0.4,

        # Sınıf Kaybı Ağırlıklandırması (Gözlük/Eldiven sınıflarını daha sert cezalandırır)
        cls         = 1.0,          # YOLO varsayılanı 0.5'ten 1.0'a çıkarıldı

        # Kayıt
        project     = str(KAGGLE_WORKING / "runs" / "detect"),
        name        = "custom_ppe_v4",
        exist_ok    = True,
        save        = True,
        plots       = True,
        verbose     = True,
        workers     = 4,
        cache       = "ram",
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
    print("=== KKD Sürüm V15 Egitim Pipeline ===\n")
    extract_combined()        # Adım 1
    merge_gloves()            # Adım 2
    clean_dataset_labels()    # Adım 3 (YENİ - Dinamik Temizleme)
    stratify_dataset_labels() # Adım 3.5 (YENİ - Sınıf Dengeleme)
    write_yaml()              # Adım 4
    train()                   # Adım 5
