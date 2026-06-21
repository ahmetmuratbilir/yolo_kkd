"""
merge_gloves_to_combined.py
===========================
Eldiven ZIP datasetlerini mevcut combined_ppe dataseti ile birleştirir.

Yapılan işlemler:
  1. Her ZIP açılır
  2. Etiketler → gloves_pos (ID=5) / gloves_neg (ID=6) olarak dönüştürülür
  3. Resim + etiket dosyaları combined_ppe/train|valid|test klasörlerine kopyalanır
  4. data.yaml güncellenmez (sınıf yapısı aynı kalır)

Mevcut dataset:  100.437 train / 10.789 valid / 5.611 test
Hedef:           gloves_pos ve gloves_neg örnekleri artırılır

Kullanım:
  python merge_gloves_to_combined.py
"""

import zipfile
import os
import shutil
import random
from pathlib import Path

# ─── Ayarlar ────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent
DATASET    = BASE_DIR / "datasets" / "combined_ppe"

# Mevcut sınıf ID'leri (data.yaml ile uyumlu)
GLOVES_POS_ID = 5   # gloves_pos
GLOVES_NEG_ID = 6   # gloves_neg

# Validasyon oranı (yeni eklenen resimlerin %15'i val'e, %5'i test'e gider)
VAL_RATIO  = 0.15
TEST_RATIO = 0.05

# Kullanılacak ZIP'ler ve her birinin sınıf eşleştirmesi
# Format: (zip_dosyasi, {zip_sinif_id: hedef_sinif_id, ...}, aciklama)
# zip_sinif_id = ZIP'in kendi sınıf numarası (data.yaml'daki index)
# Sadece eşleştirmede belirtilen sınıflar alınır; diğerleri atlanır.
ZIP_CONFIGS = [
    (
        "GLOVES.v1i.yolov8.zip",
        # nc=2: 0=no_gloves, 1=with_gloves
        {0: GLOVES_NEG_ID, 1: GLOVES_POS_ID},
        "GLOVES.v1i (no_gloves / with_gloves)"
    ),
    (
        "gloves.v1i.yolov8 (2).zip",
        # nc=2: 0=Glove, 1=No-Glove  ← DİKKAT: 0=pos, 1=neg
        {0: GLOVES_POS_ID, 1: GLOVES_NEG_ID},
        "gloves.v1i(2) (Glove / No-Glove)"
    ),
    (
        "Gloves Detection.v1i.yolov8.zip",
        # nc=2: 0=Gloves, 1=Protective Glasses
        # Sadece Gloves (0) alıyoruz, Glasses'ı atlıyoruz
        {0: GLOVES_POS_ID},
        "Gloves Detection.v1i (sadece Gloves=pos)"
    ),
    (
        "gloves detection.v4i.yolov8.zip",
        # nc=1: 0=Gloves
        {0: GLOVES_POS_ID},
        "gloves detection.v4i (Gloves=pos)"
    ),
    (
        "gloves.v1i.yolov8 (1).zip",
        # nc=1: 0=gloves
        {0: GLOVES_POS_ID},
        "gloves.v1i(1) (gloves=pos)"
    ),
]

# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def convert_label_file(src_lines, class_map):
    """
    YOLO formatı etiket dosyasını dönüştürür.
    src_lines: ZIP'ten okunan etiket satırları listesi
    class_map: {eski_id: yeni_id} dict
    Döner: (dönüştürülmüş satırlar, atlandı mı?)
    """
    out = []
    for line in src_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        if old_id not in class_map:
            continue  # Bu ZIP'te kullanmadığımız sınıf — atla
        new_id = class_map[old_id]
        out.append(f"{new_id} " + " ".join(parts[1:]))
    return out


def choose_split():
    """Rastgele train/valid/test bölümü seç."""
    r = random.random()
    if r < TEST_RATIO:
        return "test"
    elif r < TEST_RATIO + VAL_RATIO:
        return "valid"
    else:
        return "train"


def safe_name(zip_name, orig_name, split):
    """
    Çakışmayı önlemek için dosya adına ZIP prefix'i ekler.
    """
    stem = Path(zip_name).stem.replace(" ", "_").replace(".", "_")
    orig_stem = Path(orig_name).stem
    suffix = Path(orig_name).suffix
    return f"{stem}__{orig_stem}{suffix}"


# ─── Ana İşlem ───────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    total_added = {"train": 0, "valid": 0, "test": 0}
    total_skipped = 0

    for zip_filename, class_map, description in ZIP_CONFIGS:
        zip_path = BASE_DIR / zip_filename
        if not zip_path.exists():
            print(f"  [!] BULUNAMADI: {zip_filename} -- atlaniyor")
            continue

        print(f"\n{'='*60}")
        print(f"[ZIP] Isleniyor: {description}")
        print(f"   Dosya: {zip_filename}")
        print(f"   Sinif eslestirmesi: {class_map}")

        added = {"train": 0, "valid": 0, "test": 0}
        skipped = 0

        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_files = zf.namelist()

            # Tüm resim dosyalarını bul
            img_files = [
                f for f in all_files
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                and '/images/' in f
            ]

            print(f"   Toplam resim: {len(img_files)}")

            for img_path in img_files:
                # Karşılık gelen etiket yolunu bul
                # ZIP namelist her zaman forward slash kullanır — Path() kullanma
                lbl_path = img_path.replace('/images/', '/labels/')
                dot = lbl_path.rfind('.')
                lbl_path = lbl_path[:dot] + '.txt'

                if lbl_path not in all_files:
                    # Etiketsiz resim — arka plan olabilir, atla
                    skipped += 1
                    continue

                # Etiketi oku ve dönüştür
                raw_lbl = zf.read(lbl_path).decode('utf-8', errors='ignore')
                converted = convert_label_file(raw_lbl.splitlines(), class_map)

                if not converted:
                    # Bu resimde hiç kullanılacak sınıf yok
                    skipped += 1
                    continue

                # Hangi split?
                split = choose_split()

                # Hedef yollar
                img_dst_dir = DATASET / split / "images"
                lbl_dst_dir = DATASET / split / "labels"
                img_dst_dir.mkdir(parents=True, exist_ok=True)
                lbl_dst_dir.mkdir(parents=True, exist_ok=True)

                new_name_img = safe_name(zip_filename, Path(img_path).name, split)
                new_name_lbl = Path(new_name_img).with_suffix('.txt').name

                img_dst = img_dst_dir / new_name_img
                lbl_dst = lbl_dst_dir / new_name_lbl

                # Zaten varsa atla (ikinci çalıştırma güvenliği)
                if img_dst.exists():
                    skipped += 1
                    continue

                # Resmi yaz
                img_data = zf.read(img_path)
                img_dst.write_bytes(img_data)

                # Etiketi yaz
                lbl_dst.write_text('\n'.join(converted) + '\n', encoding='utf-8')

                added[split] += 1

        print(f"   [OK] Eklendi  -> train:{added['train']} | valid:{added['valid']} | test:{added['test']}")
        print(f"   [--] Atlandi  -> {skipped}")

        for k in total_added:
            total_added[k] += added[k]
        total_skipped += skipped

    # ─── Özet ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TAMAMLANDI!")
    print(f"   Toplam eklenen -> train:{total_added['train']} | valid:{total_added['valid']} | test:{total_added['test']}")
    print(f"   Toplam eklenen = {sum(total_added.values())} resim")
    print(f"   Atlanan        = {total_skipped} resim")
    print()

    # Yeni dataset boyutunu göster
    for split in ['train', 'valid', 'test']:
        img_dir = DATASET / split / "images"
        if img_dir.exists():
            n = len(list(img_dir.iterdir()))
            print(f"   {split:5s} toplam: {n:,} resim")

    print()
    print("Sonraki adim: python train_v3_improved.py")
    print()
    print("NOT: data.yaml degistirilmedi -- sinif yapisi ayni kaliyor")
    print("     gloves_pos=ID5, gloves_neg=ID6")


if __name__ == "__main__":
    main()
