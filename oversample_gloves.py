"""
oversample_gloves.py
====================
Gloves sınıfı (gloves_pos=5, gloves_neg=6) için oversampling yapar.

Neden gerekli?
  - Dataset ~103K resim ama gloves annotation'ları çok az
  - Model "az örnekli" sınıfı önemsiz sayabilir
  - Bu script gloves içeren resimleri N kez kopyalayarak dengeyi düzeltir

Kullanım:
  python oversample_gloves.py --factor 3   (3x kopya, önerilen)
  python oversample_gloves.py --factor 2   (2x kopya, muhafazakar)

Güvenlik:
  - Sadece train split'e dokunur (val/test'e asla)
  - İdempotent: tekrar çalıştırırsan "zaten var" diyip atlar
  - Orijinal dosyalara dokunmaz, sadece yeni kopyalar ekler
"""

import argparse
import random
from pathlib import Path
from shutil import copy2

DATASET       = Path("datasets/combined_ppe")
GLOVES_IDS    = {5, 6}   # gloves_pos, gloves_neg
TRAIN_IMG_DIR = DATASET / "train" / "images"
TRAIN_LBL_DIR = DATASET / "train" / "labels"


def has_gloves(lbl_path: Path) -> bool:
    """Etiket dosyasında gloves sınıfı var mı?"""
    try:
        for line in lbl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and int(line.split()[0]) in GLOVES_IDS:
                return True
    except Exception:
        pass
    return False


def oversample(factor: int):
    print(f"Oversampling basliyor (faktor={factor}x) ...")
    print(f"  Train klasoru: {TRAIN_LBL_DIR}")

    # Gloves iceren tum etiket dosyalarini bul
    gloves_files = []
    for lbl in TRAIN_LBL_DIR.iterdir():
        if lbl.suffix == ".txt" and has_gloves(lbl):
            gloves_files.append(lbl)

    print(f"  Gloves iceren train resim sayisi: {len(gloves_files)}")

    if not gloves_files:
        print("  [!] Hic gloves dosyasi bulunamadi! merge_gloves_to_combined.py calistirildi mi?")
        return

    added = 0
    skipped = 0

    for lbl_src in gloves_files:
        # Karsilik gelen resmi bul
        stem = lbl_src.stem
        img_src = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = TRAIN_IMG_DIR / (stem + ext)
            if candidate.exists():
                img_src = candidate
                break

        if img_src is None:
            skipped += 1
            continue

        # factor-1 kez kopyala (orijinal + factor-1 kopya = factor adet toplam)
        for i in range(1, factor):
            suffix_img = img_src.suffix
            new_stem   = f"{stem}__os{i}"
            dst_img    = TRAIN_IMG_DIR / (new_stem + suffix_img)
            dst_lbl    = TRAIN_LBL_DIR / (new_stem + ".txt")

            if dst_img.exists():
                skipped += 1
                continue

            copy2(img_src, dst_img)
            copy2(lbl_src, dst_lbl)
            added += 1

    print(f"\n  [OK] Eklenen kopya : {added}")
    print(f"  [--] Atlanan       : {skipped}")

    # Yeni toplam
    total_train = len(list(TRAIN_IMG_DIR.iterdir()))
    print(f"\n  Train yeni toplam  : {total_train:,} resim")
    print()

    # Gloves annotation orani tahmini
    gloves_count_approx = len(gloves_files) * factor
    print(f"  Tahmini gloves-iceren resim: ~{gloves_count_approx:,}")
    print(f"  Tahmini gloves orani       : ~%{gloves_count_approx/total_train*100:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=3,
                        help="Kac kat cogalt (default: 3)")
    args = parser.parse_args()

    if args.factor < 2:
        print("Factor en az 2 olmali.")
    else:
        oversample(args.factor)
