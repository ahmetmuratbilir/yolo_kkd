import os
import shutil
from pathlib import Path

# combined_ppe siniflari (hedef)
# 0: person, 1: helmet_pos, 2: helmet_neg, 3: vest_pos, 4: vest_neg
# 5: gloves_pos, 6: gloves_neg, 7: goggles_pos, 8: goggles_neg, 9: smoking

GUANTES_MAP = {
    0: 5,  # glove       -> gloves_pos
    1: 7,  # goggles     -> goggles_pos
    2: 6,  # no_glove    -> gloves_neg
    3: 8,  # no_goggles  -> goggles_neg
}

GLASSES_MAP = {
    0: 7,  # eyes_with_goggles    -> goggles_pos
    1: 8,  # eyes_without_goggles -> goggles_neg
    2: 0,  # head                 -> person
    3: 1,  # head_with_helmet     -> helmet_pos
}

DEST = Path("datasets/combined_ppe")
SPLITS = ["train", "valid", "test"]

def remap_label(src_label, dst_label, class_map):
    if not src_label.exists():
        return
    lines = src_label.read_text().strip().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_cls = int(parts[0])
        if old_cls in class_map:
            parts[0] = str(class_map[old_cls])
            new_lines.append(" ".join(parts))
    if new_lines:
        dst_label.write_text("\n".join(new_lines))

def merge_dataset(src_root, class_map, prefix):
    copied_imgs = 0
    copied_labels = 0
    skipped = 0

    for split in SPLITS:
        src_imgs = src_root / split / "images"
        src_lbls = src_root / split / "labels"
        dst_imgs = DEST / split / "images"
        dst_lbls = DEST / split / "labels"
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        if not src_imgs.exists():
            continue

        for img in src_imgs.iterdir():
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            new_name = f"{prefix}_{img.name}"
            dst_img = dst_imgs / new_name
            shutil.copy2(img, dst_img)
            copied_imgs += 1

            lbl_file = src_lbls / (img.stem + ".txt")
            dst_lbl = dst_lbls / (Path(new_name).stem + ".txt")
            remap_label(lbl_file, dst_lbl, class_map)
            if dst_lbl.exists():
                copied_labels += 1
            else:
                skipped += 1

    print(f"  [OK] {prefix}: {copied_imgs} resim, {copied_labels} label kopyalandi, {skipped} label atlandi")
    return copied_imgs

print("=" * 50)
print("Dataset Birlestirme Basliyor...")
print("=" * 50)

print("\n[1/2] Guantes (Eldiven + Gozluk) ekleniyor...")
n1 = merge_dataset(Path("tmp_guantes"), GUANTES_MAP, "guantes")

print("\n[2/2] Safety Glasses (Gozluk) ekleniyor...")
n2 = merge_dataset(Path("tmp_glasses"), GLASSES_MAP, "glasses")

print("\n" + "=" * 50)
print(f"TAMAMLANDI! Toplam eklenen resim: {n1 + n2}")

total = sum(
    len(list((DEST / s / "images").glob("*")))
    for s in SPLITS
    if (DEST / s / "images").exists()
)
print(f"Combined PPE toplam resim: {total}")
print("=" * 50)
