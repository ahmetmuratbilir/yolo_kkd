import shutil
from pathlib import Path

# combined_ppe hedef siniflar:
# 0:person 1:helmet_pos 2:helmet_neg 3:vest_pos 4:vest_neg
# 5:gloves_pos 6:gloves_neg 7:goggles_pos 8:goggles_neg 9:smoking

DATASETS = {
    "ds1":  {0: 7},                          # Goggles -> goggles_pos
    "ds2":  {0: 7},                          # '0' -> goggles_pos
    "ds3":  {0: 8, 1: 2, 2: 7, 3: 1},       # Non-Goggle->neg, Non-Helmet->helmet_neg, Safety-Goggle->pos, Safety-Helmet->helmet_pos
    "ds4":  {0: 7},                          # Safety Goggles -> goggles_pos
    "ds5":  {0: 7},                          # Goggles -> goggles_pos
    "ds7":  {0: 7, 1: 8},                    # eyes_with->goggles_pos, eyes_without->goggles_neg
    "ds8":  {0: 7, 1: 0},                    # glasses->goggles_pos, person->person
    "ds9":  {0: 5},                          # object -> gloves_pos
    "ds10": {0: 5, 1: 6},                    # Glove->gloves_pos, No-Glove->gloves_neg
    "ds11": {0: 5, 1: 7},                    # Gloves->gloves_pos, Protective Glasses->goggles_pos
    "ds13": {0: 5},                          # gloves -> gloves_pos
    "ds14": {0: 6, 1: 5},                    # no_gloves->gloves_neg, with_gloves->gloves_pos
    "ds15": {0: 5},                          # Gloves -> gloves_pos
}

DEST = Path("datasets/combined_ppe")
SPLITS = ["train", "valid", "test"]
SRC_ROOT = Path("tmp_analyze")

def remap_label(src_label, dst_label, class_map):
    if not src_label.exists():
        return 0
    lines = src_label.read_text(errors='ignore').strip().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        # Segmentasyon formati kontrolu (5'ten fazla koordinat varsa atla)
        if len(parts) > 5:
            continue
        old_cls = int(float(parts[0]))
        if old_cls in class_map:
            parts[0] = str(class_map[old_cls])
            new_lines.append(" ".join(parts))
    if new_lines:
        dst_label.write_text("\n".join(new_lines))
        return len(new_lines)
    return 0

total_imgs = 0
total_labels = 0

print("=" * 55)
print("DATASET BIRLESTIRME BASLIYOR (13 dataset)")
print("=" * 55)

for ds_name, class_map in DATASETS.items():
    src = SRC_ROOT / ds_name
    if not src.exists():
        print(f"[HATA] {ds_name} bulunamadi, atlaniyor...")
        continue

    ds_imgs = 0
    ds_labels = 0

    for split in SPLITS:
        src_imgs = src / split / "images"
        src_lbls = src / split / "labels"
        dst_imgs = DEST / split / "images"
        dst_lbls = DEST / split / "labels"
        dst_imgs.mkdir(parents=True, exist_ok=True)
        dst_lbls.mkdir(parents=True, exist_ok=True)

        if not src_imgs.exists():
            continue

        for img in src_imgs.iterdir():
            if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            new_name = f"{ds_name}_{img.name}"
            shutil.copy2(img, dst_imgs / new_name)
            ds_imgs += 1

            lbl = src_lbls / (img.stem + ".txt")
            dst_lbl = dst_lbls / (Path(new_name).stem + ".txt")
            n = remap_label(lbl, dst_lbl, class_map)
            if n > 0:
                ds_labels += 1

    print(f"  [OK] {ds_name}: {ds_imgs} resim, {ds_labels} label")
    total_imgs += ds_imgs
    total_labels += ds_labels

print("\n" + "=" * 55)
print(f"TAMAMLANDI!")
print(f"  Eklenen resim : {total_imgs}")
print(f"  Eklenen label : {total_labels}")

grand_total = sum(
    len(list((DEST / s / "images").glob("*")))
    for s in SPLITS if (DEST / s / "images").exists()
)
print(f"  GENEL TOPLAM  : {grand_total} resim")
print("=" * 55)
