import os
import zipfile
import shutil
import yaml
import glob
from pathlib import Path

# Yeni standart sınıflarımız
# 0: person
# 1: helmet_pos
# 2: helmet_neg
# 3: vest_pos
# 4: vest_neg
# 5: gloves_pos
# 6: gloves_neg
# 7: goggles_pos
# 8: goggles_neg
# 9: smoking

TARGET_DIR = Path("datasets/combined_ppe")

def get_class_mapping(original_classes):
    """Verilen orjinal sınıf listesini bizim hedef ID'lerimize eşler."""
    mapping = {}
    
    # Standart isimleri kontrol edelim (küçük harfe çevirip)
    for i, cls in enumerate(original_classes):
        c = cls.lower().strip()
        
        # Person
        if c in ["person", "a person"]: mapping[i] = 0
        
        # Helmet
        elif c in ["hardhat", "a hardhat", "hard_hat", "helmet"]: mapping[i] = 1
        elif c in ["no_hardhat", "no_hard_hat", "no-hardhat"]: mapping[i] = 2
        
        # Vest
        elif c in ["vest", "a vest", "safety vest"]: mapping[i] = 3
        elif c in ["no_vest", "no-safety vest"]: mapping[i] = 4
        
        # Gloves
        elif c in ["gloves", "a glove", "glove"]: mapping[i] = 5
        elif c in ["no_gloves", "no_glove"]: mapping[i] = 6
        
        # Goggles
        elif c in ["eye protection", "goggles", "glasses"]: mapping[i] = 7
        elif c in ["no_goggles", "no_glasses"]: mapping[i] = 8
        
        # Smoking
        elif c in ["smoking", "cigarette", "vape"]: mapping[i] = 9
        
        else:
            mapping[i] = None # Yoksay
            
    return mapping

def process_split(src_path, split_name, mapping):
    """train, valid veya test klasörünü işler ve kopyalar."""
    # Kaynak yollar (images, labels)
    src_images = src_path / split_name / "images"
    src_labels = src_path / split_name / "labels"
    
    if not src_images.exists() or not src_labels.exists():
        return 0
        
    tgt_images = TARGET_DIR / split_name / "images"
    tgt_labels = TARGET_DIR / split_name / "labels"
    tgt_images.mkdir(parents=True, exist_ok=True)
    tgt_labels.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for label_file in src_labels.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            orig_id = int(parts[0])
            if orig_id in mapping and mapping[orig_id] is not None:
                new_id = mapping[orig_id]
                new_lines.append(f"{new_id} " + " ".join(parts[1:]))
                
        img_name = label_file.stem + ".jpg" # YOLO genelde jpg kullanır
        src_img = src_images / img_name
        if not src_img.exists():
            # Farklı uzantı dene
            for ext in [".jpeg", ".png", ".JPG", ".PNG"]:
                if (src_images / (label_file.stem + ext)).exists():
                    src_img = src_images / (label_file.stem + ext)
                    img_name = label_file.stem + ext
                    break
                    
        if src_img.exists():
            import uuid
            new_stem = f"{label_file.stem}_{uuid.uuid4().hex[:6]}"
            new_img_name = new_stem + src_img.suffix
            new_lbl_name = new_stem + ".txt"
            
            with open(tgt_labels / new_lbl_name, "w") as f:
                f.write("\n".join(new_lines))
                
            shutil.copy2(src_img, tgt_images / new_img_name)
            count += 1
            
    return count

def main():
    import uuid
    # SADECE SMOKING VERILERINI CEKIYORUZ
    zips = glob.glob("Smoking*.zip")
    
    for z in zips:
        print(f"\nİşleniyor: {z}")
        
        unique_temp = Path(f"temp_extract_{uuid.uuid4().hex[:8]}")
        unique_temp.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(z, 'r') as zf:
                zf.extractall(unique_temp)
                
            yaml_files = list(unique_temp.rglob("data.yaml"))
            if not yaml_files:
                print("data.yaml bulunamadı, atlanıyor.")
                continue
                
            yaml_file = yaml_files[0]
            base_path = yaml_file.parent
            
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
                
            classes = data.get("names", [])
            mapping = get_class_mapping(classes)
            print(f"Orijinal Sınıflar: {classes}")
            print(f"Eşleşme Haritası: {mapping}")
            
            total_copied = 0
            for split in ["train", "valid", "test"]:
                c = process_split(base_path, split, mapping)
                total_copied += c
                
            print(f"Toplam {total_copied} görüntü aktarıldı.")
            
        finally:
            if unique_temp.exists():
                try:
                    shutil.rmtree(unique_temp)
                except Exception as e:
                    print(f"Klasör silinemedi {unique_temp}: {e}")
        
    # Final data.yaml oluştur
    yaml_content = f"""path: {TARGET_DIR.absolute().as_posix()}
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
    with open(TARGET_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print("\n[BASARILI] Smoking sınıfları 'datasets/combined_ppe' klasörüne eklendi!")

if __name__ == "__main__":
    main()
