import os
import shutil
from pathlib import Path

# Kaggle'dan inen ham veri setinin yolu
KAGGLE_DIR = Path(r"C:\Users\ahmet murat bilir\.cache\kagglehub\datasets\snehilsanyal\construction-site-safety-image-dataset-roboflow\versions\3\css-data")

# Hedef klasörümüz (çalışma alanında)
TARGET_DIR = Path(r"c:\Users\ahmet murat bilir\Desktop\nukleerraktoruygulaması\yolo_egitim\datasets\kaggle_ppe")

# Kaggle Veri Setindeki Orijinal Sınıf ID'leri:
# 0: Hardhat
# 1: Mask
# 2: NO-Hardhat
# 3: NO-Mask
# 4: NO-Safety Vest
# 5: Person
# 6: Safety Cone
# 7: Safety Vest
# 8: machinery
# 9: vehicle

# Bizim Yeni Standart (Phase 2) Sınıf ID'lerimiz:
# 0: person
# 1: helmet_pos
# 2: helmet_neg
# 3: vest_pos
# 4: vest_neg
# 5: gloves_pos
# 6: gloves_neg
# 7: goggles_pos
# 8: goggles_neg

# Dönüşüm Haritası (Kaggle ID -> Bizim ID)
# None olanlar yoksayılacak (silinecek)
CLASS_MAPPING = {
    0: 1,  # Hardhat -> helmet_pos
    1: None, # Mask (İstemiyoruz)
    2: 2,  # NO-Hardhat -> helmet_neg
    3: None, # NO-Mask
    4: 4,  # NO-Safety Vest -> vest_neg
    5: 0,  # Person -> person
    6: None, # Safety Cone
    7: 3,  # Safety Vest -> vest_pos
    8: None, # machinery
    9: None  # vehicle
}

def process_labels_and_copy_images(split_name):
    print(f"[{split_name.upper()}] İşleniyor...")
    
    src_images = KAGGLE_DIR / split_name / "images"
    src_labels = KAGGLE_DIR / split_name / "labels"
    
    tgt_images = TARGET_DIR / split_name / "images"
    tgt_labels = TARGET_DIR / split_name / "labels"
    
    tgt_images.mkdir(parents=True, exist_ok=True)
    tgt_labels.mkdir(parents=True, exist_ok=True)
    
    if not src_labels.exists():
        print(f"Uyarı: {src_labels} bulunamadı!")
        return

    processed_count = 0
    ignored_count = 0
    
    for label_file in src_labels.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            orig_class = int(parts[0])
            if orig_class in CLASS_MAPPING and CLASS_MAPPING[orig_class] is not None:
                new_class = CLASS_MAPPING[orig_class]
                new_line = f"{new_class} " + " ".join(parts[1:])
                new_lines.append(new_line)
                
        # Sadece geçerli label'i olan resimleri veya genel olarak tüm resimleri kopyalayabiliriz.
        # YOLO arka plan resimlerini (labelsız) de kabul eder, ancak gereksiz yük olmasın diye
        # label yoksa da boş .txt oluşturup resmi kopyalayalım.
        with open(tgt_labels / label_file.name, "w") as f:
            f.write("\n".join(new_lines))
            
        # Resmi kopyala (.jpg veya .jpeg)
        img_name = label_file.stem + ".jpg"
        src_img_file = src_images / img_name
        
        # Eğer jpg yoksa png vs ara
        if not src_img_file.exists():
            for ext in [".jpeg", ".png", ".JPG", ".PNG"]:
                if (src_images / (label_file.stem + ext)).exists():
                    src_img_file = src_images / (label_file.stem + ext)
                    img_name = label_file.stem + ext
                    break
        
        if src_img_file.exists():
            shutil.copy2(src_img_file, tgt_images / img_name)
            processed_count += 1
        else:
            print(f"Resim bulunamadı: {img_name}")
            
    print(f"[{split_name.upper()}] Tamamlandı. {processed_count} görüntü kopyalandı.\n")

def main():
    print("Kaggle verileri yeni formata dönüştürülüp 'datasets/kaggle_ppe' klasörüne kopyalanıyor...\n")
    
    for split in ["train", "valid", "test"]:
        process_labels_and_copy_images(split)
        
    # data.yaml oluştur
    yaml_content = f"""path: {TARGET_DIR.absolute().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 9
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
"""
    with open(TARGET_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"Bitti! Tüm veriler şuraya kopyalandı: {TARGET_DIR}")
    print("data.yaml dosyası oluşturuldu. Artık YOLO eğitiminde kullanılabilir!")

if __name__ == "__main__":
    main()
