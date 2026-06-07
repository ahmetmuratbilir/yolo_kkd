import os
import glob
import shutil

# =====================================================================
# Kaggle Veri Seti Düzenleyici ve Sınıf Eşleştirici
# =====================================================================
# Bu script, Kaggle'dan karışık halde indirdiğiniz resim ve etiketleri,
# sizin projenizin sınıflarına çevirip doğru 'images' ve 'labels' 
# klasörlerine dağıtır.

# Kaggle Sınıfları (Dataset ekran görüntüsüne göre)
# 0: helmet, 1: no_helmet, 2: no_vest, 3: person, 4: vest

# Sizin Sınıflarınız
# 0: person, 1: helmet_pos, 2: helmet_neg, 3: vest_pos, 4: vest_neg, ...

CLASS_MAP = {
    0: 1,  # helmet -> helmet_pos
    1: 2,  # no_helmet -> helmet_neg
    2: 4,  # no_vest -> vest_neg
    3: 0,  # person -> person
    4: 3   # vest -> vest_pos
}

def process_and_distribute(raw_dir, dest_images_dir, dest_labels_dir):
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_labels_dir, exist_ok=True)
    
    txt_files = glob.glob(os.path.join(raw_dir, "*.txt"))
    
    if not txt_files:
        print(f"[HATA] {raw_dir} içinde hiç .txt dosyası bulunamadı!")
        return

    processed_count = 0
    
    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        name_without_ext = os.path.splitext(filename)[0]
        
        # Olası resim uzantılarını kontrol et
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            possible_path = os.path.join(raw_dir, name_without_ext + ext)
            if os.path.exists(possible_path):
                img_path = possible_path
                break
                
        if not img_path:
            # Resim yoksa bu txt'yi atla
            continue
            
        # 1. Etiketi Oku ve Dönüştür
        with open(txt_path, "r") as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            class_id = int(parts[0])
            if class_id in CLASS_MAP:
                new_class_id = CLASS_MAP[class_id]
                new_line = f"{new_class_id} " + " ".join(parts[1:]) + "\n"
                new_lines.append(new_line)
                
        # 2. Yeni etiketi hedef 'labels' klasörüne yaz
        out_txt_path = os.path.join(dest_labels_dir, filename)
        with open(out_txt_path, "w") as f:
            f.writelines(new_lines)
            
        # 3. Resmi hedef 'images' klasörüne kopyala
        out_img_path = os.path.join(dest_images_dir, os.path.basename(img_path))
        shutil.copy2(img_path, out_img_path)
        
        processed_count += 1
        
    print("="*50)
    print(f"✅ İŞLEM TAMAMLANDI")
    print(f"📂 İşlenen, Dönüştürülen ve Taşınan Dosya Çifti : {processed_count}")
    print(f"🖼️ Resimler şuraya gitti : {dest_images_dir}")
    print(f"📝 Etiketler şuraya gitti: {dest_labels_dir}")
    print("="*50)

if __name__ == "__main__":
    # Karışık indirdiğiniz dosyaların atılacağı klasör
    RAW_DOWNLOADS_DIR = "datasets/raw_downloads"
    
    # Sizin ana projenizin eğitim klasörleri
    DEST_IMAGES = "datasets/combined_ppe/train/images"
    DEST_LABELS = "datasets/combined_ppe/train/labels"
    
    process_and_distribute(RAW_DOWNLOADS_DIR, DEST_IMAGES, DEST_LABELS)
