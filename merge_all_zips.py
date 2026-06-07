import os
import glob
import shutil
import zipfile
import yaml

# =====================================================================
# Devasa Veri Birleştirici (Mega Dataset Merger)
# =====================================================================

RAW_DIR = "datasets/raw_downloads"
DEST_IMAGES = "datasets/combined_ppe/train/images"
DEST_LABELS = "datasets/combined_ppe/train/labels"
TEMP_DIR = "datasets/temp_extract"

# Bizim Global Sınıf Numaralarımız
# 0: person, 1: helmet_pos, 2: helmet_neg, 3: vest_pos, 4: vest_neg
# 5: gloves_pos, 6: gloves_neg, 7: goggles_pos, 8: goggles_neg, 9: smoking

def get_global_id(class_name):
    n = str(class_name).lower().replace('-', ' ').replace('_', ' ')
    
    # Önce "yokluk" bildiren kelimeleri kontrol et
    if 'no ' in n or 'no' == n.split()[0] or 'without' in n:
        if 'helmet' in n or 'hardhat' in n or 'helmate' in n: return 2
        if 'vest' in n or 'jacket' in n: return 4
        if 'glove' in n: return 6
        if 'goggle' in n or 'glass' in n or 'face' in n: return 8
        return None # no mask, no shoes gibi bizim alakasız yoklukları atla
        
    # Pozitif sınıfları kontrol et
    if 'helmet' in n or 'hardhat' in n or 'helmate' in n: return 1
    if 'vest' in n or 'jacket' in n: return 3
    if 'glove' in n: return 5
    if 'goggle' in n or 'glass' in n: return 7
    
    # İnsan
    if 'person' in n or 'worker' in n:
        if 'unsafe' in n:
            return None # Bütün adama çizilmiş kutuyu atla, bize parçalar lazım
        return 0
        
    return None # Ayakkabı, makine, maske vs çöpe gitsin (Arka plan olur)

def process_zip(zip_path):
    zip_name = os.path.basename(zip_path)
    prefix = os.path.splitext(zip_name)[0].replace(' ', '_') + "_"
    
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    
    print(f"\n[{zip_name}] Çıkartılıyor...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as archive:
            archive.extractall(TEMP_DIR)
    except Exception as e:
        print(f"Zip çıkartma hatası: {e}")
        return
        
    # data.yaml bul
    yaml_files = glob.glob(os.path.join(TEMP_DIR, "**/*.yaml"), recursive=True)
    yaml_files = [y for y in yaml_files if 'data' in y.lower() or 'dataset' in y.lower()]
    
    if not yaml_files:
        print(f"[{zip_name}] Yaml dosyası bulunamadı. Atlanıyor.")
        return
        
    with open(yaml_files[0], 'r') as f:
        data = yaml.safe_load(f)
        
    if 'names' not in data:
        print(f"[{zip_name}] Sınıf isimleri bulunamadı. Atlanıyor.")
        return
        
    # Local id'leri Global id'lere eşleştir
    local_to_global = {}
    names = data['names']
    
    print(f"--- Sınıf Eşleştirmeleri ---")
    if isinstance(names, list):
        for idx, name in enumerate(names):
            gid = get_global_id(name)
            local_to_global[idx] = gid
            print(f"Local {idx} ({name}) -> Global {gid}")
    elif isinstance(names, dict):
        for idx, name in names.items():
            gid = get_global_id(name)
            local_to_global[idx] = gid
            print(f"Local {idx} ({name}) -> Global {gid}")
            
    # Resim ve etiketleri bul
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        all_images.extend(glob.glob(os.path.join(TEMP_DIR, "**", ext), recursive=True))
        
    print(f"Bulunan resim sayısı: {len(all_images)}")
    moved_count = 0
    
    for img_path in all_images:
        img_dir = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(img_name)[0]
        
        # Etiketi bul (genelde aynı klasörde veya ../labels/ içinde olur)
        txt_path = os.path.join(img_dir, name_no_ext + ".txt")
        if not os.path.exists(txt_path):
            # roboflow'da genelde images/ ve labels/ ayrıdır
            parent_dir = os.path.dirname(img_dir)
            if os.path.basename(img_dir) == 'images':
                txt_path = os.path.join(parent_dir, 'labels', name_no_ext + ".txt")
                
        if not os.path.exists(txt_path):
            continue # Etiketi yoksa atla
            
        # Etiketi oku ve çevir
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            local_id = int(parts[0])
            global_id = local_to_global.get(local_id)
            
            if global_id is not None:
                new_line = f"{global_id} " + " ".join(parts[1:]) + "\n"
                new_lines.append(new_line)
                
        # Dosyaları taşı (Çakışmayı önlemek için başa prefix ekle)
        new_img_name = prefix + img_name
        new_txt_name = prefix + name_no_ext + ".txt"
        
        dest_img_path = os.path.join(DEST_IMAGES, new_img_name)
        dest_txt_path = os.path.join(DEST_LABELS, new_txt_name)
        
        shutil.copy2(img_path, dest_img_path)
        with open(dest_txt_path, 'w') as f:
            f.writelines(new_lines)
            
        moved_count += 1
        
    print(f"[{zip_name}] Başarıyla {moved_count} veri işlendi ve aktarıldı.")
    
if __name__ == "__main__":
    os.makedirs(DEST_IMAGES, exist_ok=True)
    os.makedirs(DEST_LABELS, exist_ok=True)
    
    zips = glob.glob(os.path.join(RAW_DIR, "*.zip"))
    print(f"TOPLAM İŞLENECEK ZIP: {len(zips)}")
    
    for z in zips:
        process_zip(z)
        
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        
    print("\nTÜM İŞLEMLER BAŞARIYLA TAMAMLANDI!")
