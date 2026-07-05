import os
from pathlib import Path

def clean_yolo_labels(dataset_path):
    print("=" * 60)
    print(f"VERI SETI ETIKET TEMIZLEME BASLADI: {dataset_path}")
    print("=" * 60)
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"[!] Hata: {dataset_path} dizini bulunamadı.")
        return
        
    stats = {
        "total_files": 0,
        "cleaned_files": 0,
        "deleted_files": 0,
        "out_of_bounds_fixed": 0,
        "invalid_size_removed": 0,
        "duplicates_removed": 0,
        "missing_person_context": 0
    }
    
    # YOLO Sınıfları
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
    
    # Tüm txt dosyalarını tara
    label_files = list(dataset_path.glob("**/labels/*.txt"))
    stats["total_files"] = len(label_files)
    
    print(f"Toplam {stats['total_files']:,} etiket dosyası taranıyor...")
    
    for file_path in label_files:
        if not file_path.exists():
            continue
            
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        valid_boxes = []
        has_person = False
        has_body_ppe = False  # Baret veya yelek var mı? (Gözlük/Eldiven yakın çekimlerinde insan kutusu aranmaz)
        
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
                # Sayısal olmayan hatalı satırları atla
                file_changed = True
                continue
                
            # 1. Koordinat Sınır Kontrolü (Out of bounds check)
            # YOLO'da değerler [0, 1] aralığında olmalıdır. Küçük sapmaları clip edelim.
            orig_coords = (x, y, w, h)
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            if (x, y, w, h) != orig_coords:
                stats["out_of_bounds_fixed"] += 1
                file_changed = True
                
            # 2. Geçersiz Boyut Kontrolü (Genişlik veya yükseklik sıfır/negatif olamaz)
            if w <= 0.001 or h <= 0.001:
                stats["invalid_size_removed"] += 1
                file_changed = True
                continue
                
            # 3. Mükerrer (Duplicate) Kutu Kontrolü
            # Aynı sınıftan aynı koordinatlara sahip kutuları teke indirgeyelim
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
            
        # 4. Kural Kontrolü: Baret veya yelek (gövde İSG) var ama Person (insan) kutusu yoksa
        if has_body_ppe and not has_person:
            stats["missing_person_context"] += 1
            valid_boxes = [] # Boşaltarak dosyayı sileceğiz veya temizleyeceğiz.
            file_changed = True
            
        # Değişiklikleri dosyaya yaz veya temizle
        if file_changed:
            stats["cleaned_files"] += 1
            if not valid_boxes:
                # Eğer hiç geçerli kutu kalmadıysa etiket dosyasını (ve varsa resmini) silebilir veya boş bırakabiliriz.
                # YOLO boş txt dosyalarını "background image" (arka plan) olarak görür. Biz boş etiket olarak kaydedelim.
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("")
                stats["deleted_files"] += 1
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    for box in valid_boxes:
                        f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                        
    print("\n" + "=" * 60)
    print("TEMIZLIK SONUÇLARI VE RAPORU:")
    print("=" * 60)
    print(f"Taranan Toplam Dosya               : {stats['total_files']:,}")
    print(f"Düzeltilen / Temizlenen Dosya      : {stats['cleaned_files']:,}")
    print(f"Sınır Dışına Çıkan Koordinatlar   : {stats['out_of_bounds_fixed']:,} adet")
    print(f"Silinen Sıfır/Bozuk Nesneler       : {stats['invalid_size_removed']:,} adet")
    print(f"Elenen Mükerrer (Aynı) Kutular     : {stats['duplicates_removed']:,} adet")
    print(f"İnsansız PPE Tespit Edilen Dosyalar: {stats['missing_person_context']:,} adet (Temizlendi)")
    print(f"Tamamen Boşaltılan Etiket Dosyaları : {stats['deleted_files']:,}")
    print("=" * 60)

if __name__ == "__main__":
    # Yerel veri seti yolu
    local_dataset = "c:/Users/ahmet murat bilir/Desktop/nukleerraktoruygulaması/okuldakiler/yolo_egitim/datasets/combined_ppe"
    clean_yolo_labels(local_dataset)
