import os
import glob

# =====================================================================
# SH17 Veri Setinden -> Sizin Modelinize (custom_ppe_v3) Etiket Dönüştürücü
# =====================================================================
# SH17 veri setindeki 17 sınıfı (0-indexed), sizin 10 sınıflı sisteminize
# dönüştürür. İhtiyacınız olmayan sınıflar (ayakkabı vb.) silinir ve arka plan olur.

# SH17 YOLO sınıfları (0-indexed):
# 0: Person, 1: Head, 2: Face, 3: Glasses, 8: Hands, 9: Gloves, 
# 12: Safety-vest, 14: Helmet

CLASS_MAP = {
    0: 0,   # SH17 Person       -> Bizim person
    14: 1,  # SH17 Helmet       -> Bizim helmet_pos
    1: 2,   # SH17 Head         -> Bizim helmet_neg (Kasksız baş)
    12: 3,  # SH17 Safety-vest  -> Bizim vest_pos
    # Not: SH17'de vest_neg (yeleksiz) diye bir sınıf yok, onu sizin boş ortam çekimleriniz çözecek.
    9: 5,   # SH17 Gloves       -> Bizim gloves_pos
    8: 6,   # SH17 Hands        -> Bizim gloves_neg (Eldivensiz çıplak el)
    3: 7,   # SH17 Glasses      -> Bizim goggles_pos
    2: 8    # SH17 Face         -> Bizim goggles_neg (Gözlüksüz çıplak yüz)
}

def convert_sh17_labels(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    if not txt_files:
        print(f"[HATA] {input_dir} içinde .txt dosyası bulunamadı!")
        return

    converted_files = 0
    ignored_objects = 0
    mapped_objects = 0
    
    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        out_path = os.path.join(output_dir, filename)
        
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
                mapped_objects += 1
            else:
                # İhtiyacımız olmayan bir nesne (örn: ayakkabı). Çöpe at, arka plan olsun.
                ignored_objects += 1
                
        # Dosyayı yaz (Eğer içi boş kalırsa bilerek boş txt oluşturur, tam aradığımız arka plan çözümü)
        with open(out_path, "w") as f:
            f.writelines(new_lines)
            
        converted_files += 1
        
    print("="*50)
    print(f"✅ İŞLEM TAMAMLANDI")
    print(f"📂 Çevrilen Dosya Sayısı : {converted_files}")
    print(f"🎯 Eklenen Nesne Sayısı  : {mapped_objects}")
    print(f"🗑️ Silinen Nesne Sayısı  : {ignored_objects} (Arka plan yapıldı)")
    print(f"💾 Yeni etiketler şurada : {output_dir}")
    print("="*50)

if __name__ == "__main__":
    # TODO: İndirdiğiniz SH17 veri setindeki 'labels' klasörünün yolunu buraya yazın
    INPUT_LABELS_DIR = "datasets/sh17_labels_raw" 
    
    # Çevrilen etiketlerin kaydedileceği yeni klasör
    OUTPUT_LABELS_DIR = "datasets/sh17_labels_converted"
    
    convert_sh17_labels(INPUT_LABELS_DIR, OUTPUT_LABELS_DIR)
