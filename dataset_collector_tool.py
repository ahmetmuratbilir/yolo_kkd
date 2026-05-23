import cv2
import os
import time
from pathlib import Path

# Klasör yapısını tanımlayalım
BASE_DIR = Path("dataset_raw")
DIRS = {
    "helmet_pos": BASE_DIR / "helmet_pos",
    "helmet_neg": BASE_DIR / "helmet_neg",
    "vest_pos": BASE_DIR / "vest_pos",
    "vest_neg": BASE_DIR / "vest_neg",
    "gloves_pos": BASE_DIR / "gloves_pos",
    "gloves_neg": BASE_DIR / "gloves_neg",
    "goggles_pos": BASE_DIR / "goggles_pos",
    "goggles_neg": BASE_DIR / "goggles_neg",
    "misc_pos": BASE_DIR / "misc_pos" # Genel/karışık pozitifler için
}

# Tüm klasörleri oluştur
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

def save_image(frame, category):
    """Görüntüyü ilgili klasöre zaman damgasıyla kaydeder."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Dosya ismine milisaniye de ekleyelim ki seri basmalarda üst üste binmesin
    ms = int((time.time() % 1) * 1000)
    filename = f"{timestamp}_{ms}.jpg"
    filepath = DIRS[category] / filename
    
    cv2.imwrite(str(filepath), frame)
    print(f"[KAYDEDILDI] -> {category}/{filename}")

def main():
    print("====================================================")
    print(" VERI TOPLAMA ARACI (DATASET COLLECTOR) BASLATILIYOR")
    print("====================================================")
    print("KULLANIM (Kamera penceresi aktifken tuşlara basın):")
    print("  '1' -> helmet_pos  (Gerçek Kask)")
    print("  '2' -> helmet_neg  (Kask yok: şapka, kel, saç vb.)")
    print("  '3' -> vest_pos    (Gerçek İSG Yeleği)")
    print("  '4' -> vest_neg    (Yelek yok: sarı/yeşil tişört, atlet, çıplak)")
    print("  '5' -> gloves_pos  (Gerçek Eldiven)")
    print("  '6' -> gloves_neg  (Eldiven yok: çıplak el, sarı arka plan)")
    print("  '7' -> goggles_pos (Gerçek İş Gözlüğü)")
    print("  '8' -> goggles_neg (Gözlük yok: normal gözlük, güneş gözlüğü)")
    print("  'p' -> misc_pos    (Genel)")
    print("  'q' -> ÇIKIŞ")
    print("====================================================\n")

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("HATA: Kamera açılamadı!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü okunamadı!")
            break
            
        cv2.putText(frame, "1:Kask+ 2:Kask- 3:Yelek+ 4:Yelek- 5:Eld+ 6:Eld- 7:Goz+ 8:Goz-", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow("Veri Toplama (Dataset Collector)", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Çıkış yapılıyor...")
            break
        elif key == ord('1'): save_image(frame, "helmet_pos")
        elif key == ord('2'): save_image(frame, "helmet_neg")
        elif key == ord('3'): save_image(frame, "vest_pos")
        elif key == ord('4'): save_image(frame, "vest_neg")
        elif key == ord('5'): save_image(frame, "gloves_pos")
        elif key == ord('6'): save_image(frame, "gloves_neg")
        elif key == ord('7'): save_image(frame, "goggles_pos")
        elif key == ord('8'): save_image(frame, "goggles_neg")
        elif key == ord('p'): save_image(frame, "misc_pos")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
