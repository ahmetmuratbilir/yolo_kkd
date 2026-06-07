import cv2
import os

# =====================================================================
# Boş Ortam (Arka Plan) Verisi Çıkarma Aracı
# =====================================================================
# Bu script, sahada işçiler yokken çektiğiniz videoyu saniyede 1 kare (FPS)
# olacak şekilde fotoğraflara böler ve yanlarına "içi boş" .txt etiketleri 
# oluşturarak YOLO'nun "Background" eğitim verisini hazırlar.

def extract_background_frames(video_path, output_dir):
    if not os.path.exists(video_path):
        print(f"[HATA] Video dosyası bulunamadı: {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Handle NaN or 0
        fps = 30.0
        
    frame_id = 0
    saved = 0

    print(f"[{video_path}] işleniyor. FPS: {fps:.2f}...")

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        # Saniyede tam 1 kare kaydet (Örn: FPS 30 ise her 30. kareyi al)
        if frame_id % int(fps) == 0:
            name = f"bg_{saved:04d}"
            image_path = os.path.join(output_dir, f"{name}.jpg")
            txt_path = os.path.join(output_dir, f"{name}.txt")
            
            # Görüntüyü kaydet
            cv2.imwrite(image_path, frame)
            
            # Yanına İÇİ BOŞ etiket (.txt) oluştur
            open(txt_path, "w").close()
            
            saved += 1
            
        frame_id += 1

    video.release()
    print("="*50)
    print(f"✅ İŞLEM TAMAMLANDI")
    print(f"🎬 Çıkarılan ve etiketlenen (boş) kare sayısı: {saved}")
    print(f"💾 Kayıt Yeri: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    # TODO: Çektiğiniz boş ortam videosunun yolunu buraya yazın
    VIDEO_PATH = "bos_saha_videosu.mp4" 
    
    # Çıkarılacak fotoğrafların ve boş etiketlerin kaydedileceği klasör
    OUTPUT_DIR = "datasets/dataset_negative"
    
    extract_background_frames(VIDEO_PATH, OUTPUT_DIR)
