import os
import cv2
from pathlib import Path
from services.detector import PPEDetector

INPUT_DIR = Path("test_images")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def test_model(model_path, images, conf=0.15):
    if not os.path.exists(model_path):
        print(f"Model bulunamadi: {model_path}")
        return
        
    print(f"\n==========================================")
    print(f"MODEL TEST EDILIYOR: {os.path.basename(model_path)}")
    print(f"==========================================")
    
    detector = PPEDetector(model_path)
    
    total_persons = 0
    total_helmets = 0
    total_vests = 0
    total_gloves = 0
    
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
            
        raw = detector.detect(frame, conf=conf)
        dets = detector.filter_by_class_conf(raw)
        
        persons = sum(1 for d in dets if d["class_name"] == "person")
        helmets = sum(1 for d in dets if d["class_name"] == "helmet_pos")
        vests   = sum(1 for d in dets if d["class_name"] == "vest_pos")
        gloves  = sum(1 for d in dets if d["class_name"] == "glove_pos")
        
        total_persons += persons
        total_helmets += helmets
        total_vests += vests
        total_gloves += gloves
        
    print(f"Toplam Gorsel: {len(images)}")
    print(f"Bulunan Kisi Sayisi : {total_persons}")
    print(f"Bulunan Kask Sayisi : {total_helmets}")
    print(f"Bulunan Yelek Sayisi: {total_vests}")
    print(f"Bulunan Eldiven Sys : {total_gloves}")


def main():
    image_paths = sorted(p for p in INPUT_DIR.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    
    # Sadece kullanicinin "Ekran" ile baslayan fotograflari uzerinde test edelim
    user_images = [p for p in image_paths if "Ekran" in p.name]
    
    if not user_images:
        print("Ekran goruntusu bulunamadi, tum fotograflari test edecegim.")
        user_images = image_paths

    test_model("models/ppe_model.pt", user_images, conf=0.15)
    test_model("models/vyra_yolo_ppe_best.pt", user_images, conf=0.15)
    test_model("models/tanish_yolov8n_ppe_6class.pt", user_images, conf=0.15)


if __name__ == "__main__":
    main()
