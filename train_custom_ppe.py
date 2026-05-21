from ultralytics import YOLO

def main():
    print("Eğitim (Training) Başlıyor...")
    print("Kullanılan Veri Seti: datasets/combined_ppe/data.yaml")
    print("Model: YOLOv8 Small (yolov8s.pt)")
    
    # YOLOv8 Small modelini yüklüyoruz (hız ve doğruluk açısından en dengeli model)
    model = YOLO("yolov8s.pt")
    
    # Eğitimi başlatıyoruz
    # device parametresi verilmediği için sistemde GPU (CUDA) varsa otomatik kullanır, yoksa CPU'da eğitir.
    results = model.train(
        data="datasets/combined_ppe/data.yaml",
        epochs=30,  # 30 Epoch başlangıç için ideal ve hızlıdır.
        imgsz=640,  # Görüntü boyutu
        batch=16,   # Batch size (RAM/VRAM'e göre otomatik de ayarlanabilir)
        name="custom_ppe_phase2", # Sonuçların kaydedileceği klasör ismi
        patience=10 # Model 10 epoch boyunca gelişmezse eğitimi erken durdurur
    )
    
    print("Eğitim Tamamlandı! En iyi model weights/best.pt olarak kaydedildi.")

if __name__ == "__main__":
    main()
