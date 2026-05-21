from ultralytics import YOLO
import torch

def main():
    # GPU kontrolü
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU bulundu: {gpu_name} ({vram:.1f} GB VRAM)")
        device = 'cuda'
        # RTX 4050 Laptop = 6GB VRAM → batch=16 güvenli
        batch = 16
    else:
        print("⚠️  GPU bulunamadı, CPU ile devam ediliyor...")
        device = 'cpu'
        batch = 8

    print(f"📂 Veri Seti: datasets/combined_ppe/data.yaml")
    print(f"🤖 Model: YOLOv8 Small (yolov8s.pt)")
    print(f"⚙️  Cihaz: {device} | Batch: {batch}")
    print("-" * 50)

    model = YOLO("yolov8s.pt")

    results = model.train(
        data="datasets/combined_ppe/data.yaml",
        epochs=30,
        imgsz=640,
        batch=batch,
        device=device,          # ← GPU kullan
        amp=True,               # ← Otomatik Mixed Precision (hız +%20-30)
        cache=True,             # ← Resimleri RAM'e önbellekle (hız +%15-20)
        workers=4,              # ← Veri yükleme thread sayısı
        name="custom_ppe_gpu",  # ← Kayıt klasörü
        patience=10,
        optimizer="AdamW",      # ← Adam daha hızlı yakınsar
        lr0=0.001,
        exist_ok=True,
    )

    print("\n✅ Eğitim Tamamlandı!")
    print(f"📁 Model kaydedildi: runs/detect/custom_ppe_gpu/weights/best.pt")

if __name__ == "__main__":
    main()
