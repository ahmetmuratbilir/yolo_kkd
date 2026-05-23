from ultralytics import YOLO
import torch

def main():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        device = 'cuda'
        batch = 16
    else:
        print("GPU bulunamadi, CPU ile devam ediliyor...")
        device = 'cpu'
        batch = 8

    print("Model: best.pt (devam egitimi)")
    print(f"Cihaz: {device} | Batch: {batch}")
    print("-" * 50)

    # Onceki best.pt'den devam et
    model = YOLO("runs/detect/custom_ppe_gpu/weights/best.pt")

    results = model.train(
        data="datasets/combined_ppe/data.yaml",
        epochs=15,           # Devam egitimi icin 15 epoch yeterli
        imgsz=640,
        batch=batch,
        device=device,
        amp=True,
        cache=True,
        workers=4,
        name="custom_ppe_v2",  # Yeni klasore kaydet
        patience=7,
        optimizer="AdamW",
        lr0=0.0005,          # Daha dusuk LR (devam egitimi icin)
        exist_ok=True,
    )

    print("\nEgitim Tamamlandi!")
    print("Model: runs/detect/custom_ppe_v2/weights/best.pt")

if __name__ == "__main__":
    main()
