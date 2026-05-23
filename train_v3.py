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
        print("GPU bulunamadi, CPU kullaniliyor...")
        device = 'cpu'
        batch = 8

    print("Model: best.pt'den devam (v3 - 62K resim)")
    print(f"Cihaz: {device} | Batch: {batch}")
    print("Max epoch: 100 | Erken durma: 15 epoch iyilesme olmazsa")
    print("-" * 50)

    model = YOLO("runs/detect/custom_ppe_gpu/weights/best.pt")

    results = model.train(
        data="datasets/combined_ppe/data.yaml",
        epochs=100,          # Model ne kadar gerekirse o kadar egitilir
        imgsz=640,
        batch=batch,
        device=device,
        amp=True,
        cache=True,
        workers=4,
        name="custom_ppe_v3",
        patience=15,         # 15 epoch iyilesme olmazsa otomatik durur
        optimizer="AdamW",
        lr0=0.0005,
        exist_ok=True,
    )

    print("\nEgitim Tamamlandi!")
    print("Model: runs/detect/custom_ppe_v3/weights/best.pt")

if __name__ == "__main__":
    main()
