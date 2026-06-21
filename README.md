# 🦺 PPE Detection with YOLOv11m

Nükleer reaktör sahalarında **Kişisel Koruyucu Donanım (KKD)** tespiti için eğitilmiş YOLOv11m modeli.

## 📊 Model Performansı (V4)

| Metrik | Sonuç |
|--------|-------|
| **mAP50** | **%94.2** |
| **mAP50-95** | **%72.8** |
| **Model** | **YOLOv11m** (Sürüm 4) |

Eski v3 modelindeki zincirleme fine-tune ezberleme sorunu (overfitting), bu sürümde sıfır yolo11m tabanından başlanarak ve veri setleri birleştirilerek (ortak eğitim yöntemi ile) tamamen çözülmüştür. Sınıflar genelinde dengeli ve yüksek bir genelleme başarısı elde edilmiştir.

## 🏗️ Proje Hakkında

Bu proje, nükleer reaktör sahalarında çalışan personelin KKD uyumluluğunu otomatik olarak denetlemek amacıyla geliştirilmiştir.

### Tespit Edilen Sınıflar (10 sınıf)
- `person` - Kişi
- `helmet_pos` - Baret takıyor ✅
- `helmet_neg` - Baret takmıyor ❌
- `vest_pos` - Yelek takıyor ✅
- `vest_neg` - Yelek takmıyor ❌
- `gloves_pos` - Eldiven takıyor ✅
- `gloves_neg` - Eldiven takmıyor ❌
- `goggles_pos` - Gözlük takıyor ✅
- `goggles_neg` - Gözlük takmıyor ❌
- `smoking` - Sigara içiyor 🚬

## 🤖 Eğitim Detayları (V4)

| Parametre | Değer |
|-----------|-------|
| Model | YOLOv11m |
| Epochs | 150 (Patience: 25) |
| Image Size | 640x640 |
| Batch Size | 16 |
| Platform | Kaggle (GPU T4 x2) |
| Veri Seti | Combined PPE Dataset V2 + Gloves Dataset (Ortak Eğitim) |

## 📥 Model İndirme

Eğitilmiş YOLOv11m model ağırlıklarını (V4) GitHub Releases üzerinden doğrudan indirebilirsiniz:

👉 **[YOLOv11m Best Model Weights (v4.0.0)](https://github.com/ahmetmuratbilir/yolo_kkd/releases/download/v4.0.0/best.pt)**

## 🚀 Kullanım

```python
from ultralytics import YOLO

# Modeli yükle
model = YOLO('best.pt')

# Görüntü üzerinde tahmin yap
results = model('foto.jpg')
results[0].show()

# Kamera ile canlı test
results = model(source=0, show=True)  # 0 = webcam
```

## 📁 Proje Yapısı

```
yolo_egitim/
├── kaggle_kernel/
│   └── kernel.ipynb        # Kaggle eğitim notebook'u
├── datasets/
│   └── combined_ppe/       # Veri seti (train/valid/test)
├── data.yaml               # Veri seti konfigürasyonu
├── best.pt                 # En iyi model ağırlıkları (Kaggle'dan indir)
└── README.md
```

## 📄 Lisans

Bu proje MIT lisansı ile lisanslanmıştır.
