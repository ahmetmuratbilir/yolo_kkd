# 🦺 PPE Detection with YOLOv11m

Nükleer reaktör sahalarında **Kişisel Koruyucu Donanım (KKD)** tespiti için eğitilmiş YOLOv11m modeli.

## 📊 Model Performansı

| Metrik | Sonuç |
|--------|-------|
| **mAP50** | %68.1 |
| **mAP50-95** | %40.9 |
| **Precision** | %72.1 |
| **Recall** | %64.0 |

### Sınıf Bazında Başarı (mAP50)

| Sınıf | Başarı |
|-------|--------|
| 🧍 Person | %84.0 |
| ⛑️ Baret (Takıyor) | %80.3 |
| ❌ Baret (Takmıyor) | %79.3 |
| 🦺 Yelek (Takıyor) | %74.9 |
| ❌ Yelek (Takmıyor) | %80.6 |
| 🥽 Gözlük (Takıyor) | %84.3 |
| 🤚 Eldiven (Takıyor) | %53.1 |
| 🚬 Sigara | %74.3 |

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

## 🤖 Eğitim Detayları

| Parametre | Değer |
|-----------|-------|
| Model | YOLOv11m |
| Epochs | 100 |
| Image Size | 640x640 |
| Batch Size | 16 |
| Platform | Kaggle (GPU T4 x2) |
| Eğitim Süresi | ~12 saat |
| Veri Seti | Combined PPE Dataset V2 (~100K görsel) |

## 📥 Model İndirme

Eğitilmiş model ağırlıklarını Kaggle'dan indirebilirsiniz:

👉 **[PPE YOLOv8m Best Model Weights - Kaggle](https://www.kaggle.com/datasets/muratbilir/ppe-yolov8m-best-weights)**

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
