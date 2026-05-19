# ISG KKD Algılama Sistemi

Gerçek zamanlı kask · yelek · maske · eldiven tespiti.

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Model — `models/ppe_model.pt`

Bu proje kask/yelek/maske/eldiven görmüş **hazır eğitilmiş** bir YOLO modeli ister.
Aşağıdakilerden birini `models/` klasörüne koy ve `config.py → MODEL_PATH` güncelle.

### Seçenek 1 – Roboflow'dan hazır PPE modeli (önerilen)

Roboflow Universe'de ücretsiz PPE modelleri var:
https://universe.roboflow.com/

Arama: `PPE detection` → "Download Model" → Format: **YOLOv8 PyTorch** → `.pt` dosyasını al.

Popüler hazır modeller:
| Model | Sınıflar | Link |
|---|---|---|
| keremberke/hard-hat-detection | helmet, person | roboflow.com |
| SomaDhan/ppe-detection-3aasr | helmet, vest, mask, glove | roboflow.com |
| Nerdio/PPE-Detection | helmet, vest, mask | roboflow.com |

Roboflow Python ile indirme:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="SENIN_API_KEY")
project = rf.workspace("...").project("ppe-detection-...")
model = project.version(1).download("yolov8")
```

### Seçenek 2 – GitHub hazır ağırlıklar

- https://github.com/niconielsen32/PPE-Detection
- https://github.com/ultralytics/assets (genel COCO; kask yok ama kişi tespiti için)

### Seçenek 3 – Kendi modelini eğit

Roboflow veya LabelImg ile kendi veri setini hazırla.
```bash
yolo train model=yolov8n.pt data=ppe.yaml epochs=50 imgsz=640
```

---

## Çalıştırma

```bash
python main.py
```

- `q` tuşu → çıkış
- IP kamera için `config.py` içinde `CAMERA_SOURCE = "rtsp://..."` yap

---

## Eldiven rengi ayarı

`config.py` içinde `GLOVE_COLOR_RANGES` listesine kendi eldiveninin HSV aralığını ekle.

HSV renk bulma aracı:
```python
import cv2, numpy as np
img = cv2.imread("eldiven_foto.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(hsv[h, w])  # tıkladığın pikselin değeri
```

---

## Proje yapısı

```
isg-ppe-detection/
├── main.py
├── config.py
├── requirements.txt
├── models/
│   └── ppe_model.pt          ← buraya koy
└── services/
    ├── detector.py            ← YOLO çıkarımı
    ├── glove_color_detector.py← HSV renk analizi
    ├── rule_engine.py         ← eşleştirme + uyarı
    └── drawing.py             ← OpenCV çizim
```
