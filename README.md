# ISG KKD Algılama Sistemi

Gerçek zamanlı kask · yelek · maske · eldiven tespiti.

---

## Kurulum

```bash
pip install -r requirements.txt
```

---

## Model — çoklu PPE modeli

Bu kurulum tek modele güvenmez. `models/ppe_model.pt` kask/yelek/kişi için ana modeldir;
`vyra_yolo_ppe_best.pt` ve `tanish_yolov8n_ppe_6class.pt` yardımcı model olarak aynı karede
çalışıp eldiven/gözlük/kask/yelek adaylarını ekler. Çakışan kutular otomatik birleştirilir.

Yardımcı modelleri indirmek için:

```powershell
.\download_vyra_model.ps1
.\download_tanish_model.ps1
```

İlk komut `models/vyra_yolo_ppe_best.pt` dosyasını indirir. Model sınıfları arasında
`Gloves`, `NO-Gloves`, `Goggles`, `NO-Goggles`, `Hardhat`, `Mask`,
`Safety Vest` ve `Person` bulunur.

İkinci komut `models/tanish_yolov8n_ppe_6class.pt` dosyasını indirir. Bu model
`Gloves`, `Vest`, `goggles`, `helmet`, `mask`, `safety_shoe` sınıflarıyla gelir.

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

## Yeni dayanıklılık ayarları

- `CLASS_ALIASES`: farklı PPE modellerindeki sınıf adlarını ortak isimlere çevirir.
- `ENABLE_AUX_PPE_MODELS`: yardımcı modelleri ensemble mantığıyla ana modele ekler.
- `ENABLE_TRACKING`: kişilere kareler arasında sabit ID verir.
- `ENABLE_STATUS_SMOOTHING`: tek karelik kaçırmalardan doğan yanlış alarmları azaltır.
- `REQUIRED_EQUIPMENTS`: hangi ekipmanın gerçekten zorunlu olduğunu belirler.
- `MIN_EQUIPMENT_OVERLAP` / `MIN_GLOVE_OVERLAP`: ekipman-kişi eşleştirme hassasiyetini ayarlar.
- `ENABLE_VEST_COLOR_FALLBACK`: model yeleği kaçırırsa torso bölgesinde fosforlu yelek rengi arar.

---

## Test ve kendi veri havuzu

Web/test görselleri indir:

```powershell
.\download_test_images.ps1
```

`test_images/` klasöründeki görselleri benchmark et:

```powershell
.\run_app.bat
```

veya statik görseller için:

```powershell
.\run_benchmark.bat
```

Çıktılar `benchmark_output/annotated/` ve `benchmark_output/results.json` içine yazılır.

Otomatik pseudo-label indeksini yenile:

```powershell
.\run_auto_index.bat
```

Çıktılar `datasets/auto_ppe/` içine yazılır. Bu etiketler eğitimden önce gözle kontrol edilmelidir.

Canlı kamerada yanlış/şüpheli örnek gördüğünde `c` tuşuna bas. Sistem ham kareyi ve metadata'yı
`dataset_review/` içine kaydeder. Otomatik yelek renk fallback'i kullanılan kareler de hard-example
olarak kaydedilir. Bu görüntüler sonradan LabelImg/Roboflow/CVAT ile etiketlenip YOLO formatına
aktarılmalıdır.

Etiketli veri hazır olduğunda fine-tune:

```powershell
.\train_custom_ppe.ps1
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
    ├── person_tracker.py      ← kareler arası kişi ID takibi
    ├── status_smoother.py     ← kısa süreli yanlış alarm filtresi
    └── drawing.py             ← OpenCV çizim
```
