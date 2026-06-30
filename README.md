# 🦺 Nükleer Reaktör İSG KKD Tespit Sistemi (YOLOv11m - Sürüm V4)

Bu repo, nükleer reaktör sahalarında iş güvenliği kurallarının denetlenmesi amacıyla geliştirilmiş **Kişisel Koruyucu Donanım (KKD)** tespit sistemidir. Sistem; kask (baret), yelek, gözlük, eldiven ve sigara içme ihlallerini gerçek zamanlı olarak izler ve raporlar.

---

## 📊 Model Performansı (Seans 3 - Güncel)

Modelimiz **Combined PPE + Gloves + Goggles (Safety Glasses V5)** veri setlerinin birleştirilmesiyle **YOLOv11m** mimarisi üzerinde sıfırdan eğitilmektedir. Toplamda **31 epoch** tamamlanmıştır.

### Seans 2 vs Seans 3 Karşılaştırma Raporu
Yerel GPU (RTX 4050) üzerinde **11.187 doğrulama görseli** ile yapılan test sonuçları:

| Sınıf (Class) | Seans 2 mAP50 (%) | Seans 3 mAP50 (%) | Net Değişim | Durum |
| :--- | :---: | :---: | :---: | :--- |
| **person** (Kişi) | 80.00% | **81.86%** | `+1.86%` | 📈 Yükseldi |
| **helmet_pos** (Baretli) | 77.00% | **80.54%** | `+3.54%` | 📈 Yükseldi |
| **helmet_neg** (Baretsiz) | **82.00%** | 79.67% | `-2.33%` | 📉 Hafif Düştü |
| **vest_pos** (Yelekli) | 68.00% | **73.65%** | `+5.65%` | 📈 Yükseldi |
| **vest_neg** (Yeleksiz) | **86.00%** | 80.11% | `-5.89%` | 📉 Düştü |
| **gloves_pos** (Eldivenli) | 52.00% | **56.52%** | `+4.52%` | 📈 Yükseldi |
| **gloves_neg** (Eldivensiz) | **55.00%** | 53.32% | `-1.68%` | 📉 Stabil |
| **goggles_pos** (Gözlüklü) | **86.00%** | 85.31% | `-0.69%` | 📉 Stabil |
| **goggles_neg** (Gözlüksüz) | **0.00%** | **23.60%** | `+23.60%` | 🚀 **Büyük İlerleme** |
| **smoking** (Sigara) | **80.00%** | 77.96% | `-2.04%` | 📉 Stabil |
| **GENEL ORTALAMA (ALL)** | 67.49% | **69.32%** | `+1.83%` | 📈 **Sürekli İyileşme** |

> [!NOTE]
> **Overfitting Kontrolü:** Train Loss: **1.228** vs Val Loss: **1.162**. Doğrulama kaybı hala eğitim kaybından düşüktür; bu da modelde **kesinlikle ezberleme (overfitting) olmadığını** ve öğrenme kapasitesinin açık olduğunu gösterir.

---

## 🚀 Yerel Kurulum ve Kullanım

Sistem yerel GPU'nuz (RTX 4050) üzerinde çalışacak şekilde yapılandırılmıştır.

### 1. Gereksinimlerin Yüklenmesi
```bash
# Sanal ortamı aktifleştirin
.\.venv\Scripts\activate

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

### 2. Canlı Webcam Uygulamasını Çalıştırma
Sistemi canlı kamera kaynağı üzerinden gerçek zamanlı test etmek için:
```bash
python main.py
```
* Kamera ayarlarını değiştirmek (örneğin RTSP adresi tanımlamak) için [config.py](file:///c:/Users/ahmet%20murat%20bilir/Desktop/nukleerraktoruygulaması/okuldakiler/yolo_egitim/config.py) dosyasındaki `CAMERA_SOURCE = 0` değerini güncelleyin.

### 3. Kayıtlı Video Üzerinde Test
Belirli bir test videosunu koşturmak ve ihlalleri kaydetmek için:
```bash
python detect_video.py --source test_videos/ornek.mp4
```

---

## 🛠️ Yardımcı Araçlar (Scratch Utilities)

Bağlantı kopmalarını yönetmek ve doğrulama yapmak için repo içerisine iki adet kritik araç eklenmiştir:

### 1. Kopma Korumalı Akıllı İndirici
Kaggle sunucularından devasa indirme paketlerini yerel internet kopmalarına yakalanmadan çekmek için geliştirilmiştir. Dosyaları 2 MB'lık alt parçalar halinde indirir ve 6 kez otomatik yeniden dener.
```bash
python scratch/download_selective_remote_zip.py
```
* Bu script Kaggle API'den en son çıktıyı alır, içinden yalnızca `best.pt`, `last.pt` ve `results.csv` dosyalarını seçerek indirir ve otomatik olarak `models/` dizinine yerleştirir.

### 2. Sınıf Bazlı Performans Ölçer (GPU Validation)
Yerel doğrulama veri setini (11.187 resim) kullanarak modelinizin sınıf bazlı başarı yüzdelerini hesaplar ve terminale tablo halinde basar:
```bash
python scratch/validate_classes.py
```

---

## 🔄 Kaggle Limitleri Yenilendiğinde Seans 4 (Resume) Nasıl Başlatılır?

Haftalık 30 saatlik Kaggle GPU limitiniz dolduğunda yenilenmesini (83 saat) bekleyin. Limit yenilendiğinde eğitimi kaldığı yerden 12 saat daha sürdürmek için:

### Adım 1: Kaggle Scriptindeki Resume Ayarını Kontrol Edin
[kaggle_kernel/train_v4.py](file:///c:/Users/ahmet%20murat%20bilir/Desktop/nukleerraktoruygulaması/okuldakiler/yolo_egitim/kaggle_kernel/train_v4.py) dosyasında resume satırını şu şekilde ayarlayın:
```python
# Eğer en son checkpoint'ten devam edilecekse:
model = YOLO('/kaggle/input/custom-ppe-v4-training/last.pt')
# veya doğrudan resume parametresiyle:
model.train(..., resume=True)
```

### Adım 2: Kaggle Kernel'ı Push Edin
Aşağıdaki komutla yeni seansı (Seans 4) Kaggle üzerinde tetikleyin:
```bash
python run_kaggle_train.py
```
Bu komut, kodları Kaggle API aracılığıyla pushlar ve eğitimi uzaktan başlatır. Eğitim sürerken `results.csv` üzerinden model başarısının tırmanışını izleyebilirsiniz.
