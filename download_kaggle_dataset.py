import kagglehub

def main():
    print("Kaggle'dan PPE (İSG) veri seti indiriliyor...")
    # snehilsanyal/construction-site-safety-image-dataset-roboflow
    # Bu veri seti Roboflow formatında YOLO etiketleri içerir.
    path = kagglehub.dataset_download("snehilsanyal/construction-site-safety-image-dataset-roboflow")
    print(f"\nİndirme Tamamlandı! Veri Seti Yolu:\n{path}")

if __name__ == "__main__":
    main()
