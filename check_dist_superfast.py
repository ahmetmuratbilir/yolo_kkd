import os
from collections import Counter
import time

labels_dir = 'datasets/combined_ppe/train/labels'
counts = Counter()
start_time = time.time()

print("Dosyalar listeleniyor...")
try:
    files = [f.path for f in os.scandir(labels_dir) if f.name.endswith('.txt')]
    total = len(files)
    print(f"Toplam dosya: {total}. Icerikler okunuyor...")

    for i, path in enumerate(files):
        if (i+1) % 20000 == 0:
            print(f"Okunan: {i+1}/{total}")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.split(maxsplit=1)
                if parts:
                    counts[parts[0]] += 1
except Exception as e:
    print(f"Hata: {e}")

classes = {'0':'person', '1':'helmet_pos', '2':'helmet_neg', '3':'vest_pos', 
           '4':'vest_neg', '5':'gloves_pos', '6':'gloves_neg', '7':'goggles_pos', '8':'goggles_neg'}

print("\n--- SONUC ---")
for k, v in counts.items():
    if k in classes:
        print(f"{classes[k]}: {v:,}")

print(f"Sure: {time.time() - start_time:.2f}s")
