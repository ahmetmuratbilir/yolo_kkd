import os
from collections import Counter
import time

labels_dir = 'datasets/combined_ppe/train/labels'
counts = Counter()

start_time = time.time()
file_count = 0

print("Sayım başlatıldı, lütfen bekleyin...")

# os.scandir is much faster than os.listdir on Windows for huge directories
try:
    with os.scandir(labels_dir) as entries:
        for entry in entries:
            if entry.name.endswith('.txt') and entry.is_file():
                file_count += 1
                if file_count % 20000 == 0:
                    print(f"İşlenen dosya: {file_count}...")
                    
                with open(entry.path, 'r', encoding='utf-8', errors='ignore') as file:
                    for line in file:
                        if line.strip():
                            try:
                                counts[int(line.split()[0])] += 1
                            except ValueError:
                                pass
except Exception as e:
    print(f"Hata oluştu: {e}")

classes = {0:'person', 1:'helmet_pos', 2:'helmet_neg', 3:'vest_pos', 
           4:'vest_neg', 5:'gloves_pos', 6:'gloves_neg', 7:'goggles_pos', 8:'goggles_neg'}

print("\n--- SONUÇ ---")
for k,v in sorted(counts.items()):
    print(f'{classes.get(k,k)}: {v:,}')

print(f"\nToplam okunan dosya: {file_count:,}")
print(f"Geçen süre: {time.time() - start_time:.2f} saniye")
