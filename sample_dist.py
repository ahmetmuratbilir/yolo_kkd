import os
import random
from collections import Counter

labels_dir = 'datasets/combined_ppe/train/labels'
counts = Counter()

# Sadece dosya isimlerini al (cok hizli)
try:
    files = os.listdir(labels_dir)
    txt_files = [f for f in files if f.endswith('.txt')]
    
    total_files = len(txt_files)
    sample_size = min(2000, total_files)
    
    # Rastgele 2000 dosya sec
    sampled_files = random.sample(txt_files, sample_size)
    
    print(f"Toplam {total_files} dosya var.")
    print(f"{sample_size} rastgele dosya uzerinden istatistiksel oranti hesaplaniyor...\n")
    
    for f in sampled_files:
        with open(os.path.join(labels_dir, f), 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                if line.strip():
                    try:
                        counts[int(line.split()[0])] += 1
                    except ValueError:
                        pass
                        
    classes = {0:'person', 1:'helmet_pos', 2:'helmet_neg', 3:'vest_pos', 
               4:'vest_neg', 5:'gloves_pos', 6:'gloves_neg', 7:'goggles_pos', 8:'goggles_neg'}

    print("--- TAHMINI SINIF DAGILIMI (Orantiya Gore) ---")
    
    total_labels_in_sample = sum(counts.values())
    if total_labels_in_sample == 0:
        print("Orneklemde etiket bulunamadi.")
    else:
        for k,v in sorted(counts.items()):
            percentage = (v / total_labels_in_sample) * 100
            estimated_total = int((v / sample_size) * total_files)
            print(f"{classes.get(k,k):<12} : ~{estimated_total:<6} (%{percentage:.1f})")
            
except Exception as e:
    print(f"Hata: {e}")
