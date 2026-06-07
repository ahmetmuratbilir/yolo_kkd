import os
from collections import Counter

labels_dir = 'datasets/combined_ppe/train/labels'
counts = Counter()

for f in os.listdir(labels_dir):
    if f.endswith('.txt'):
        with open(os.path.join(labels_dir, f)) as file:
            for line in file:
                if line.strip():
                    try:
                        counts[int(line.split()[0])] += 1
                    except ValueError:
                        pass

classes = {0:'person', 1:'helmet_pos', 2:'helmet_neg', 3:'vest_pos', 
           4:'vest_neg', 5:'gloves_pos', 6:'gloves_neg', 7:'goggles_pos', 8:'goggles_neg'}

for k,v in sorted(counts.items()):
    print(f'{classes.get(k,k)}: {v:,}')
