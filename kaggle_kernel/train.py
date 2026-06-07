import os
import zipfile
import subprocess

# Ensure ultralytics is installed
subprocess.run(["pip", "install", "-q", "ultralytics"])

from ultralytics import YOLO

# Create datasets directory
os.makedirs('/kaggle/working/datasets', exist_ok=True)

print("Extracting combined_ppe.zip...")
with zipfile.ZipFile('/kaggle/input/combined-ppe/combined_ppe.zip', 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/datasets')
print("Extraction complete.")

# Create a data.yaml on the fly for Kaggle
yaml_content = """
path: /kaggle/working/datasets/combined_ppe
train: train/images
val: valid/images
test: test/images

nc: 10
names:
  0: person
  1: helmet_pos
  2: helmet_neg
  3: vest_pos
  4: vest_neg
  5: gloves_pos
  6: gloves_neg
  7: goggles_pos
  8: goggles_neg
  9: smoking
"""

with open('/kaggle/working/data.yaml', 'w') as f:
    f.write(yaml_content)

print("Starting training with YOLO11m on Kaggle...")
# Initialize the model (using pretrained weights as recommended by Claude)
model = YOLO('yolo11m.pt')

# Train the model
model.train(
    data='/kaggle/working/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,          # Kaggle has 16GB VRAM, batch 16 is safe
    patience=20,       # Early stopping
    project='/kaggle/working/runs',
    name='train_100_epochs',
    save=True
)

print("Training finished!")
