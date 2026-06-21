import os
import shutil
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    upload_dir = base_dir / "gloves_upload"
    upload_dir.mkdir(exist_ok=True)
    
    zip_files = [
        "GLOVES.v1i.yolov8.zip",
        "gloves.v1i.yolov8 (2).zip",
        "Gloves Detection.v1i.yolov8.zip",
        "gloves detection.v4i.yolov8.zip",
        "gloves.v1i.yolov8 (1).zip",
    ]
    
    print("Copying ZIP files to gloves_upload directory...")
    for zip_file in zip_files:
        src = base_dir / zip_file
        dst = upload_dir / zip_file
        if src.exists():
            print(f"  Copying {zip_file}...")
            shutil.copy2(src, dst)
        else:
            print(f"  [!] Warning: {zip_file} not found in workspace!")
            
    print("\nPreparation complete!")
    print("To upload the gloves dataset to Kaggle, run the following command in PowerShell:")
    print("  .venv\\Scripts\\kaggle.exe datasets create -p gloves_upload")

if __name__ == "__main__":
    main()
