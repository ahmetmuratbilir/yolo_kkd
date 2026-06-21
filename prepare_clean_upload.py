import os
import shutil
from pathlib import Path

def main():
    base_dir = Path(__file__).parent
    upload_dir = base_dir / "gloves_upload_clean"
    upload_dir.mkdir(exist_ok=True)
    
    # Mapping: original -> clean name
    mapping = {
        "GLOVES.v1i.yolov8.zip": "gloves_v1i_yolov8.zip",
        "gloves.v1i.yolov8 (2).zip": "gloves_v1i_yolov8_2.zip",
        "Gloves Detection.v1i.yolov8.zip": "gloves_detection_v1i_yolov8.zip",
        "gloves detection.v4i.yolov8.zip": "gloves_detection_v4i_yolov8.zip",
        "gloves.v1i.yolov8 (1).zip": "gloves_v1i_yolov8_1.zip",
    }
    
    print("Copying and renaming ZIP files to gloves_upload_clean...")
    for orig, clean in mapping.items():
        src = base_dir / orig
        dst = upload_dir / clean
        if src.exists():
            print(f"  {orig} -> {clean}")
            shutil.copy2(src, dst)
        else:
            print(f"  [!] Warning: {orig} not found!")

    # Write dataset-metadata.json with a new slug to avoid conflict with the broken one
    metadata = {
        "title": "Gloves PPE ZIPs Cleaned",
        "id": "muratbilir/gloves-ppe-zips-cleaned",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    import json
    with open(upload_dir / "dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("\nDataset metadata written.")

if __name__ == "__main__":
    main()
