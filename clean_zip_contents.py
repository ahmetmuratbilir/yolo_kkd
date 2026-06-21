import os
import zipfile
from pathlib import Path

def clean_zip(src_path, dst_path):
    print(f"Cleaning {src_path.name} -> {dst_path.name}...")
    
    files_removed = []
    
    with zipfile.ZipFile(src_path, 'r') as zin:
        with zipfile.ZipFile(dst_path, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                name = item.filename
                # Remove root level data.yaml and text files
                if name in ["data.yaml", "README.dataset.txt", "README.roboflow.txt"] or ("/" not in name and name.endswith(".txt")) or name == "data.yaml":
                    files_removed.append(name)
                    continue
                # Copy other files
                zout.writestr(item, zin.read(item.filename))
                
    print(f"  Removed from {src_path.name}: {files_removed}")

def main():
    base_dir = Path(__file__).parent
    src_dir = base_dir / "gloves_upload_clean"
    dst_dir = base_dir / "gloves_upload_final"
    dst_dir.mkdir(exist_ok=True)
    
    for zip_file in src_dir.glob("*.zip"):
        dst_file = dst_dir / zip_file.name
        try:
            clean_zip(zip_file, dst_file)
        except Exception as e:
            print(f"  [!] Error processing {zip_file.name}: {e}")
            
    # Copy dataset-metadata.json
    src_meta = src_dir / "dataset-metadata.json"
    dst_meta = dst_dir / "dataset-metadata.json"
    if src_meta.exists():
        import shutil
        shutil.copy2(src_meta, dst_meta)
        
    print("\nAll ZIP files have been cleaned of conflicting files and saved to gloves_upload_final!")
    print("You can now drag and drop the files from gloves_upload_final to Kaggle.")

if __name__ == "__main__":
    main()
