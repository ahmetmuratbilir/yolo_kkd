import zipfile
import os
import yaml
import glob

raw_dir = "datasets/raw_downloads"
zips = glob.glob(os.path.join(raw_dir, "*.zip"))

print(f"Bulunan ZIP sayisi: {len(zips)}")

for z in zips:
    print(f"\n--- {os.path.basename(z)} ---")
    try:
        with zipfile.ZipFile(z, 'r') as archive:
            yaml_files = [f for f in archive.namelist() if f.endswith('data.yaml') or f.endswith('dataset.yaml')]
            if not yaml_files:
                print("Yaml bulunamadi.")
                continue
            
            # Sadece ilk yaml'i oku
            with archive.open(yaml_files[0]) as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    print(f"Siniflar ({len(data['names'])} adet):")
                    if isinstance(data['names'], list):
                        for i, name in enumerate(data['names']):
                            print(f"  {i}: {name}")
                    elif isinstance(data['names'], dict):
                        for k, v in data['names'].items():
                            print(f"  {k}: {v}")
                else:
                    print("Yaml icinde 'names' anahtari yok.")
    except Exception as e:
        print(f"Hata: {e}")
