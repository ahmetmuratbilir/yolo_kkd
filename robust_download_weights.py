import os
import sys
import time
import zipfile
import shutil
from pathlib import Path
import requests
from kaggle import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

def get_fresh_download_url():
    """Fetches a fresh signed download URL for _output_.zip from Kaggle API."""
    print("Fetching fresh download URL from Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    with api.build_kaggle_client() as kaggle:
        request = ApiListKernelSessionOutputRequest()
        request.user_name = "muratbilir"
        request.kernel_slug = "custom-ppe-v4-training"
        response = kaggle.kernels.kernels_api_client.list_kernel_session_output(request)
        
        for item in response.files or []:
            if item.file_name == "_output_.zip":
                return item.url
    return None

def download_file_with_resume(local_path):
    temp_path = Path(local_path).with_suffix(".tmp")
    
    retry_count = 0
    max_retries = 50
    total_size = 0
    
    # We will get a fresh URL at the start and when it expires
    url = get_fresh_download_url()
    if not url:
        print("[!] Error: Could not get download URL from Kaggle.")
        return False
        
    while retry_count < max_retries:
        try:
            # Check existing temp file size
            existing_size = temp_path.stat().st_size if temp_path.exists() else 0
            
            # Prepare headers
            headers = {}
            if existing_size > 0:
                print(f"\nResuming download from byte {existing_size}...")
                headers["Range"] = f"bytes={existing_size}-"
            else:
                print("\nStarting download from the beginning...")
                headers.pop("Range", None)
                
            # Get request (streamed)
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # If 403 Forbidden, the signed URL has likely expired. Fetch a new one.
            if r.status_code in [403, 401]:
                print(f"\n[!] Got status {r.status_code} (URL likely expired). Fetching fresh URL...")
                url = get_fresh_download_url()
                if not url:
                    print("[!] Error: Could not refresh download URL.")
                    return False
                time.sleep(2)
                continue
                
            if r.status_code not in [200, 206]:
                print(f"\n[!] Server returned error status {r.status_code}. Retrying in 5s...")
                time.sleep(5)
                retry_count += 1
                continue
                
            # Get total content size
            if r.status_code == 200:
                # Starting from scratch, content-length is the total size
                total_size = int(r.headers.get("content-length", 0))
            elif r.status_code == 206:
                # Resuming, content-length is the remaining size. Total size is remaining + existing
                remaining_size = int(r.headers.get("content-length", 0))
                total_size = remaining_size + existing_size
                
            print(f"Total target size: {total_size / (1024*1024):.2f} MB")
            
            # Check if range request was accepted
            is_range = (r.status_code == 206)
            mode = "ab" if (existing_size > 0 and is_range) else "wb"
            if not is_range and existing_size > 0:
                print("Warning: Server did not accept Range request, restarting download...")
                existing_size = 0
                
            # Download in chunks
            last_print_time = time.time()
            with open(temp_path, mode) as f:
                for chunk in r.iter_content(chunk_size=1024*1024): # 1 MB chunks
                    if chunk:
                        f.write(chunk)
                        existing_size += len(chunk)
                        
                        # Print progress at most once per second to reduce terminal lag
                        current_time = time.time()
                        if current_time - last_print_time > 1.0 or existing_size == total_size:
                            percent = (existing_size / total_size * 100) if total_size > 0 else 0
                            print(f"\rProgress: {percent:.2f}% ({existing_size/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)", end="", flush=True)
                            last_print_time = current_time
                            
            # Verify if complete
            if temp_path.stat().st_size == total_size and total_size > 0:
                os.replace(temp_path, local_path)
                print(f"\n[SUCCESS] Download completed!")
                return True
            else:
                print(f"\n[!] Connection closed early. Downloaded {temp_path.stat().st_size} of {total_size} bytes. Retrying...")
                retry_count += 1
                
        except (requests.exceptions.RequestException, Exception) as e:
            print(f"\n[!] Download error: {e}")
            print("Waiting 5 seconds before retrying...")
            time.sleep(5)
            retry_count += 1
            
    print("\n[!] Error: Max retries exceeded.")
    return False

def main():
    base_dir = Path(__file__).parent
    
    # Destination directory for the weights
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    local_zip_path = base_dir / "_output_.zip"
    
    # Download using our robust resumable downloader
    success = download_file_with_resume(local_zip_path)
    
    if not success:
        print("[!] Download failed.")
        return
        
    print("\n=== Extracting Model Weights ===")
    target_weight_in_zip = "runs/detect/custom_ppe_v4/weights/best.pt"
    target_results_in_zip = "runs/detect/custom_ppe_v4/results.png"
    
    extracted_weight_path = models_dir / "best.pt"
    extracted_results_path = base_dir / "results.png"
    
    try:
        with zipfile.ZipFile(local_zip_path, 'r') as z:
            all_files = z.namelist()
            print(f"Total files in output ZIP: {len(all_files)}")
            
            # Find best.pt path dynamically
            weight_path_in_zip = None
            for f in all_files:
                if f.endswith("best.pt"):
                    weight_path_in_zip = f
                    break
                    
            if weight_path_in_zip:
                print(f"Found weight file in zip: {weight_path_in_zip}")
                with z.open(weight_path_in_zip) as source, open(extracted_weight_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                print(f"[SUCCESS] Saved weight to: {extracted_weight_path}")
            else:
                print("[!] Error: best.pt not found inside the ZIP archive!")
                
            # Find results.png path dynamically
            results_path_in_zip = None
            for f in all_files:
                if f.endswith("results.png"):
                    results_path_in_zip = f
                    break
                    
            if results_path_in_zip:
                print(f"Found results file in zip: {results_path_in_zip}")
                with z.open(results_path_in_zip) as source, open(extracted_results_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                print(f"[SUCCESS] Saved results plot to: {extracted_results_path}")
                
            # Find results.csv path dynamically
            csv_path_in_zip = None
            for f in all_files:
                if f.endswith("results.csv"):
                    csv_path_in_zip = f
                    break
                    
            if csv_path_in_zip:
                print(f"Found csv file in zip: {csv_path_in_zip}")
                extracted_csv_path = base_dir / "results.csv"
                with z.open(csv_path_in_zip) as source, open(extracted_csv_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                print(f"[SUCCESS] Saved results CSV to: {extracted_csv_path}")
            else:
                print("[!] Error: results.csv not found inside the ZIP archive!")
                
            # Find last_checkpoint.pt path dynamically
            last_checkpoint_path_in_zip = None
            for f in all_files:
                if f.endswith("last_checkpoint.pt"):
                    last_checkpoint_path_in_zip = f
                    break
                    
            if last_checkpoint_path_in_zip:
                print(f"Found last checkpoint file in zip: {last_checkpoint_path_in_zip}")
                extracted_last_checkpoint_path = base_dir / "last_checkpoint.pt"
                with z.open(last_checkpoint_path_in_zip) as source, open(extracted_last_checkpoint_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                print(f"[SUCCESS] Saved last checkpoint to: {extracted_last_checkpoint_path}")
            else:
                print("[!] Warning: last_checkpoint.pt not found inside the ZIP archive!")
                
    except Exception as e:
        print(f"[!] Error during extraction: {e}")
        
    # Cleanup the big zip file
    print("\n=== Cleaning Up 7.6GB ZIP Archive ===")
    try:
        if local_zip_path.exists():
            os.remove(local_zip_path)
            print("[OK] Deleted _output_.zip successfully. Disk space cleared.")
    except Exception as e:
        print(f"Warning: Failed to delete zip archive: {e}")
        
    print("\nAll done!")

if __name__ == "__main__":
    main()
