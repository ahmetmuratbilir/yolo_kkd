import os
import io
import sys
import time
import zipfile
import requests
import shutil
from pathlib import Path
from kaggle import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

class RemoteZipFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        self.buffer = b""
        self.buffer_start = 0
        print("Initializing remote zip stream...")
        
        # Get total size using Range request for 1 byte (HEAD might be blocked or return 0 for signed URLs)
        r = requests.get(self.url, headers={'Range': 'bytes=0-0'}, allow_redirects=True, timeout=30)
        if r.status_code not in [200, 206]:
            raise Exception(f"Failed to connect to remote zip: HTTP {r.status_code}")
            
        content_range = r.headers.get('Content-Range', '')
        if '/' in content_range:
            self.size = int(content_range.split('/')[-1])
        else:
            self.size = int(r.headers.get('content-length', 0))
            
        print(f"Remote file size: {self.size / (1024*1024):.2f} MB")
            
    def readable(self):
        return True
        
    def seekable(self):
        return True
        
    def seek(self, offset, whence=io.SEEK_SET):
        prev = self.position
        if whence == io.SEEK_SET:
            self.position = offset
        elif whence == io.SEEK_CUR:
            self.position += offset
        elif whence == io.SEEK_END:
            self.position = self.size + offset
        print(f"Seek: {prev} -> {self.position} (offset={offset}, whence={whence})")
        return self.position
        
    def tell(self):
        return self.position
        
    def _fetch_remote(self, start, length):
        if start + length > self.size:
            length = self.size - start
        if length <= 0:
            return b""
        
        print(f"Fetching remote chunk: bytes={start}-{start + length - 1} ({length/1e6:.2f} MB)...")
        # Download in smaller sub-chunks to avoid timeouts
        sub_chunk_size = 2 * 1024 * 1024 # 2 MB
        data_buffer = io.BytesIO()
        downloaded = 0
        
        while downloaded < length:
            chunk_to_get = min(sub_chunk_size, length - downloaded)
            chunk_start = start + downloaded
            chunk_end = chunk_start + chunk_to_get - 1
            
            # Retry up to 6 times for each sub-chunk
            for attempt in range(6):
                try:
                    headers = {'Range': f'bytes={chunk_start}-{chunk_end}'}
                    r = requests.get(self.url, headers=headers, timeout=25)
                    if r.status_code in [200, 206]:
                        data_buffer.write(r.content)
                        downloaded += len(r.content)
                        if length > 2 * 1024 * 1024: # Only print for chunks larger than 2MB
                            print(f"  -> Downloaded {downloaded/1e6:.2f}/{length/1e6:.2f} MB...", flush=True)
                        break
                    else:
                        raise Exception(f"HTTP Status {r.status_code}")
                except Exception as e:
                    print(f"  [!] Attempt {attempt+1} failed to fetch sub-chunk {chunk_start}-{chunk_end}: {e}")
                    if attempt == 5:
                        raise e
                    time.sleep(5)
        return data_buffer.getvalue()
        
    def read(self, size=-1):
        if size == -1 or size is None:
            size = self.size - self.position
        if self.position >= self.size:
            return b""
        if self.position + size > self.size:
            size = self.size - self.position
            
        # Check if the requested range is within the buffer
        if self.buffer_start <= self.position < self.buffer_start + len(self.buffer):
            offset = self.position - self.buffer_start
            available = len(self.buffer) - offset
            if available >= size:
                # Satisfied from buffer
                data = self.buffer[offset:offset+size]
                self.position += len(data)
                return data
            else:
                # Part of it is in buffer, yield it and fetch remainder with read-ahead
                part1 = self.buffer[offset:]
                self.position += len(part1)
                
                # Fetch next chunk with read-ahead (e.g. 8 MB)
                read_size = max(size - len(part1), 8 * 1024 * 1024)
                part2 = self._fetch_remote(self.position, read_size)
                self.buffer = part2
                self.buffer_start = self.position
                
                requested_part2 = part2[:size - len(part1)]
                self.position += len(requested_part2)
                return part1 + requested_part2
        else:
            # Not in buffer, fetch new chunk with read-ahead
            read_size = max(size, 8 * 1024 * 1024)
            data = self._fetch_remote(self.position, read_size)
            self.buffer = data
            self.buffer_start = self.position
            
            requested_data = data[:size]
            self.position += len(requested_data)
            return requested_data

def get_fresh_download_url():
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

def main():
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    url = get_fresh_download_url()
    if not url:
        print("[!] Error: Could not get download URL from Kaggle.")
        return
        
    try:
        # Wrap remote zip
        remote_file = RemoteZipFile(url)
        
        print("\nOpening remote zip archive...")
        with zipfile.ZipFile(remote_file) as z:
            all_files = z.namelist()
            print(f"Total files in remote ZIP: {len(all_files)}")
            
            # 1. Extract best.pt
            weight_path = None
            for f in all_files:
                if f.endswith("best.pt"):
                    weight_path = f
                    break
                    
            if weight_path:
                print(f"Extracting {weight_path} from remote zip...")
                out_path = models_dir / "best.pt"
                with z.open(weight_path) as source, open(out_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                print(f"[SUCCESS] Saved weight to: {out_path}")
            else:
                print("[!] Error: best.pt not found inside the remote ZIP archive!")
                
            # 2. Extract results.csv
            csv_path = None
            for f in all_files:
                if f.endswith("results.csv"):
                    csv_path = f
                    break
                    
            if csv_path:
                print(f"Extracting {csv_path} from remote zip...")
                out_path = models_dir / "results.csv"
                with z.open(csv_path) as source, open(out_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                shutil.copy2(out_path, base_dir / "results.csv")
                print(f"[SUCCESS] Saved results CSV to: {out_path} and workspace root")
            else:
                print("[!] Error: results.csv not found inside the remote ZIP archive!")
                
            # 3. Extract results.png
            png_path = None
            for f in all_files:
                if f.endswith("results.png"):
                    png_path = f
                    break
                    
            if png_path:
                print(f"Extracting {png_path} from remote zip...")
                out_path = models_dir / "results.png"
                with z.open(png_path) as source, open(out_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                shutil.copy2(out_path, base_dir / "results.png")
                print(f"[SUCCESS] Saved results PNG to: {out_path} and workspace root")
            else:
                print("[!] Error: results.png not found inside the remote ZIP archive!")

            # 4. Extract last_checkpoint.pt
            last_checkpoint_path = None
            for f in all_files:
                if f.endswith("last_checkpoint.pt"):
                    last_checkpoint_path = f
                    break
                    
            if last_checkpoint_path:
                print(f"Extracting {last_checkpoint_path} from remote zip...")
                out_path = models_dir / "last.pt"
                with z.open(last_checkpoint_path) as source, open(out_path, "wb") as target:
                    shutil.copyfileobj(source, target)
                shutil.copy2(out_path, base_dir / "last_checkpoint.pt")
                print(f"[SUCCESS] Saved last checkpoint to: {out_path} and workspace root")
            else:
                print("[!] Warning: last_checkpoint.pt not found inside the remote ZIP archive!")
                
    except Exception as e:
        print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()
