import os
import time
import subprocess
from pathlib import Path

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()

def main():
    base_dir = Path(__file__).parent
    python_exe = base_dir / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        print("[!] python.exe not found at expected location.")
        return

    print("Checking dataset status on Kaggle...")
    dataset_ref = "muratbilir/gloves-ppe-zips-cleaned"
    
    # Wait until dataset is ready
    retries = 30
    for i in range(retries):
        code, stdout, stderr = run_cmd(f'"{python_exe}" -m kaggle datasets status {dataset_ref}')
        if code == 0:
            status = stdout.lower()
            print(f"  Dataset status check {i+1}/{retries}: {status}")
            if "ready" in status:
                print("[OK] Dataset is processed and READY on Kaggle!")
                break
        else:
            print(f"  [!] Dataset status check {i+1}/{retries} failed (it may still be uploading or creating): {stderr or stdout}")
        
        time.sleep(15)
    else:
        print("[!] Timeout waiting for dataset to become ready. We will attempt to push anyway...")

    # Push the kernel
    print("\nPushing training kernel to Kaggle...")
    kernel_dir = base_dir / "kaggle_kernel"
    code, stdout, stderr = run_cmd(f'"{python_exe}" -m kaggle kernels push -p "{kernel_dir}" --accelerator NvidiaTeslaT4')
    
    if code == 0:
        print("\n" + "="*60)
        print("[SUCCESS] Kernel pushed successfully to Kaggle!")
        print(stdout)
        print("You can monitor the training progress here:")
        print("  https://www.kaggle.com/muratbilir/custom-ppe-v4-training")
        print("="*60)
    else:
        print(f"\n[!] Failed to push kernel: {stderr or stdout}")

if __name__ == "__main__":
    main()
