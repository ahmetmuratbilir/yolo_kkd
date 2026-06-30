import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    base_dir = Path(__file__).parent.parent
    model_path = base_dir / "models" / "best.pt"
    data_yaml = base_dir / "data.yaml"
    
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    print("Running validation on validation set...")
    # Run validation (device=0 uses GPU for maximum speed, batch=32)
    results = model.val(
        data=str(data_yaml),
        split="val",
        batch=32,
        device=0,
        verbose=False
    )
    
    # Extract class metrics
    names = model.names
    ap50 = results.box.ap50
    ap = results.box.ap # AP50-95
    p = results.box.p # Precision
    r = results.box.r # Recall
    
    print("\n--- CLASS-BY-CLASS METRICS ---")
    print("| Class Name | AP50 (%) | AP50-95 (%) | Precision (%) | Recall (%) |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for i, name in names.items():
        print(f"| {name} | {ap50[i]*100:.2f}% | {ap[i]*100:.2f}% | {p[i]*100:.2f}% | {r[i]*100:.2f}% |")
    print("--------------------------------")

if __name__ == "__main__":
    main()
