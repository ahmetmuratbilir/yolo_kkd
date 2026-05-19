from ultralytics import YOLO
import cv2
import os

def run_detection():
    # Load the model
    model = YOLO("yolo12n.pt")
    
    # Path to the generated image
    img_path = r"C:\Users\ahmet murat bilir\.gemini\antigravity\brain\870c5146-711c-48fd-b051-fa890231ca03\nuclear_reactor_control_room_1777069392794.png"
    
    # Run inference
    results = model(img_path)
    
    # Save the result
    res_img = results[0].plot()
    output_path = "detection_result.jpg"
    cv2.imwrite(output_path, res_img)
    
    print(f"Detection completed. Result saved to {os.path.abspath(output_path)}")
    
    # Print what was detected
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Detected: {model.names[cls_id]} with confidence {conf:.2f}")

    # Display the result image in a window
    print("Sonuç penceresi açılıyor. Kapatmak için klavyeden herhangi bir tuşa basın...")
    cv2.imshow("YOLOv12 Nesne Tespiti", res_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()
