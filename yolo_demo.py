from ultralytics import YOLO
import cv2

def run_yolo_demo():
    # Load a pretrained YOLOv8 model (yolov8n.pt is the nano version, small and fast)
    # It will automatically download if not present
    model = YOLO("yolo12n.pt")

    # Define the class IDs we want to detect from the COCO dataset
    # COCO class mapping (subset examples):
    # 0: person
    # 2: car
    # 5: bus
    # 39: bottle
    # 67: cell phone
    # You can find the full list here: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
    
    # Let's detect persons (0) and cars (2) for this demo


    # Open the webcam (0 is usually the default camera)
    # If you want to run on a video file, replace 0 with the path to the video file e.g., "video.mp4"
    print("Webcam başlatılıyor...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Hata: Webcam açılamadı. Lütfen kameranın bağlı ve başka bir uygulama tarafından kullanılmadığından emin olun.")
        return

    print("Webcam başarıyla açıldı. 'q' tuşuna basarak çıkabilirsiniz.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference on the frame
        # classes=target_classes filters the detections to only those specific IDs
        results = model(frame, verbose=False,conf=0.6)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Sonuçları (koordinatlar, sınıf, güven skoru) işlemek için örnek döngü:
        for result in results:
            for box in result.boxes:
                # Koordinatları al
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Sınıf ID'si ve ismini al
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                # Güven skorunu al
                conf = float(box.conf[0])
                
                print(f"Tespit: {class_name} | Güven: {conf:.2f} | Kutu: [{x1}, {y1}, {x2}, {y2}]")

        # Display the resulting frame
        cv2.imshow("YOLOv12 Canlı Kamera Tespiti", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yolo_demo()
