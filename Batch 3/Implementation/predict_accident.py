import cv2
# import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"runs\detect\train10\weights\best.pt")  # Ensure you have the right model path

# Define class names
class_names = ['bike', 'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
               'car', 'car_bike_accident', 'car_car_accident', 'car_object_accident',
               'car_person_accident', 'person']

# Open a video capture (0 for webcam, or provide video path)
cap = cv2.VideoCapture(r'D:\deskstop 1\Road accident detection with rapid emergency notification using gan and yolov11\Accident.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index
            
            if conf > 0.3:  # Confidence threshold
                label = f"{class_names[cls]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLO Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
