import cv2
import streamlit as st
import numpy as np
import tempfile
import os
import time
from ultralytics import YOLO
import image_dehazer
from sort.sort import Sort  # SORT tracking algorithm

# Load YOLO models
accident_model_path = r"D:\Accident_Detection_and_Alert_System_Using_Yolov11+GAN\runs\detect\train10\weights\best.pt"
anpr_model_path = r"C:\Users\iproat26\Desktop\Anpr\Vehicle-Registration-Plates-2\runs\detect\train2\weights\best.pt"

accident_model = YOLO(accident_model_path)
anpr_model = YOLO(anpr_model_path)

# Define class names
class_names = ['bike', 'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
               'car', 'car_bike_accident', 'car_car_accident', 'car_object_accident',
               'car_person_accident', 'person']

# Initialize SORT Tracker
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)

# Folder to save number plate images
save_folder = "Detected_Plates"
os.makedirs(save_folder, exist_ok=True)

st.title("Road Accident Detection with Vehicle Counting & Number Plate Recognition")

# Video uploader
uploaded_file = st.file_uploader("Upload a Video for Detection", type=["mp4", "avi", "mov"])

# Function to detect accidents and track vehicles
def detect_accidents(frame):
    if frame.mean() < 100:
        frame, _ = image_dehazer.remove_haze(frame)

    results = accident_model(frame)
    detected_classes = []
    detections = []
    accident_detected = False  # Flag to check for accidents

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.3:
                label = class_names[cls]
                detected_classes.append(label)
                color = (0, 0, 255) if 'accident' in label else (0, 255, 0)
                font_scale = 1 if 'accident' in label else 0.5

                if 'accident' in label:
                    accident_detected = True  # Mark accident detection

                # Save detection for tracking
                detections.append([x1, y1, x2, y2, conf])

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    # Update SORT tracker
    tracked_objects = tracker.update(np.array(detections))

    # Count Vehicles
    vehicle_count = {"car": 0, "bike": 0}
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, track)
        for label in detected_classes:
            if label in ["car", "bike"]:
                vehicle_count[label] += 1

    # Calculate severity percentage
    total_objects = len(detected_classes)
    accident_objects = sum(1 for x in detected_classes if "accident" in x)
    severity = (accident_objects / total_objects) * 100 if total_objects > 0 else 0

    return frame, detected_classes, vehicle_count, severity, accident_detected

# Function to detect and save number plates
def detect_and_save_plate(frame):
    results = anpr_model(frame)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > 0.3:  # Confidence threshold
                plate_crop = frame[y1:y2, x1:x2]

                # Generate filename with timestamp
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                # milliseconds = int(time.time() * 1000) % 1000  # Extract milliseconds
                file_path = os.path.join(save_folder, f"plate_{timestamp}.jpg")

                cv2.imwrite(file_path, plate_crop)
                print(f"Saved: {file_path}")
                return file_path  # Return saved file path

    return None

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    cap = cv2.VideoCapture(temp_video_path)
    FRAME_WINDOW = st.image([])
    detected_text = st.empty()
    count_text = st.empty()
    severity_text = st.empty()
    plate_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame, detected_classes, vehicle_count, severity, accident_detected = detect_accidents(frame)
        except:
            pass

        frame = cv2.resize(frame, (640, 360))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(frame)
        detected_text.markdown(f"<h2 style='color:red;'>Detected Classes: {', '.join(set(detected_classes))}</h2>", unsafe_allow_html=True)
        count_text.markdown(f"<h2 style='color:blue;'>Vehicle Count - Cars: {vehicle_count['car']}, Bikes: {vehicle_count['bike']}</h2>", unsafe_allow_html=True)
        severity_text.markdown(f"<h2 style='color:orange;'>Severity: {severity:.2f}%</h2>", unsafe_allow_html=True)

        if accident_detected:
            plate_path = detect_and_save_plate(frame)
            if plate_path:
                plate_text.markdown(f"<h2 style='color:green;'>Number Plate Saved: {plate_path}</h2>", unsafe_allow_html=True)

    cap.release()
    st.success("Processing Complete")
