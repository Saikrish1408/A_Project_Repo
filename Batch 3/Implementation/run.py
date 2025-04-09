import cv2
import streamlit as st
import numpy as np
import tempfile
from ultralytics import YOLO
import image_dehazer

# Load YOLO model
model_path = r"D:\Accident_Detection_and_Alert_System_Using_Yolov11+GAN\runs\detect\train10\weights\best.pt"
model = YOLO(model_path)

# Define class names
class_names = ['bike', 'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
               'car', 'car_bike_accident', 'car_car_accident', 'car_object_accident',
               'car_person_accident', 'person']

st.title("Road Accident Detection with Rapid Emergency Notification Using GAN and YOLOv11")

# Video uploader
uploaded_file = st.file_uploader("Upload a Video for Accident Detection", type=["mp4", "avi", "mov"])

def detect_accidents(frame):
    if frame.mean() < 100:
        frame, _ = image_dehazer.remove_haze(frame)
    
    results = model(frame)
    detected_classes = []
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    
    return frame, detected_classes

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name
    
    cap = cv2.VideoCapture(temp_video_path)
    FRAME_WINDOW = st.image([])
    detected_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, detected_classes = detect_accidents(frame)
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        
        detected_text.markdown(f"<h2 style='color:red;'>Detected Classes: {', '.join(set(detected_classes))}</h2>", unsafe_allow_html=True)
        
    cap.release()
    st.success("Processing Complete")