import cv2
import streamlit as st
import numpy as np
import tempfile
import os
import time
import smtplib
from email.message import EmailMessage
from ultralytics import YOLO
import image_dehazer
from sort import Sort  # SORT tracking algorithm
import pandas as pd

# Load YOLO models
accident_model_path = "best_accident.pt"
anpr_model_path = "best_anpr.pt"

accident_model = YOLO(accident_model_path)
anpr_model = YOLO(anpr_model_path)

# Define class names
class_names = ['bike', 'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
               'car', 'car_bike_accident', 'car_car_accident', 'car_object_accident',
               'car_person_accident', 'person']

# Initialize SORT Tracker
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)

# Folders for saving images
plate_folder = "Detected_Plates"
accident_folder = "Accident_Images"
os.makedirs(plate_folder, exist_ok=True)
os.makedirs(accident_folder, exist_ok=True)

# Email Credentials
EMAIL_SENDER = "amsasaravanan101214@gmail.com"
EMAIL_PASSWORD = "cxbr mmab lqoy ayan"  # App password
EMAIL_RECEIVER = "amsasaravanan101214@gmail.com"

# Excel storage setup
excel_file = "accident_log.xlsx"
if not os.path.exists(excel_file):
    df = pd.DataFrame(columns=["Time", "Location", "Severity", "Vehicle Count", "Accident Image", "License Plate Image"])
    df.to_excel(excel_file, index=False)

def send_email(accident_img_path, plate_img_path, severity):
    msg = EmailMessage()
    msg['Subject'] = "Accident Alert!"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.set_content(f"An accident has been detected with a severity of {severity:.2f}%. Please check the attached images.")

    with open(accident_img_path, 'rb') as f:
        msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename='Accident.jpg')
    
    if plate_img_path and os.path.exists(plate_img_path):
        with open(plate_img_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype='image', subtype='jpeg', filename='License_Plate.jpg')
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)

def detect_accidents(frame):
    if frame.mean() < 100:
        frame, _ = image_dehazer.remove_haze(frame)
    
    results = accident_model(frame)
    vehicle_count = {"car": 0, "bike": 0}
    detected_classes = []
    accident_detected = False
    max_confidence = 0.0
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            if conf > 0.75:  # Increased confidence threshold
                label = class_names[cls]
                detected_classes.append(label)
                if 'accident' in label:
                    accident_detected = True
                    max_confidence = max(max_confidence, conf)
                    
                    if max_confidence <= 0.6:
                        color = (0, 255, 0)  # Green (Low Severity)
                    elif max_confidence <= 0.8:
                        color = (0, 165, 255)  # Orange (Medium Severity)
                    else:
                        color = (0, 0, 255)  # Red (High Severity)
                    
                    cv2.putText(frame, f"Severity: {max_confidence:.2f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                vehicle_count["car" if "car" in label else "bike"] += 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame, vehicle_count, max_confidence, accident_detected, detected_classes

def detect_license_plate(frame):
    results = anpr_model(frame)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_crop = frame[y1:y2, x1:x2]
            plate_path = os.path.join(plate_folder, f"plate_{int(time.time())}.jpg")
            cv2.imwrite(plate_path, plate_crop)
            return plate_path
    return None

st.title("Road Accident Detection with Emergency Notification")

# st.set_page_config(layout="wide")  # Increase video display size

detected_text = st.empty()
count_text = st.empty()
severity_text = st.empty()
progress_bar = st.progress(0)

uploaded_file = st.file_uploader("Upload a Video for Accident Detection", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    cap = cv2.VideoCapture(temp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, vehicle_count, severity, accident_detected, detected_classes = detect_accidents(frame)
        
        detected_text.markdown(f"**Detected Classes:** {', '.join(set(detected_classes))}")
        count_text.markdown(f"**Vehicle Count:** Cars: {vehicle_count.get('car', 0)}, Bikes: {vehicle_count.get('bike', 0)}")
        if accident_detected:
            severity_text.markdown(f"**Severity:** {severity:.2f}%")
        
        st.image(frame, channels="BGR", use_column_width=True)
        
        if accident_detected:
            accident_img_path = os.path.join(accident_folder, f"accident_{int(time.time())}.jpg")
            cv2.imwrite(accident_img_path, frame)
            plate_img_path = detect_license_plate(frame)
            
            send_email(accident_img_path, plate_img_path if plate_img_path else "", severity)
            st.warning("Accident detected! Emergency alert sent.")
        
        current_frame += 1
        progress_bar.progress(current_frame / total_frames)
    
    cap.release()
    st.success("Processing Complete")
