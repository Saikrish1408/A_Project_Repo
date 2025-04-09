import cv2
import streamlit as st
import numpy as np
import tempfile
from ultralytics import YOLO
import image_dehazer
from twilio.rest import Client
import time
import torch
torch.classes.__path__ = []

# Twilio Credentials

TWILIO_SID = "AC6194ea93f89b655cd831950f82f67638"
TWILIO_AUTH_TOKEN = "445161dc6ec171c5f1705a7d303d2aca"
TWILIO_PHONE_NUMBER = "+15673353638"
RECIPIENT_PHONE_NUMBER = "+919384729676"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

accident_start_time = None
accident_detected = False
ALERT_THRESHOLD = 2  # seconds


def send_twilio_alert(accident_type):

    message = f"ALERT: A {accident_type} has been detected. Immediate attention required! "
    client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=RECIPIENT_PHONE_NUMBER)



# Load YOLO model
model_path = r"runs\detect\train10\weights\best.pt"
model = YOLO(model_path)

# Define class names
class_names = ['bike', 'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
               'car', 'car_bike_accident', 'car_car_accident', 'car_object_accident',
               'car_person_accident', 'person']

st.title("Road Accident Detection with Rapid Emergency Notification")

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
    detected_classes_set = set()
    
    count=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, detected_classes = detect_accidents(frame)
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        
        detected_text.markdown(f"<h2 style='color:white;'>Detected Classes: </h2>" + f"<h2 style='color:red'>{', '.join(set(detected_classes))}</h2>", unsafe_allow_html=True)
        if any('accident' in accident for accident in detected_classes):
            if not accident_detected:
                accident_start_time = time.time()
                accident_detected = True
            elif time.time() - accident_start_time >= ALERT_THRESHOLD:
                send_twilio_alert(', '.join(detected_classes))
                accident_detected = False  # Reset after sending alert
        else:
            accident_detected = False  # Reset if no accident is found
        
        
        
    cap.release()
    
    if detected_classes_set:
        FRAME_WINDOW.image(frame)
        detected_text.markdown(f"<h1 style='color:red; font-size: 36px;'>Detected Classes: {', '.join(detected_classes_set)}</h1>", unsafe_allow_html=True)
    
    st.success("Processing Complete")