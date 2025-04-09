import cv2
import streamlit as st
import numpy as np
import tempfile
from ultralytics import YOLO
import image_dehazer
from twilio.rest import Client
import smtplib
from email.message import EmailMessage
import os
import time
from datetime import datetime

# Twilio Credentials (Replace with actual credentials)
TWILIO_SID = "AC6194ea93f89b655cd831950f82f67638"
TWILIO_AUTH_TOKEN = "445161dc6ec171c5f1705a7d303d2aca"
TWILIO_PHONE_NUMBER = "+15673353638"
RECIPIENT_PHONE_NUMBER = "+919384729676"
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Email Credentials (Replace with your details)
SENDER_EMAIL = "amsasaravanan101214@gmail.com"
APP_PASSWORD = "cxbr mmab lqoy ayan"  # App password, not regular password
RECEIVER_EMAIL = "amsasaravanan101214@gmail.com"
# Load YOLO models
model_vehicle = YOLO(r"runs\detect\train10\weights\best.pt")  # Accident detection model
model_plate = YOLO(r"C:\Users\iproat26\Desktop\Anpr\License-Plate-Detection\best.pt")  # License plate model

# Define class names
class_names = ['bike', 'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
               'car', 'car_bike_accident', 'car_car_accident', 'car_object_accident',
               'car_person_accident', 'person']

# Streamlit UI
st.title("🚦 Real-Time Road Accident Detection & Emergency Alert System")

uploaded_file = st.file_uploader("📂 Upload a Video for Accident Detection", type=["mp4", "avi", "mov"])

accident_start_time = None
accident_detected = False
ALERT_THRESHOLD = 2  # seconds before sending alert


def send_twilio_alert(accident_type):
    """Send emergency alert via SMS"""
    message = f"🚨 ALERT: A {accident_type} has been detected! Immediate attention required!"
    client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=RECIPIENT_PHONE_NUMBER)


def send_email_alert(plate_number, image_path, severity, vehicle_count, accident_time):
    """Send email with accident details & images"""
    msg = EmailMessage()
    msg["Subject"] = "🚨 Emergency Alert: Accident Detected!"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(f"""
    Accident Detected!
    - License Plate: {plate_number}
    - Time: {accident_time}
    - Severity: {severity}
    - Vehicle Count: {vehicle_count}
    
    Attached is the accident image.
    """)

    with open(image_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="accident.jpg")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("[✅] Email Sent Successfully!")
    except Exception as e:
        print("[❌] Error Sending Email:", e)


def detect_accidents(frame):
    """Detect accidents & vehicles, apply dehazing if needed"""
    if frame.mean() < 100:
        frame, _ = image_dehazer.remove_haze(frame)

    results = model_vehicle(frame)
    detected_classes = []
    accident_detected = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.3:
                label = class_names[cls]
                detected_classes.append(label)
                if 'accident' in label:
                    accident_detected = True
                    color = (0, 0, 255)  # Red for accidents
                    font_scale = 1
                else:
                    color = (0, 255, 0)  # Green for normal vehicles
                    font_scale = 0.5
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    return frame, detected_classes, accident_detected


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

        frame, detected_classes, accident_happened = detect_accidents(frame)
        frame = cv2.resize(frame, (640, 360))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

        detected_text.markdown(f"<h2 style='color:white;'>Detected Classes: </h2>"
                               f"<h2 style='color:red'>{', '.join(set(detected_classes))}</h2>", unsafe_allow_html=True)

        if accident_happened:
            if not accident_detected:
                accident_start_time = time.time()
                accident_detected = True
            elif time.time() - accident_start_time >= ALERT_THRESHOLD:
                send_twilio_alert(', '.join(detected_classes))
                accident_detected = False  # Reset after sending alert

                # Capture accident image & extract license plate
                accident_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                accident_folder = f"accidents/{accident_time}"
                os.makedirs(accident_folder, exist_ok=True)
                image_path = f"{accident_folder}/accident.jpg"
                cv2.imwrite(image_path, frame)

                # License plate detection
                results_plate = model_plate(frame)
                plate_number = "UNKNOWN"
                for plate in results_plate[0].boxes.data.cpu().numpy():
                    px1, py1, px2, py2, _, _ = map(int, plate)
                    plate_crop = frame[py1:py2, px1:px2]
                    plate_path = f"{accident_folder}/plate.jpg"
                    cv2.imwrite(plate_path, plate_crop)
                    plate_number = f"PLATE_{accident_time}"

                # Determine severity
                vehicle_count = len(detected_classes)
                severity = "High" if vehicle_count > 3 else "Medium" if vehicle_count > 1 else "Low"

                # Send email alert
                send_email_alert(plate_number, image_path, severity, vehicle_count, accident_time)

        else:
            accident_detected = False  # Reset if no accident is found

    cap.release()
    st.success("✅ Processing Complete!")
