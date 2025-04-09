import cv2
import streamlit as st
import numpy as np
import tempfile
import os
import time
from ultralytics import YOLO
import image_dehazer
from sort import Sort  # SORT tracking algorithm
import smtplib
from email.message import EmailMessage
import random

# Load YOLO models
accident_model_path = r"best_accident.pt"
anpr_model_path = r"best_anpr.pt" 

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

# Email Credentials (Replace with your details)
SENDER_EMAIL = "amsasaravanan101214@gmail.com"
APP_PASSWORD = "cxbr mmab lqoy ayan"  # App password, not regular password
RECEIVER_EMAIL = "k.gokulappaduraikjgv@gmail.com"

# Streamlit App Title
st.title("Road Accident Detection with Vehicle Counting & Number Plate Recognition")

# Function to send email alert
def send_email_alert(plate_number, accident_image_path, plate_image_path, severity, vehicle_count, accident_time, location):
    """Sends an email with accident details and attached images."""
    
    msg = EmailMessage()
    msg["Subject"] = "🚨 Emergency Alert: Accident Detected!"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    # Email body content
    msg.set_content(f"""
    Accident Detected!
    - License Plate: {plate_number}
    - Time: {accident_time}
    - Location: {location}
    - Severity: {severity}%
    - Vehicle Count: Cars: {vehicle_count['car']}, Bikes: {vehicle_count['bike']}
    
    Attached are the accident image and number plate image.
    """)

    # Attach accident image
    if os.path.exists(accident_image_path):
        with open(accident_image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="accident.jpg")

    # Attach number plate image
    if os.path.exists(plate_image_path):
        with open(plate_image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="number_plate.jpg")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("[✅] Email Sent Successfully!")
    except Exception as e:
        print(f"[❌] Error Sending Email: {e}")

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
    if len(detections) > 0:
        tracked_objects = tracker.update(np.array(detections))
    else:
        tracked_objects = []

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
                file_path = os.path.join(save_folder, f"plate_{timestamp}.jpg")

                cv2.imwrite(file_path, plate_crop)
                print(f"Saved: {file_path}")
                return file_path  # Return saved file path

    return None

# Function to generate random location (latitude and longitude)
def generate_random_location():
    lat = round(random.uniform(-90, 90), 6)
    lon = round(random.uniform(-180, 180), 6)
    return f"Lat: {lat}, Lon: {lon}"

# Video uploader
uploaded_file = st.file_uploader("Upload a Video for Detection", type=["mp4", "avi", "mov"])

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

    # Flag to ensure email is sent only once per accident
    email_sent = False

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

        if accident_detected and not email_sent:
            plate_path = detect_and_save_plate(frame)
            if plate_path:
                plate_text.markdown(f"<h2 style='color:green;'>Number Plate Saved: {plate_path}</h2>", unsafe_allow_html=True)
                
                # Save the accident-detected frame
                accident_image_path = os.path.join(save_folder, f"accident_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                cv2.imwrite(accident_image_path, frame)

                # Get current time and random location
                accident_time = time.strftime("%Y-%m-%d %H:%M:%S")
                location = generate_random_location()

                # Send email with accident details
                send_email_alert("Detected Plate", accident_image_path, plate_path, severity, vehicle_count, accident_time, location)
                email_sent = True  # Ensure email is sent only once

    cap.release()
    st.success("Processing Complete")