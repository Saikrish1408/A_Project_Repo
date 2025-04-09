import cv2
import streamlit as st
import numpy as np
import tempfile
import os
import time
from ultralytics import YOLO
import image_dehazer
from deep_sort_realtime.deepsort_tracker import DeepSort  # DeepSORT tracking algorithm
import smtplib
from email.message import EmailMessage
import random

# Load YOLO models
accident_model_path = r"best_accident.pt"
anpr_model_path = r"best_anpr.pt"

# Define folders
accident_images_folder = "ACCIDENT_IMAGES"
detected_plates_folder = "DETECTED_PLATES"

# Create folders if they don't exist
os.makedirs(accident_images_folder, exist_ok=True)
os.makedirs(detected_plates_folder, exist_ok=True)

accident_model = YOLO(accident_model_path)
anpr_model = YOLO(anpr_model_path)

# Define class names
class_names = ['bike', 'bike_bike_accident', 'bike_object_accident', 'bike_person_accident',
               'car', 'car_bike_accident', 'car_car_accident', 'car_object_accident',
               'car_person_accident', 'person']

# Initialize DeepSORT Tracker
deep_sort = DeepSort(max_age=50)

# Folder to save number plate images
save_folder = "Detected_Plates"
os.makedirs(save_folder, exist_ok=True)

# # Email Credentials (Replace with your details)
# SENDER_EMAIL = "amsasaravanan101214@gmail.com"
# APP_PASSWORD = "cxbr mmab lqoy ayan"  # App password, not regular password
# RECEIVER_EMAIL = "vijayaprabha764@gmail.com"

# Streamlit App Title
st.title("Road Accident Detection with Vehicle Counting & Number Plate Recognition")

# Function to send email alert
SENDER_EMAIL = "amsasaravanan101214@gmail.com"
APP_PASSWORD = "cxbr mmab lqoy ayan"  # App password, not regular password
RECEIVER_EMAIL = "vijayaprabhaappar@gmail.com"



# Function to send email alert
def send_email_alert(accident_image_path, plate_image_path, severity, vehicle_count, accident_time, location):
    """
    Sends an email with accident details and attached images.

    Parameters:
        accident_image_path (str): Path to the accident-detected image.
        plate_image_path (str): Path to the number plate image.
        severity (float): Severity percentage of the accident.
        vehicle_count (dict): Dictionary with counts of cars and bikes.
        accident_time (str): Time of the accident.
        location (str): Location of the accident (latitude and longitude).
    """
    # Create the email message
    msg = EmailMessage()
    msg["Subject"] = "🚨 Emergency Alert: Accident Detected!"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    # Email body content
    msg.set_content(f"""
    Accident Detected!
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
    try:
        if os.path.exists(plate_image_path):
            with open(plate_image_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="number_plate.jpg")
    except:
        pass
    # Send the email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("[✅] Email Sent Successfully!")
    except Exception as e:
        print(f"[❌] Error Sending Email: {e}")

# Function to detect accidents and track vehicles
def detect_accidents(frame):
    # if frame.mean() < 100:
        # frame, _ = image_dehazer.remove_haze(frame)

    results = accident_model(frame)
    detected_classes = []
    detections = []
    accident_detected = False  # Flag to check for accidents

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.5:
                label = class_names[cls]
                detected_classes.append(label)
                color = (0, 0, 255) if 'accident' in label else (0, 255, 0)
                font_scale = 1 if 'accident' in label else 0.5

                if 'accident' in label:
                    accident_detected = True  # Mark accident detection

                # Save detection for tracking
                detections.append(([x1, y1, x2, y2], conf, label))

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

    # Update DeepSORT tracker
    tracks = deep_sort.update_tracks(detections, frame=frame)

    # Count Vehicles
    vehicle_count = {"car": 0, "bike": 0}
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        label = track.get_det_class()
        if label in ["car", "bike"]:
            vehicle_count[label] += 1

    # Calculate severity percentage
    total_objects = len(detected_classes)
    accident_objects = sum(1 for x in detected_classes if "accident" in x)
    severity = (accident_objects / total_objects) * 100 if total_objects > 0 else 0

    return frame, detected_classes, vehicle_count, severity, accident_detected

def detect_and_save_plate(frame, model_path="best_anpr.pt", save_folder="detected_plates"):
    """
    Detects number plates in a given frame, saves the detected plate as an image,
    and returns the saved file path.

    Parameters:
    - frame (numpy.ndarray): Input video frame.
    - model_path (str): Path to the YOLO model file.
    - save_folder (str): Directory to save detected plate images.

    Returns:
    - str or None: Path of the saved plate image, or None if no plate was detected.
    """
    # Load YOLO model
    model = YOLO(model_path)
    results = model(frame)
    plate_image_path = None  # Store file path if a plate is detected

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if conf > 0.5:  # Confidence threshold
                plate_crop = frame[y1:y2, x1:x2]
                
                if plate_crop.size > 0:
                    # Ensure directory exists
                    os.makedirs(save_folder, exist_ok=True)
                    
                    # Generate filename with timestamp
                    file_path = os.path.join(save_folder, f"plate_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
                    
                    # Save plate image
                    cv2.imwrite(file_path, plate_crop)
                    print(f"Saved Plate Image: {file_path}")
                    
                    plate_image_path = file_path  # Store saved file path
    
    return plate_image_path

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


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        frame, detected_classes, vehicle_count, severity, accident_detected = detect_accidents(frame)
 

        frame = cv2.resize(frame, (640, 360))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(frame)
        detected_text.markdown(f"<h2 style='color:red;'>Detected Classes: {', '.join(set(detected_classes))}</h2>", unsafe_allow_html=True)
        count_text.markdown(f"<h2 style='color:blue;'>Vehicle Count - Cars: {vehicle_count['car']}, Bikes: {vehicle_count['bike']}</h2>", unsafe_allow_html=True)
        severity_text.markdown(f"<h2 style='color:orange;'>Severity: {severity:.2f}%</h2>", unsafe_allow_html=True)

        if accident_detected and severity>25:
            # Save accident image
            accident_image_path = os.path.join(accident_images_folder, f"accident_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
            cv2.imwrite(accident_image_path, frame)
            print(f"Saved Accident Image: {accident_image_path}")

            # Detect and save number plate
            plate_path = detect_and_save_plate(frame)
             # Get current time and location
            accident_time = time.strftime("%Y-%m-%d %H:%M:%S")
            location = generate_random_location()  # Replace with actual location logic
                # Send Email
            # send_email_alert("Detected Plate", accident_image_path, plate_path, severity, vehicle_count, accident_time, location)
            send_email_alert(accident_image_path, plate_path, severity, vehicle_count, accident_time, location)
                
    cap.release()
    st.success("Processing Complete")