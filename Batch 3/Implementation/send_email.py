import smtplib
from email.message import EmailMessage
import os

# Email Credentials (Replace with your details)
SENDER_EMAIL = "amsasaravanan101214@gmail.com"
APP_PASSWORD = "cxbr mmab lqoy ayan"  # App password, not regular password
RECEIVER_EMAIL = "amsasaravanan101214@gmail.com"

def send_email_alert(plate_number, image_path, severity, vehicle_count, accident_time):
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
    - Severity: {severity}
    - Vehicle Count: {vehicle_count}
    
    Attached is the accident image.
    """)

    # Attach accident image
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="accident.jpg")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("[✅] Email Sent Successfully!")
    except Exception as e:
        print(f"[❌] Error Sending Email: {e}")

# Example usage:
send_email_alert("TN-10-XYZ-1234", r"Pothale_Dection_yolov11\pothole_detected_frames\2025-03-02\06-46-57_.png", "High", 4, "2025-03-04 14:30:00")
