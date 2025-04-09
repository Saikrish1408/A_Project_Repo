import smtplib
from email.message import EmailMessage
import os

# Email Credentials (Replace with your details)
SENDER_EMAIL = "amsasaravanan101214@gmail.com"
APP_PASSWORD = "cxbr mmab lqoy ayan"  # App password, not regular password
RECEIVER_EMAIL = "vijayaprabhaappar@gmail.com"

# Function to send email alert
def send_email_alert(plate_number, accident_image_path, plate_image_path, severity, vehicle_count, accident_time, location):
    """
    Sends an email with accident details and attached images.

    Parameters:
        plate_number (str): Detected license plate number.
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

    # Send the email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("[✅] Email Sent Successfully!")
    except Exception as e:
        print(f"[❌] Error Sending Email: {e}")

# Example usage
if __name__ == "__main__":
    # Example data
    plate_number = "TN-10-XYZ-1234"
    accident_image_path = "accident_detected_frame.jpg"  # Path to the accident-detected image
    plate_image_path = "detected_plate.jpg"  # Path to the number plate image
    severity = 100.0  # Severity percentage
    vehicle_count = {"car": 2, "bike": 1}  # Vehicle count
    accident_time = "2023-10-15 14:30:00"  # Time of the accident
    location = "Lat: 12.9716, Lon: 77.5946"  # Location (latitude and longitude)

    # Send the email
    send_email_alert(plate_number, accident_image_path, plate_image_path, severity, vehicle_count, accident_time, location)