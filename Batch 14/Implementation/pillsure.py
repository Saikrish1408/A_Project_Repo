import cv2
import time
from pyfirmata import Arduino, util
# Initialize the Arduino board
#port = '/dev/ttyUSB0'  # Change this to the correct port
port = 'COM4'
board = Arduino(port)

# Pin for the pill dispenser
DISPENSE_PIN = 4  
board.digital[DISPENSE_PIN].write(1)  # Set dispenser to OFF initially



import mediapipe as mp
import pandas as pd
import time
from datetime import datetime
import pygame
from gtts import gTTS
import os
from twilio.rest import Client
import numpy as np
import pickle
import cv2
import os
import model as embedding
import imutils
import argparse
import torch
import csv
from datetime import datetime

# we save 'RetinaFace' model at 'models/retinaface'
# we load retinaface model to detect facess
import torch.backends.cudnn as cudnn
from models.retinaface.config import cfg
from models.retinaface.prior_box import PriorBox
from models.retinaface.py_cpu_nms import py_cpu_nms
from models.retinaface.retinaface import RetinaFace
from models.retinaface.box_utils import decode , decode_landm
import torchvision.transforms.functional as F
import threading
from PIL import Image
import time
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

currentDir = os.getcwd()

# paths to embedding pickle file
embeddingPickle = os.path.join(currentDir, "output/Embeddings.pickle")

# path to save recognizer pickle file
recognizerPickle = os.path.join(currentDir, "output/Recognizers.pickle")

# path to save labels pickle file
labelPickle = os.path.join(currentDir, "output/Labels.pickle")

# # path to save prdictedImages
# predictedImg = os.path.join(currentDir, "predictedImg")
# if not os.path.exists(predictedImg):
#     os.mkdir(predictedImg)

# Use argparse to get image path on commend line



# loading 'RetinaFace' weights to detect facess
trained_model_path = "models/retinaface/weights/Final_Retinaface.pth"
cpu = True
confidence_threshold = 0.05
top_k = 5000
nms_threshold = 0.3
keep_top_k = 750
# save_image_path = "predictedImg"
vis_threshold = 0.6
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    #print('Missing keys:{}'.format(len(missing_keys)))
    #print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    #print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

### remove_prefix
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    #print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


### load_model
def load_model(model, pretrained_path, load_to_cpu):
    #print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

torch.set_grad_enabled(False)

#net and model
net = RetinaFace(phase="test")
net = load_model(net , trained_model_path, cpu)
net.eval()
print("Finished loading model!")
cudnn.benchmark = True
device = torch.device("cpu" if cpu else "cuda")
net = net.to(device)

resize = 1

# load embedding model
embedder = embedding.InceptionResnetV1(pretrained="vggface2").eval()

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(recognizerPickle, "rb").read())
label = pickle.loads(open(labelPickle, "rb").read())
# loading embeddings pickle
data = pickle.loads(open(embeddingPickle, "rb").read())

COLORS = np.random.randint(0, 255, size=(len(label.classes_), 3), dtype="uint8")

Embeddings = np.array(data["embeddings"])
names = np.array(data["names"])
print("Embeddings ", Embeddings.shape)
print("Names ", names.shape)
#print("Labels ", labels.shape)

# Excel file path
excel_file = "PatientLogs.xlsx"

# Initialize Excel DataFrame
columns = ["Patient Name", "Detected Time", "Pill Taken Time", "Status"]
df = pd.DataFrame(columns=columns)

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))


# Initialize pygame mixer
pygame.mixer.init()

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Twilio setup (replace with your Twilio credentials)
TWILIO_SID = "AC24c8f547bce5b98b1ef263a100d85711"
TWILIO_AUTH_TOKEN = "70a1a934175a676bbbc2126b6be1531e"
TWILIO_PHONE_NUMBER = "+12293983226"
RECIPIENT_PHONE_NUMBER = "+919345261039"
client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Excel file setup
excel_file = 'medication_log.xlsx'

# Check if the file exists, if not create it with headers
try:
    df = pd.read_excel(excel_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=['Date', 'Time', 'Patient Name', 'Medication Taken'])

# Function to log data to Excel
def log_to_excel(patient_name, status):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    time = now.strftime('%H:%M:%S')
    new_row = {'Date': date, 'Time': time, 'Patient Name': patient_name, 'Medication Taken': status}
    global df
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(excel_file, index=False)

# Function to convert text to speech
def text_to_speech(text, lang='en'):
    audio_file = f"audio_{text}.mp3"
    if not os.path.exists(audio_file):
        tts = gTTS(text=text, lang=lang)
        tts.save(audio_file)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Function to send missed pill alert
def send_twilio_alert(patient_name):
    message = f"ALERT: {patient_name} has missed their medication. Please check on them."
    client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=RECIPIENT_PHONE_NUMBER)

# Dummy face recognition function
def recognize_face(frame):
    img_pil = Image.fromarray(frame)
    img_pil = img_pil.resize((640, 360), resample=Image.BICUBIC)
    img_raw = np.array(img_pil)
    img_raw_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img = np.float32(img_raw)

    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    #print('net forward time: {:.4f}'.format(time.time() - tic))

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)

    for b in dets:
        if b[4] < vis_threshold:
            continue
        boxes = np.array(b[0:4])
        boxes = boxes.astype('int')
        (startX, startY, endX, endY) = boxes
        face = img_raw_rgb[startY:endY, startX:endX]

        try:
            faceRead = Image.fromarray(face)
            faceRead = faceRead.resize((130, 130), resample=Image.BICUBIC)
            faceRead = F.to_tensor(faceRead)
        except Exception as e:
            print(f"[Error] - resizing face: {e}")
            continue

        # getting embeddings for cropped faces
        faceEmbed = embedder(faceRead.unsqueeze(0))
        flattenEmbed = faceEmbed.squeeze(0).detach().numpy()

        # predicting class
        array = np.array(flattenEmbed).reshape(1, -1)
        preds = recognizer.predict_proba(array)[0]
        
        
        j = np.argmax(preds)
        proba = preds[j]
        name = label.classes_[j]
        #print(name,j)
        #print(name,Detected_faces)
        #if name in Detected_faces:
            #   continue

        result = np.where(names == name)
        resultEmbeddings = Embeddings[result]

        dists = []
        for emb in resultEmbeddings:
            d = distance(emb, flattenEmbed)
            dists.append(d)
        distarray = np.array(dists)
        min_dist = np.min(distarray)
                
        if proba == 1 and min_dist <= 0.8: 
            color = [int(c) for c in COLORS[j]]
            cv2.rectangle(img_raw, (startX, startY), (endX, endY), color, 2)
            text = "{}: {:.2f}".format(name, proba)
            cv2.putText(img_raw, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print('face detected',text)

            return name, True
# Pill monitoring function
def monitor_pill_intake(frame, patient_name):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)
    pill_taken = False

    if hand_results.multi_hand_landmarks and face_results.detections:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Get index fingertip coordinates
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            finger_x, finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Get face bounding box
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                face_x_min = int(bboxC.xmin * w)
                face_y_min = int(bboxC.ymin * h)
                face_width = int(bboxC.width * w)
                face_height = int(bboxC.height * h)

                # Check if the hand is near the mouth
                if (face_x_min < finger_x < face_x_min + face_width) and (face_y_min < finger_y < face_y_min + face_height):
                    pill_taken = True
                    break

    return pill_taken
# Function to log data to Excel
def log_to_excel(patient_name, status, pill_taken_time=None):
    detected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pill_time = pill_taken_time if pill_taken_time else "N/A"

    global df
    new_data = pd.DataFrame([{
        "Patient Name": patient_name,
        "Detected Time": detected_time,
        "Pill Taken Time": pill_time,
        "Status": status
    }])

    df = pd.concat([df, new_data], ignore_index=True)
    df.to_excel(excel_file, index=False)
    print(f"Logged {patient_name} with status {status}")


# Initialize the webcam
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(1)

# List to track detected patients
detected_patients = []
running = True  # Flag to control the loop

# Function to dispense a pill
def dispense_pill():
    # pass
    print("Dispensing Pill...")
    board.digital[DISPENSE_PIN].write(0)  # Activate dispenser
    time.sleep(15)  # Dispense for 2 seconds
    board.digital[DISPENSE_PIN].write(1)  # Stop dispenser
    print("Pill Dispensed.")

try:
    while cap.isOpened() and running:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Step 1: Face Recognition
        recognition_result = recognize_face(frame)
        patient_name, face_recognized = recognition_result if recognition_result else ("unknown", False)
        
        if patient_name == "unknown" or not face_recognized:
            cv2.putText(frame, "Waiting for known patient...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                running = False
                break
            continue  

        if patient_name in detected_patients:
            text_to_speech(f"{patient_name} already detected. Please wait for the next patient.")
            continue  

        text_to_speech(f"{patient_name} detected. Please take your pill.")
        start_time = time.time()
        pill_taken = False

        # Step 2: Dispense the Pill
        dispense_pill()
        log_to_excel(patient_name, "Detected")

        # Step 3: Pill Monitoring (15 seconds)
        while time.time() - start_time < 30 and running:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            pill_taken = monitor_pill_intake(frame, patient_name)

            if pill_taken:
                cv2.putText(frame, 'Pill Taken', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                break
            else:
                cv2.putText(frame, 'Waiting for Pill...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Pill Monitoring', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                running = False
                break

        # Step 4: Logging & Alerting
        if pill_taken:
            text_to_speech("Pill detected as taken. Logging entry.")
            log_to_excel(patient_name, "Taken")
        else:
            text_to_speech("Pill not taken")
            send_twilio_alert(f'{patient_name} Pill not taken. Sending alert.')
            log_to_excel(patient_name, "Missed")

        if not pill_taken:
            log_to_excel(patient_name, "Missed")
            
        detected_patients.append(patient_name)
        

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            running = False
            break  

    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"An error occurred: {e}")
    cap.release()
    cv2.destroyAllWindows()