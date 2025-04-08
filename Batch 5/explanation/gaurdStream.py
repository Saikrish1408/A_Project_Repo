import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import cv2
import shutil
import time

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160)

# Function to delete empty folders
def delete_empty_folders(folder_path):
    deleted_folders = []
    for folder_name in os.listdir(folder_path):
        folder = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder) and not os.listdir(folder):  # If folder is empty
            shutil.rmtree(folder)  # Delete the empty folder
            deleted_folders.append(folder_name)
    return deleted_folders

# Function to process images, extract embeddings, and train the model
def process_images_and_train_model(dataset_dir, output_dir):
    deleted_folders = delete_empty_folders(dataset_dir)
    if deleted_folders:
        st.error(f"Deleted empty folders: {', '.join(deleted_folders)}. Please retake pictures.")
        return

    embedder = InceptionResnetV1(pretrained='vggface2').eval()
    imagePaths = []
    names = []
    imageIDs = []
    embeddings = []

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                imagePath = os.path.join(person_dir, image_name)
                image = Image.open(imagePath)
                name = person_name
                imageID = image_name.split('.')[0]

                try:
                    face = mtcnn(image)
                except Exception as e:
                    st.error(f"[Error] Processing image {imagePath}: {str(e)}")
                    continue

                if face is not None:
                    faceEmbed = embedder(face.unsqueeze(0))
                    flattenEmbed = faceEmbed.squeeze(0).detach().numpy()

                    imagePaths.append(imagePath)
                    imageIDs.append(imageID)
                    names.append(name)
                    embeddings.append(flattenEmbed)

    embeddings_pickle_path = os.path.join(output_dir, "Embeddings.pickle")
    data = {
        "paths": imagePaths,
        "names": names,
        "imageIDs": imageIDs,
        "embeddings": embeddings
    }

    with open(embeddings_pickle_path, 'wb') as f:
        pickle.dump(data, f)

    label = LabelEncoder()
    labels = label.fit_transform(names)

    recognizer = KNeighborsClassifier(n_neighbors=5)
    recognizer.fit(np.array(embeddings), labels)

    recognizer_pickle_path = os.path.join(output_dir, "Recognizers.pickle")
    labels_pickle_path = os.path.join(output_dir, "Labels.pickle")

    with open(recognizer_pickle_path, 'wb') as f:
        pickle.dump(recognizer, f)

    with open(labels_pickle_path, 'wb') as f:
        pickle.dump(label, f)

    st.success(f"Model and embeddings saved in {output_dir}")

def capture_images_from_camera(person_name, role, num_images=10):
    captured_images = 0
    st.info("Capturing faces... Please wait.")

    for i in range(num_images):
        #cap = cv2.VideoCapture('rtsp://admin:vision%40321@192.168.29.221:554/Streaming/Channels/101')
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to grab frame.")
            break

        person_folder = os.path.join(f"gaurd_faces_id/{person_name}_{role}")  # Include role in folder name
        os.makedirs(person_folder, exist_ok=True)

        img_path = os.path.join(person_folder, f"{person_name}_{role}_{captured_images + 1}.png")
        cv2.imwrite(img_path, frame)
        captured_images += 1

        st.success(f"Captured {captured_images}/{num_images} images for {person_name} as {role}.")
        
    cap.release()
        #time.sleep(1)
def start_camera_stream():
    os.system('python gaurdStreamIt.py')  # Assuming the streaming script is saved as 'your_streaming_script.py'

# Streamlit app setup
st.title("Gaurdian AI Unknown person and Unsual acitivity detection for gaurd children")
st.sidebar.header("Upload Person Images")

dataset_dir = 'gaurd_faces_id'
output_dir = 'gaurd_output'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

person_name = st.sidebar.text_input("Enter Person's Name:")
role_options = ['Friend', 'Family']
role_selection = st.sidebar.selectbox("Select Role:", role_options)

camera_button = st.sidebar.button("Open Camera")

if camera_button:
    if not person_name:
        st.error("Please enter the person's name before opening the camera.")
    else:
        capture_images_from_camera(person_name, role_selection.lower())

# Train model button
if st.button("Train Model"):
    if os.path.exists(os.path.join(dataset_dir)):
        process_images_and_train_model(dataset_dir, output_dir)
    else:
        st.error("Please upload or capture images first.")
# Add a button to start the camera stream after training
if st.button("Start Camera Stream"):
    start_camera_stream()
