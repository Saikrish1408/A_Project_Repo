from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import pickle
import pandas as pd
import numpy as np
from flask import session
import uuid
import json
from datetime import datetime
from PIL import Image
import cv2
import os
import os

import tensorflow as tf
import random
import os
import PIL
import cv2
import keras
import matplotlib.pyplot as plt

def preprocess(IMG_SAVE_PATH):
    dataset = []
   
    try:
                imgpath=PIL.Image.open(IMG_SAVE_PATH)
                imgpath=imgpath.convert('RGB')
                img = np.asarray(imgpath)
                img = cv2.resize(img, (331,331))
                img=img/255.
                dataset.append(img)
    except FileNotFoundError:
                print('Image file not found. Skipping...')
    return dataset
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, BatchNormalization, GlobalAveragePooling2D, Dropout, Concatenate, Input
from tensorflow.keras.models import Model

# Custom CNN model definition
def create_custom_cnn(input_shape=(331, 331, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3))(inputs)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(32, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(64, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)  # Flatten to create a 1D vector
    custom_cnn_model = Model(inputs, x)
    
    return custom_cnn_model

# Base model (InceptionResNetV2) definition
def create_combined_model(num_classes=2, cnn_input_shape=(331, 331, 3), inception_input_shape=(331, 331, 3)):
    # Custom CNN model
    custom_cnn = create_custom_cnn(input_shape=cnn_input_shape)
    
    # Pretrained InceptionResNetV2 (without top layers)
    base_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                         weights='imagenet',
                                                         input_shape=inception_input_shape)
    base_model.trainable = False  # Freeze InceptionResNetV2 layers
    
    # Flatten the output of the base model (InceptionResNetV2)
    inception_output = Flatten()(base_model.output)
    
    # Concatenate the output of custom CNN and InceptionResNetV2
    concatenated = Concatenate()([custom_cnn.output, inception_output])
    
    # Add dense layers after concatenation
    x = Dense(512, activation='relu')(concatenated)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    
    # Output layer (adjust the output for the number of classes)
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=[custom_cnn.input, base_model.input], outputs=output)
    
    return model

# Instantiate the model
PET = create_combined_model(num_classes=2)
PET.load_weights('PET.h5')
MRI = create_combined_model(num_classes=2)
MRI.load_weights('MRI.h5')
CTSCAN = create_combined_model(num_classes=2)
CTSCAN.load_weights('CTSCANMODEL.h5')

MRI.summary()



app = Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'

CORS(app, resources={r"/*": {"origins": "*"}}) 

@app.route('/')
def hello():
    message= ''
    return render_template("lung.html",message = message,name="Food Label")

@app.route('/index')
def index():
    message= ''
    return render_template("lung.html",message = message,name="Food Label")


@app.route('/lung')
def lung():
    message= ''
    return render_template("lung.html",message = message,name="Food Label")



@app.route('/lungdetect', methods=['POST'])
def lungdetect():
    photo=request.files["photo"]
    photo_name = photo.filename
    scanType= request.form["scanType"]
    photo.save("static/imagelung/" + photo_name)
    image=cv2.imread("static/imagelung/"+ photo_name)
    message="No Cancer"
    traindata=preprocess("static/imagelung/" + photo_name)
    xtest=np.array(traindata)
    model=PET
    if(scanType=="CTSCAN"):
      model=CTSCAN
    elif(scanType=="MRI"):
      model=MRI
    Y_pred = model.predict([xtest, xtest])
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    print(Y_pred_classes)
    
    if(Y_pred_classes[0]==0):
      message="CANCER"
    print(message)
    print(scanType)
    print(image)
    values = [2, 2.5, 3]
    selected_value = random.choice(values)



    return render_template("lungresult.html", imageangleType=selected_value, message = message,imagetype = scanType,image="static/imagelung/" + photo_name)

  


@app.route('/home')
def home():
    message= ''
    return render_template("lung.html",message = message,name="Food Label")



if __name__ == '__main__':
    app.run(debug=True)
