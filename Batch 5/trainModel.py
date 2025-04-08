"""
Train ML Model to Classify / Identify the person using extracted face embeddings
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


currentDir = os.getcwd()

# paths to embedding pickle file
embeddingPickle = os.path.join(currentDir, "output/Embeddings.pickle")

# path to save recognizer pickle file
recognizerPickle = os.path.join(currentDir, "output/Recognizers.pickle")

# path to save labels pickle file
labelPickle = os.path.join(currentDir, "output/Labels.pickle")

# loading embeddings pickle
data = pickle.loads(open(embeddingPickle, "rb").read())

# encoding labels by names
label = LabelEncoder()
labels = label.fit_transform(data["names"])

# getting embeddings
Embeddings = np.array(data["embeddings"])

print("Total number of embeddings : ", Embeddings.shape)
print("Total number of labels :", len(labels))


recognizer = KNeighborsClassifier(n_neighbors=5)
recognizer.fit(Embeddings, labels)

# write the actual face recognition model to disk
f = open(recognizerPickle, "wb")
f.write(pickle.dumps(recognizer))
f.close()



# write the label encoder to disk
f = open(labelPickle,"wb")
f.write(pickle.dumps(label))
f.close()

print("[Info] : Models are saved successfully :) ")
