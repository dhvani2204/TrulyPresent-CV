import cv2 as cv
import numpy as np
import os
os.environ['TP_CPP_MIN_LOG_LEVEL']='2'
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine

facenet = FaceNet()
faces_embeddings = np.load('faces_embeddings_done_final.npz')
X, Y = faces_embeddings['arr_0'], faces_embeddings['arr_1']
encoder = LabelEncoder()
encoder.fit(Y)
haarcascade = cv.CascadeClassifier("haarcascade_frontalface_default (2).xml")

# Load the uploaded image
image_path = 'image2.jpeg'  
frame = cv.imread(image_path)

rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
faces = haarcascade.detectMultiScale(gray_img, 1.3, 5)

for x, y, w, h in faces:
    face_img = rgb_img[y:y+h, x:x+w]
    resized_img = cv.resize(face_img, (160, 160))
    input_face = np.expand_dims(resized_img, axis=0)
    query_embedding = facenet.embeddings(input_face)
    min_distance = float('inf')
    recognized_label = 'Unknown'

    for i, embedding in enumerate(X):
        # Calculate cosine distance between query embedding and database embedding
        query_embedding_flat = query_embedding.flatten()
        embedding_flat = embedding.flatten()
        distance = cosine(query_embedding_flat, embedding_flat)

        # Find the closest match (smallest distance)
        if distance < min_distance:
            min_distance = distance
            recognized_label = Y[i]

    # Draw rectangle and label on the frame
    cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv.putText(frame, recognized_label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Display the resulting frame
cv.namedWindow("Resized_Window", cv.WINDOW_NORMAL) 
  
# Using resizeWindow() 
cv.resizeWindow("Resized_Window", 500,700) 
cv.imshow('Resized_Window', frame)
cv.waitKey(0)
cv.destroyAllWindows()
