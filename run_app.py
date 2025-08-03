import cv2
import numpy as np
import pygame
import random
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('emotion_model.h5')

# Load Haarcascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion categories (must match your dataset)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}

# Define songs & jokes for emotions
songs = ["calm_music.mp3", "relaxing_tune.mp3"]
jokes = ["joke1.mp3", "joke2.mp3"]

# Initialize pygame for playing sounds
pygame.init()
pygame.mixer.init()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # Extract face region
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predict emotion
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        emotion = emotion_dict[maxindex]

        # Display detected emotion
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Play music/joke based on emotion
        if emotion in ["Angry", "Sad"]:
            pygame.mixer.music.load(random.choice(songs))
            pygame.mixer.music.play()
        elif emotion == "Fearful":
            pygame.mixer.music.load(random.choice(jokes))
            pygame.mixer.music.play()

    cv2.imshow('Driver Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
