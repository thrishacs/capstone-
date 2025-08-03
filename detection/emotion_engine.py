import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pyttsx3
import random
from PIL import Image
import os
import csv
from pygame import mixer
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from collections import Counter

# ✅ Spotify Credentials
SPOTIPY_CLIENT_ID = 'dcef7601624b45d79b3a9ecf762844fd'
SPOTIPY_CLIENT_SECRET = '95d6027ac89749f3b81e30144f5be757'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# ✅ Spotify Setup
try:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="user-read-playback-state,user-modify-playback-state",
        open_browser=True
    ))
    SPOTIFY_ENABLED = True
except:
    sp = None
    SPOTIFY_ENABLED = False

# ✅ Model Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 7)
)
model.load_state_dict(torch.load("../models/emotion_model_resnet50.pth", map_location=device))
model.to(device)
model.eval()

# ✅ Labels & Spotify Track URIs
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_tracks = {
    "Angry": "spotify:track:5FVd6KXrgO9B3JPmC8OPst",
    "Disgust": "spotify:track:3AJwUDP919kvQ9QcozQPxg",
    "Fear": "spotify:track:0GONea6G2XdnHWjNZd6zt3",
    "Happy": "spotify:track:3KkXRkHbMCARz0aVfEt68P",
    "Neutral": "spotify:track:2X485T9Z5Ly0xyaghN73ed",
    "Sad": "spotify:track:7qEHsqek33rTcFNT9PFqLf",
    "Surprise": "spotify:track:6WrI0LAC5M1Rw2MnX2ZvEg",
}

# ✅ Audio Setup
mixer.init()
engine = pyttsx3.init()
engine.setProperty('rate', 150)
ASSETS_PATH = "../assets"
LOG_FILE = "../logs/emotion_log.csv"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# ✅ Log Emotion to CSV
def log_emotion(emotion, confidence):
    try:
        file_exists = os.path.isfile(LOG_FILE)
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists or os.stat(LOG_FILE).st_size == 0:
                writer.writerow(["Timestamp", "Emotion", "Confidence"])
            writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), emotion, f"{confidence * 100:.2f}%"])
            f.flush()
            print("📝 Emotion logged successfully.")
    except Exception as e:
        print(f"⚠️ Failed to log emotion: {e}")

# ✅ Plot Emotion Trends
def plot_emotion_trends():
    try:
        with open(LOG_FILE, mode='r') as f:
            reader = csv.DictReader(f)
            emotions = [row['Emotion'] for row in reader]
            counts = Counter(emotions)

        colors = {
            "Happy": "#FFD700",
            "Sad": "#1E90FF",
            "Angry": "#FF4500",
            "Disgust": "#6B8E23",
            "Fear": "#8B008B",
            "Surprise": "#FF69B4",
            "Neutral": "#A9A9A9",
            "Emotion Uncertain": "#808080"
        }

        plt.figure(figsize=(8, 5))
        plt.bar(counts.keys(), counts.values(), color=[colors.get(emotion, '#999999') for emotion in counts.keys()])
        plt.xlabel("Emotions")
        plt.ylabel("Frequency")
        plt.title("Detected Emotion Trends")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot trends: {e}")

# ✅ TTS
def speak(text):
    print(f"🎤 Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

def speak_affirmation():
    affirmations = [
        "Stay calm and drive safely. Everything is okay!",
        "Take a deep breath and focus on the road.",
        "You're in control. Drive with a positive mindset."
    ]
    speak(random.choice(affirmations))

def tell_joke():
    jokes = [
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What do you call fake spaghetti? An impasta!",
        "Why don’t skeletons fight each other? They don’t have the guts."
    ]
    speak(random.choice(jokes))

def tell_riddle():
    riddles = [
        "What has hands but can’t clap? A clock.",
        "I speak without a mouth and hear without ears. What am I? An echo.",
        "What comes once in a minute, twice in a moment, but never in a thousand years? The letter M."
    ]
    speak(random.choice(riddles))

# ✅ Music Control
def play_emotion_music(emotion):
    track_uri = emotion_tracks.get(emotion)
    played = False

    if SPOTIFY_ENABLED and sp and track_uri:
        try:
            devices = sp.devices()
            if devices and devices.get('devices'):
                sp.start_playback(uris=[track_uri])
                print(f"🎵 Playing from Spotify: {emotion}")
                print("🛰️  Source: Spotify")
                played = True
            else:
                print("⚠️ No active Spotify device found. Switching to offline.")
        except Exception as e:
            print(f"⚠️ Spotify error: {e}. Switching to offline.")

    if not played:
        try:
            for file in os.listdir(ASSETS_PATH):
                if emotion.lower() in file.lower() and file.endswith(".mp3"):
                    local_path = os.path.join(ASSETS_PATH, file)
                    mixer.music.load(local_path)
                    mixer.music.play()
                    print(f"🎧 Playing offline: {file}")
                    print("🛰️  Source: Offline")
                    played = True
                    break
            if not played:
                print(f"❌ No offline music found for emotion: {emotion}")
                print("🛰️  Source: None")
        except Exception as e:
            print(f"❌ Offline fallback failed: {e}")
            print("🛰️  Source: None")

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ GUI Window
root = tk.Tk()
root.title("Emotion Detection GUI")
root.geometry("500x300")
root.configure(bg="#f0f8ff")

status_icon = tk.StringVar()
status_icon.set("⏸️")
status_label = ttk.Label(root, textvariable=status_icon, font=("Arial", 14), background="#f0f8ff", foreground="#333")
status_label.pack(pady=20)

btn_frame = ttk.Frame(root)
btn_frame.pack(pady=10)

def start_detection():
    cap = cv2.VideoCapture(0)
    last_emotion = "Neutral"
    emotion_start_time = time.time()
    CONFIDENCE_THRESHOLD = 0.60
    new_emotion = last_emotion
    confidence = torch.tensor([0.0])

    status_icon.set("🔄 Detecting...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face_pil = Image.fromarray(face)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            if time.time() - emotion_start_time > 10:
                with torch.no_grad():
                    output = model(face_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    new_emotion = emotion_labels[predicted.item()]
                    emotion_start_time = time.time()

                    print(f"🔍 Detected Emotion: {new_emotion} with Confidence: {confidence.item() * 100:.2f}%")
                    log_emotion(new_emotion, confidence.item())

                    if confidence.item() < CONFIDENCE_THRESHOLD:
                        new_emotion = "Emotion Uncertain"
                        print("❓ Low confidence in emotion detection.")

                    if new_emotion != last_emotion:
                        mixer.music.stop()
                        play_emotion_music(new_emotion)

                        if new_emotion == "Sad":
                            speak_affirmation()
                        elif new_emotion == "Surprise":
                            tell_joke()
                        elif new_emotion == "Happy":
                            tell_riddle()

                        last_emotion = new_emotion

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {new_emotion} ({confidence.item() * 100:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection & Music", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    mixer.music.stop()
    status_icon.set("⏹️ Stopped")

btn_start = ttk.Button(btn_frame, text="▶ Start Detection", command=start_detection)
btn_start.grid(row=0, column=0, padx=10)

btn_plot = ttk.Button(btn_frame, text="📊 Plot Emotion Trends", command=plot_emotion_trends)
btn_plot.grid(row=0, column=1, padx=10)

root.mainloop()
