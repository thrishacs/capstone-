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
import webbrowser

# ✅ Spotify Credentials
SPOTIPY_CLIENT_ID = 'dcef7601624b45d79b3a9ecf762844fd'
SPOTIPY_CLIENT_SECRET = '95d6027ac89749f3b81e30144f5be757'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

# ✅ Spotify Setup
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-read-playback-state,user-modify-playback-state",
    open_browser=True
))

# ✅ Model Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 7)
)
model.load_state_dict(torch.load("emotion_model_resnet50.pth", map_location=device))
model.to(device)
model.eval()

# ✅ Labels & Spotify Track URIs (one song per emotion)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_tracks = {
    "Angry": "spotify:track:5FVd6KXrgO9B3JPmC8OPst",     # Lose Yourself
    "Disgust": "spotify:track:3AJwUDP919kvQ9QcozQPxg",   # Fix You
    "Fear": "spotify:track:0GONea6G2XdnHWjNZd6zt3",      # Numb
    "Happy": "spotify:track:3KkXRkHbMCARz0aVfEt68P",      # Sunflower
    "Sad": "spotify:track:7qEHsqek33rTcFNT9PFqLf",        # Someone You Loved
    "Surprise": "spotify:track:6WrI0LAC5M1Rw2MnX2ZvEg",   # Happy
}

# ✅ Audio Setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

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

# ✅ Spotify Play Function
def play_spotify_track(track_uri):
    try:
        sp.start_playback(uris=[track_uri])
        print(f"🎵 Playing Spotify Track: {track_uri}")
    except Exception as e:
        print(f"❌ Error playing track: {e}")
        play_spotify_fallback(track_uri)

# ✅ Fallback to web browser if Premium required
def play_spotify_fallback(track_uri):
    track_url = track_uri.replace("spotify:track:", "https://open.spotify.com/track/")
    print(f"🌐 Opening in browser: {track_url}")
    webbrowser.open(track_url)

# ✅ Image Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Start Webcam
cap = cv2.VideoCapture(0)
last_emotion = "Neutral"
emotion_start_time = time.time()
CONFIDENCE_THRESHOLD = 0.60
new_emotion = last_emotion
confidence = torch.tensor([0.0])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
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

                if confidence.item() < CONFIDENCE_THRESHOLD:
                    new_emotion = "Emotion Uncertain"
                    print("❓ Low confidence in emotion detection.")

                if new_emotion != last_emotion:
                    if new_emotion in emotion_tracks:
                        play_spotify_track(emotion_tracks[new_emotion])
                    if new_emotion == "Sad":
                        speak_affirmation()
                    elif new_emotion == "Surprise":
                        tell_joke()
                    elif new_emotion == "Happy":
                        tell_riddle()
                    last_emotion = new_emotion

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Emotion: {new_emotion} ({confidence.item() * 100:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Car Emotion Detection & Music", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
