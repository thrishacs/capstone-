# detection/spotify_player
import os
import random
from pygame import mixer
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pyttsx3

# --- Spotify Credentials ---
SPOTIPY_CLIENT_ID = "Dcef7601624b45d79b3a9ecf762844fd"
SPOTIPY_CLIENT_SECRET = "95d6027ac89749f3b81e30144f5be757"
SPOTIPY_REDIRECT_URI = "http://localhost:8888/callback"

# --- Assets ---
ASSETS_MUSIC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "music"))
os.makedirs(ASSETS_MUSIC, exist_ok=True)

# --- Mixer & TTS ---
mixer.init()
engine = pyttsx3.init()

# --- Spotify Setup ---
sp = None
SPOTIFY_ENABLED = False
try:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
        redirect_uri=SPOTIPY_REDIRECT_URI,
        scope="user-read-playback-state,user-modify-playback-state,user-read-private",
        open_browser=True
    ))
    SPOTIFY_ENABLED = True
    print("✅ Spotify initialized.")
except Exception as e:
    print(f"⚠️ Spotify init failed: {e}. Falling back to offline only.")

# --- Mood Resources ---
MOOD_TRACKS = {
    "Angry": "spotify:track:5FVd6KXrgO9B3JPmC8OPst",
    "Sad": "spotify:track:7qEHsqek33rTcFNT9PFqLf",
    "Fear": "spotify:track:0GONea6G2XdnHWjNZd6zt3",
    "Surprise": "spotify:track:6WrI0LAC5M1Rw2MnX2ZvEg",
    "Happy": "spotify:track:3KkXRkHbMCARz0aVfEt68P",
    "Neutral": None
}

AFFIRMATIONS = {
    "Angry": ["Take a deep breath. Calmness is within you.", "You are in control of your emotions."],
    "Sad": ["This too shall pass.", "You are stronger than you think."],
    "Fear": ["Courage is not absence of fear, it's action in spite of it.", "You can overcome anything."],
    "Surprise": ["Embrace the unexpected.", "New experiences bring growth."],
    "Happy": ["Spread your happiness!", "Your smile lights up the room."],
    "Neutral": ["Stay mindful and balanced.", "Enjoy the present moment."]
}

JOKES = {
    "Angry": ["Why don’t scientists trust atoms? Because they make up everything!"],
    "Sad": ["Why did the computer go to therapy? It had too many bugs."],
    "Fear": ["Why don’t ghosts like rain? It dampens their spirits."],
    "Surprise": ["Why did the scarecrow win an award? Because he was outstanding in his field!"],
    "Happy": ["Why did the bicycle fall over? Because it was two-tired!"],
    "Neutral": ["I would tell you a joke about UDP, but you might not get it."]
}

RIDDLES = {
    "Angry": ["I am always hungry, I must always be fed. The finger I touch will soon turn red. What am I? Answer: Fire"],
    "Sad": ["I speak without a mouth and hear without ears. What am I? Answer: Echo"],
    "Fear": ["The more of me you take, the more you leave behind. What am I? Answer: Footsteps"],
    "Surprise": ["I’m tall when I’m young, and I’m short when I’m old. What am I? Answer: Candle"],
    "Happy": ["What has keys but can't open locks? Answer: Piano"],
    "Neutral": ["What can travel around the world while staying in a corner? Answer: Stamp"]
}

def user_has_premium():
    if not SPOTIFY_ENABLED or not sp:
        return False
    try:
        me = sp.current_user()
        return me.get("product", "").lower() == "premium"
    except Exception:
        return False

def speak_text(text: str):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ TTS error: {e}")

def play_music_for_emotion(emotion: str) -> bool:
    """
    Play Spotify track or offline mp3 for the given emotion,
    also speak affirmations, jokes, and riddles.
    """
    emotion = emotion or "Neutral"
    played = False

    # --- Spotify Playback ---
    if SPOTIFY_ENABLED and sp and user_has_premium():
        track = MOOD_TRACKS.get(emotion)
        if track:
            try:
                devices = sp.devices()
                if devices and devices.get("devices"):
                    sp.start_playback(uris=[track])
                    print(f"🎵 Playing Spotify track for {emotion}")
                    played = True
                else:
                    print("⚠️ No active Spotify device found.")
            except Exception as e:
                print(f"⚠️ Spotify playback failed: {e}")

    # --- Offline Fallback ---
    if not played:
        try:
            files = [f for f in os.listdir(ASSETS_MUSIC) if f.lower().endswith(".mp3")]
            candidates = [f for f in files if emotion.lower() in f.lower()]
            if not candidates:
                candidates = files
            if candidates:
                choice = random.choice(candidates)
                path = os.path.join(ASSETS_MUSIC, choice)
                mixer.music.load(path)
                mixer.music.play()
                print(f"🎧 Playing offline {choice} for {emotion}")
                played = True
        except Exception as e:
            print(f"❌ Offline playback error: {e}")

    # --- Speak Affirmation, Joke, and Riddle ---
    affirmation = random.choice(AFFIRMATIONS.get(emotion, ["Stay positive!"]))
    joke = random.choice(JOKES.get(emotion, ["Have a great day!"]))
    riddle = random.choice(RIDDLES.get(emotion, ["Think smart!"]))

    print(f"💬 Affirmation: {affirmation}")
    print(f"😂 Joke: {joke}")
    print(f"🧩 Riddle: {riddle}")

    speak_text(affirmation)
    speak_text(joke)
    speak_text(riddle)

    return played

def stop_music():
    try:
        mixer.music.stop()
    except Exception:
        pass
