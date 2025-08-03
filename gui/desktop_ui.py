import tkinter as tk
import subprocess
import os
import sys
import threading
import signal

# GUI Window
root = tk.Tk()
root.title("Emotion Detection Engine")
root.geometry("400x250")
root.configure(bg="#f4f4f4")

# Global process reference
engine_process = None

# Status label
status_label = tk.Label(root, text="Status: Idle", fg="blue", bg="#f4f4f4", font=("Arial", 12))
status_label.pack(pady=10)

# Emotion preview label
emotion_label = tk.Label(root, text="Last Emotion: None", fg="black", bg="#f4f4f4", font=("Arial", 12))
emotion_label.pack(pady=10)

# Read last emotion from log
log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs", "emotion_log.csv"))
def update_emotion_preview():
    try:
        if os.path.exists(log_path):
            with open(log_path, 'r') as file:
                lines = file.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip()
                    timestamp, emotion = last_line.split(',')
                    emotion_label.config(text=f"Last Emotion: {emotion}")
    except Exception as e:
        emotion_label.config(text="Last Emotion: Error")

    root.after(3000, update_emotion_preview)

# Start Detection

def start_engine():
    global engine_process
    if engine_process is None:
        status_label.config(text="Status: Running", fg="green")
        engine_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "detection", "emotion_engine.py"))
        engine_process = subprocess.Popen([sys.executable, engine_path])

# Stop Detection

def stop_engine():
    global engine_process
    if engine_process:
        engine_process.terminate()
        engine_process.wait()
        engine_process = None
        status_label.config(text="Status: Stopped", fg="red")

# Exit button

def exit_app():
    stop_engine()
    root.destroy()

# Buttons
start_btn = tk.Button(root, text="Start Detection", bg="green", fg="white", font=("Arial", 12), command=start_engine)
start_btn.pack(pady=5)

stop_btn = tk.Button(root, text="Stop Detection", bg="orange", fg="white", font=("Arial", 12), command=stop_engine)
stop_btn.pack(pady=5)

exit_btn = tk.Button(root, text="Exit", bg="red", fg="white", font=("Arial", 12), command=exit_app)
exit_btn.pack(pady=5)

# Periodically update emotion
update_emotion_preview()

# Launch GUI
root.mainloop()
