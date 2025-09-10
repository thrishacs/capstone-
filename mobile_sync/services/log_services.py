import os
import csv
from mobile_sync.database.db_setup import SessionLocal
from mobile_sync.database.models import EmotionLog

CSV_LOG_PATH = r"D:\CAPSTONE\logs\emotion_log.csv"

def save_log(emotion, confidence):
    """Save a new log entry to the database."""
    session = SessionLocal()
    try:
        log = EmotionLog(emotion=emotion, confidence=confidence)
        session.add(log)
        session.commit()
    finally:
        session.close()

def get_logs():
    """Fetch logs from both the database and CSV file as dicts."""
    logs = []

    # 1. Load from DB
    session = SessionLocal()
    try:
        db_logs = session.query(EmotionLog).all()
        for log in db_logs:
            logs.append({
                "timestamp": log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "emotion": log.emotion,
                "confidence": f"{log.confidence:.2f}%"
            })
    finally:
        session.close()

    # 2. Load from CSV
    if os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 3:  # timestamp, emotion, confidence
                    logs.append({
                        "timestamp": row[0],
                        "emotion": row[1],
                        "confidence": row[2]
                    })

    return logs
