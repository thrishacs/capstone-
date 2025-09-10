from mobile_sync.database.db_setup import SessionLocal
from mobile_sync.database.models import EmotionLog
from config import CSV_LOG_PATH
import csv
import os

def get_latest_status():
    db = SessionLocal()
    try:
        latest = db.query(EmotionLog).order_by(EmotionLog.timestamp.desc()).first()
        if latest:  # ✅ From DB
            return {
                "timestamp": latest.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "emotion": latest.emotion,
                "confidence": f"{latest.confidence:.2f}%"
            }
        elif os.path.exists(CSV_LOG_PATH):  # ✅ Fallback to CSV
            with open(CSV_LOG_PATH, "r") as f:
                reader = list(csv.reader(f))
                if reader:
                    last_row = reader[-1]
                    return {
                        "timestamp": last_row[0],
                        "emotion": last_row[1],
                        "confidence": last_row[2]
                    }
        return {"status": "No data available"}
    finally:
        db.close()
