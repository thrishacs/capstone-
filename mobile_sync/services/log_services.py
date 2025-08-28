from database.db_setup import SessionLocal
from database.models import EmotionLog

def save_log(emotion, confidence):
    db = SessionLocal()
    new_log = EmotionLog(emotion=emotion, confidence=confidence)
    db.add(new_log)
    db.commit()
    db.refresh(new_log)
    db.close()
    return new_log

def get_logs():
    db = SessionLocal()
    logs = db.query(EmotionLog).all()
    db.close()
    return logs
