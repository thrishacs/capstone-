from database.db_setup import SessionLocal
from database.models import EmotionLog

def get_emotion_logs():
    db = SessionLocal()
    logs = db.query(EmotionLog).order_by(EmotionLog.timestamp.desc()).all()
    db.close()
    return logs

def add_emotion_log(emotion):
    db = SessionLocal()
    new_log = EmotionLog(emotion=emotion)
    db.add(new_log)
    db.commit()
    db.close()
