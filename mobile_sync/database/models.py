# mobile_sync/database/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database.db_setup import Base

class EmotionLog(Base):
    __tablename__ = "emotion_logs"

    id = Column(Integer, primary_key=True, index=True)
    emotion = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
