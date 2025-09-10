from sqlalchemy import Column, Integer, String, DateTime, Float
from .db_setup import Base
import datetime

class EmotionLog(Base):
    __tablename__ = "emotion_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    emotion = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
