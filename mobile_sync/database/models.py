from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class EmotionLog(Base):
    __tablename__ = 'emotion_logs'
    
    id = Column(Integer, primary_key=True)
    emotion = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
