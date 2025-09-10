from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Base directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Store DB inside mobile_sync/database/
DB_PATH = os.path.join(BASE_DIR, "emotion_logs.db")

# Full SQLite URL
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# Engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Session and Base
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
