# mobile_sync/api_server.py
from flask import Flask
from routes.emotion_logs import emotion_logs_bp
from database.db_setup import Base, engine

# Create DB tables
Base.metadata.create_all(bind=engine)

app = Flask(__name__)
app.register_blueprint(emotion_logs_bp, url_prefix="/logs")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
