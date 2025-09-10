from flask import Flask
from mobile_sync.database.db_setup import engine, Base
from mobile_sync.routes.emotion_logs import emotion_logs_bp  # import the blueprint

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Initialize Flask app
app = Flask(__name__)

# Register blueprints
app.register_blueprint(emotion_logs_bp, url_prefix="/api")  # all routes from emotion_logs.py will be under /api

@app.route("/")
def home():
    return {"message": "Emotion Detection API is running!"}

if __name__ == "__main__":
    app.run(debug=True)
