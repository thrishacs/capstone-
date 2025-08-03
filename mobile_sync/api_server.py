from flask import Flask
from routes.emotion_logs import emotion_bp
from database.db_setup import init_db

app = Flask(__name__)
app.register_blueprint(emotion_bp)

# Directly initialize the DB before the server starts
init_db()

if __name__ == '__main__':
    app.run(port=5001, debug=True)
