from flask import Blueprint, jsonify, request
from services.log_services import save_log, get_logs

emotion_logs_bp = Blueprint("emotion_logs", __name__)

# POST → Save log
@emotion_logs_bp.route("/", methods=["POST"])
def add_log():
    data = request.get_json()
    emotion = data.get("emotion")
    timestamp = data.get("timestamp")

    if not emotion or not timestamp:
        return jsonify({"error": "Missing fields"}), 400

    save_log(emotion, timestamp)
    return jsonify({"message": "Log saved"}), 201

# GET → Retrieve logs
@emotion_logs_bp.route("/", methods=["GET"])
def fetch_logs():
    logs = get_logs()
    return jsonify(logs), 200
