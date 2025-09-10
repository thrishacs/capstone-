from flask import Blueprint, jsonify, request
from mobile_sync.services.log_services import save_log, get_logs

emotion_logs_bp = Blueprint("emotion_logs", __name__)

@emotion_logs_bp.route("/logs/", methods=["GET"])
def fetch_logs():
    logs = get_logs()
    return jsonify(logs)

@emotion_logs_bp.route("/logs/", methods=["POST"])
def add_log():
    data = request.json
    emotion = data.get("emotion")
    confidence = data.get("confidence")
    save_log(emotion, confidence)
    return jsonify({"status": "success"}), 201
