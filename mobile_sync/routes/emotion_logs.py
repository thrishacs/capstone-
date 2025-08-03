from flask import Blueprint, jsonify, request
from services.log_services import get_emotion_logs, add_emotion_log

emotion_bp = Blueprint('emotion_bp', __name__)

@emotion_bp.route('/api/logs', methods=['GET'])
def fetch_logs():
    logs = get_emotion_logs()
    return jsonify([{
        'emotion': log.emotion,
        'timestamp': log.timestamp.isoformat()
    } for log in logs])

@emotion_bp.route('/api/logs', methods=['POST'])
def create_log():
    data = request.json
    emotion = data.get('emotion')
    if emotion:
        add_emotion_log(emotion)
        return jsonify({'message': 'Log added'}), 201
    return jsonify({'error': 'Missing emotion'}), 400
