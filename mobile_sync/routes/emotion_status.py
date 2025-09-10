from flask import Blueprint, jsonify
from mobile_sync.services.status_service import get_latest_status

status_bp = Blueprint("status", __name__)

@status_bp.route("/status", methods=["GET"])
def fetch_status():
    status = get_latest_status()
    return jsonify(status)
