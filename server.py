from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
from datetime import datetime

# ── App setup ────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Config ───────────────────────────────────────────────────────
CLASSES = ['crazing', 'inclusion', 'patches',
           'pitted_surface', 'rolled-in_scale', 'scratches']

SEVERITY_MAP = {
    'crazing': 'Minor',
    'inclusion': 'Major',
    'patches': 'Minor',
    'pitted_surface': 'Critical',
    'rolled-in_scale': 'Major',
    'scratches': 'Major',
}

prediction_history = []

# ── Load LIGHTWEIGHT MODEL (IMPORTANT) ────────────────────────────
print("🚀 Starting server...")

from ultralytics import YOLO

# Use SMALL model (Render-friendly)
yolo_model = YOLO("yolov8n.pt")

print("✅ Model loaded successfully")

# ════════════════════════════════════════════════════════════════
# ROUTE: HOME
# ════════════════════════════════════════════════════════════════
@app.route("/")
def home():
    return "API Running"

# ════════════════════════════════════════════════════════════════
# ROUTE: PREDICT
# ════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    file_bytes = file.read()

    np_arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Run inference
    results = yolo_model.predict(img, conf=0.25, verbose=False)

    detections = []

    for r in results:
        for box in (r.boxes or []):
            cls_id = int(box.cls)
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"

            detections.append({
                "class": cls_name,
                "confidence": float(box.conf),
                "box": box.xyxy[0].tolist(),
                "severity": SEVERITY_MAP.get(cls_name, "Unknown")
            })

    return jsonify({
        "status": "FAIL" if detections else "PASS",
        "num_defects": len(detections),
        "detections": detections
    })

# ════════════════════════════════════════════════════════════════
# ROUTE: HEALTH
# ════════════════════════════════════════════════════════════════
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "yolov8n"
    })

# ════════════════════════════════════════════════════════════════
# MAIN (CRITICAL FOR RENDER)
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
