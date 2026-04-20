"""
╔══════════════════════════════════════════════════════════════════╗
║          STEEL DEFECT DETECTION — FLASK API SERVER              ║
║          server.py  |  Run: python server.py                    ║
╠══════════════════════════════════════════════════════════════════╣
║  INSTALL:                                                        ║
║    pip install flask flask-cors ultralytics opencv-python        ║
║                numpy pillow                                      ║
║                                                                  ║
║  PLACE NEXT TO THIS FILE:                                        ║
║    yolov8_best.pt   ← your trained YOLOv8 weights               ║
║                                                                  ║
║  ENDPOINTS:                                                      ║
║    POST /predict   → run YOLOv8 on uploaded image               ║
║    GET  /health    → server + model status check                 ║
║    GET  /classes   → list of detectable defect classes           ║
║    GET  /history   → last 100 predictions made via API           ║
║    DELETE /history → clear prediction history                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os
import time
from datetime import datetime

# ── App setup ────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow cross-origin requests from Streamlit client

# ── Config ───────────────────────────────────────────────────────
MODEL_PATH   = "yolov8_best.pt"
CLASSES      = ['crazing', 'inclusion', 'patches',
                'pitted_surface', 'rolled-in_scale', 'scratches']
SEVERITY_MAP = {
    'crazing':          'Minor',
    'inclusion':        'Major',
    'patches':          'Minor',
    'pitted_surface':   'Critical',
    'rolled-in_scale':  'Major',
    'scratches':        'Major',
}

# ── In-memory prediction log (persists while server is running) ──
prediction_history = []

# ── Load YOLOv8 model ONCE at startup ────────────────────────────
print("=" * 55)
print("  STEEL DEFECT DETECTION — FLASK API SERVER")
print("=" * 55)

yolo_model  = None
model_error = None

if not os.path.exists(MODEL_PATH):
    model_error = (f"Model file not found: '{MODEL_PATH}'. "
                   f"Place yolov8_best.pt next to server.py")
    print(f"  ⚠️  {model_error}")
else:
    try:
        from ultralytics import YOLO
        print(f"  Loading YOLOv8 model: {MODEL_PATH} ...")
        yolo_model = YOLO(MODEL_PATH)
        # Warm-up inference so first real request is fast
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        yolo_model.predict(dummy, verbose=False)
        print(f"  ✅ Model loaded and warmed up successfully")
    except Exception as e:
        model_error = str(e)
        print(f"  ❌ Model load failed: {model_error}")

print(f"  Listening on: http://127.0.0.1:5000")
print("=" * 55)


# ════════════════════════════════════════════════════════════════
# HELPER — draw boxes on image
# ════════════════════════════════════════════════════════════════
COLORS = {
    'crazing':         (231,  76,  60),
    'inclusion':       ( 46, 204, 113),
    'patches':         ( 52, 152, 219),
    'pitted_surface':  (243, 156,  18),
    'rolled-in_scale': (155,  89, 182),
    'scratches':       ( 26, 188, 156),
}

def draw_boxes(img_bgr, detections):
    """Draw coloured bounding boxes and labels onto a copy of img_bgr."""
    annotated = img_bgr.copy()
    for d in detections:
        cls_name = d['class']
        conf     = d['confidence']
        x1, y1, x2, y2 = d['box']
        color = COLORS.get(cls_name, (255, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


# ════════════════════════════════════════════════════════════════
# ROUTE 1 — POST /predict
# Accepts:  multipart/form-data  { "image": <file> }
#           optional form fields: conf (float), iou (float)
# Returns:  JSON list of detections
# ════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    # ── Model availability check ──────────────────────────────
    if yolo_model is None:
        return jsonify({
            "error": f"Model not loaded: {model_error}",
            "detections": []
        }), 503

    # ── Image validation ──────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send field name: 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename received"}), 400

    # ── Decode image ──────────────────────────────────────────
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "Could not decode image. Send a valid JPG/PNG."}), 400

    # ── Inference parameters ──────────────────────────────────
    try:
        conf_thresh = float(request.form.get("conf", 0.25))
        iou_thresh  = float(request.form.get("iou",  0.45))
    except ValueError:
        conf_thresh, iou_thresh = 0.25, 0.45

    conf_thresh = max(0.01, min(0.99, conf_thresh))
    iou_thresh  = max(0.01, min(0.99, iou_thresh))

    # ── Run YOLOv8 inference ──────────────────────────────────
    t0 = time.time()
    results = yolo_model.predict(
        img_bgr, conf=conf_thresh, iou=iou_thresh,
        imgsz=640, verbose=False
    )
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    # ── Build detections list ─────────────────────────────────
    detections = []
    for r in results:
        for box in (r.boxes or []):
            cls_id   = int(box.cls)
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
            conf     = round(float(box.conf), 4)
            xyxy     = box.xyxy[0].cpu().numpy().astype(int).tolist()
            detections.append({
                "class":      cls_name,
                "confidence": conf,
                "box":        xyxy,          # [x1, y1, x2, y2]
                "severity":   SEVERITY_MAP.get(cls_name, "Unknown"),
            })

    # ── Log this prediction ───────────────────────────────────
    log_entry = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename":     file.filename,
        "image_size":   f"{img_bgr.shape[1]}x{img_bgr.shape[0]}",
        "conf_thresh":  conf_thresh,
        "iou_thresh":   iou_thresh,
        "num_defects":  len(detections),
        "classes_found": list({d["class"] for d in detections}),
        "inference_ms": elapsed_ms,
        "status":       "FAIL" if detections else "PASS",
    }
    prediction_history.append(log_entry)
    if len(prediction_history) > 100:
        prediction_history.pop(0)  # keep last 100

    # ── Build response ────────────────────────────────────────
    response = {
        "status":        "FAIL" if detections else "PASS",
        "num_defects":   len(detections),
        "inference_ms":  elapsed_ms,
        "conf_thresh":   conf_thresh,
        "iou_thresh":    iou_thresh,
        "detections":    detections,       # main payload for Streamlit
        "server_time":   datetime.now().strftime("%H:%M:%S"),
    }
    return jsonify(response), 200


# ════════════════════════════════════════════════════════════════
# ROUTE 2 — GET /health
# Returns server status, model status, uptime info
# ════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "model_loaded":   yolo_model is not None,
        "model_path":     MODEL_PATH,
        "model_error":    model_error,
        "classes":        CLASSES,
        "total_requests": len(prediction_history),
        "server_time":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }), 200


# ════════════════════════════════════════════════════════════════
# ROUTE 3 — GET /classes
# Returns list of detectable defect classes and their severity
# ════════════════════════════════════════════════════════════════
@app.route("/classes", methods=["GET"])
def classes():
    return jsonify({
        "classes": [
            {"id": i, "name": c, "severity": SEVERITY_MAP.get(c, "Unknown")}
            for i, c in enumerate(CLASSES)
        ]
    }), 200


# ════════════════════════════════════════════════════════════════
# ROUTE 4 — GET /history
# Returns last N prediction records (default 20, max 100)
# ════════════════════════════════════════════════════════════════
@app.route("/history", methods=["GET"])
def history():
    try:
        limit = int(request.args.get("limit", 20))
        limit = max(1, min(100, limit))
    except ValueError:
        limit = 20
    records = prediction_history[-limit:][::-1]   # newest first
    return jsonify({
        "count":   len(records),
        "records": records,
    }), 200


# ════════════════════════════════════════════════════════════════
# ROUTE 5 — DELETE /history
# Clears the in-memory prediction log
# ════════════════════════════════════════════════════════════════
@app.route("/history", methods=["DELETE"])
def clear_history():
    cleared = len(prediction_history)
    prediction_history.clear()
    return jsonify({"cleared": cleared, "message": "History cleared"}), 200


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
