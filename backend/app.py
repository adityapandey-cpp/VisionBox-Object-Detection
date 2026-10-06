"""
VisionBox — Flask + TensorFlow (Hub) + OpenCV object-detection API.

Endpoints
---------
GET  /api/models  -> available pre-trained COCO models
POST /api/detect  -> multipart form: image, model, threshold, max_det, exclude, annotate

Run:  python app.py           (dev)
      gunicorn -w 2 -t 120 -b 0.0.0.0:8000 app:app      (prod)
"""
import base64
import io
import os
import threading
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ------------------------------------------------------------------ config
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload cap
ALLOWED = {"image/jpeg", "image/png", "image/webp"}

MODELS = {
    "fast": {
        "name": "SSD MobileNet V2",
        "url": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
        "blurb": "Fastest – great for webcam / video",
    },
    "balanced": {
        "name": "SSD MobileNet V2 FPNLite 320",
        "url": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1",
        "blurb": "Balanced speed / accuracy",
    },
    "accurate": {
        "name": "Faster R-CNN ResNet50 V1 640",
        "url": "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1",
        "blurb": "Most accurate – slower",
    },
}

# COCO 90-index label map (only the 80 used classes are populated).
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train",
    8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign",
    14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis",
    36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple",
    54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app)

# ------------------------------------------------- lazy, thread-safe cache
_cache: dict = {}
_lock = threading.Lock()


def get_model(key: str):
    """Load a TF-Hub detector once and reuse it (double-checked locking)."""
    if key in _cache:
        return _cache[key]
    with _lock:
        if key not in _cache:                      # another thread may have won
            custom = os.environ.get("CUSTOM_MODEL_DIR")
            if key == "custom" and custom:
                _cache[key] = tf.saved_model.load(custom)
            else:
                _cache[key] = hub.load(MODELS[key]["url"])
    return _cache[key]


def color_for(label: str):
    """Deterministic BGR colour per class."""
    h = 0
    for ch in label:
        h = (h * 31 + ord(ch)) % 9973
    hsv = np.uint8([[[h % 180, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def annotate(bgr, dets):
    """Draw colour-coded boxes + 'label confidence' with OpenCV."""
    h, w = bgr.shape[:2]
    for d in dets:
        ymin, xmin, ymax, xmax = d["box"]
        p1 = (int(xmin * w), int(ymin * h))
        p2 = (int(xmax * w), int(ymax * h))
        c = color_for(d["label"])
        cv2.rectangle(bgr, p1, p2, c, max(2, h // 300))
        text = f'{d["label"]} {d["score"] * 100:.0f}%'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(p1[1], th + 6)
        cv2.rectangle(bgr, (p1[0], ty - th - 6), (p1[0] + tw + 8, ty), c, -1)
        cv2.putText(bgr, text, (p1[0] + 4, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 16, 24), 2)
    return bgr


# ------------------------------------------------------------------ routes
@app.get("/api/models")
def api_models():
    return jsonify([{"key": k, **{kk: vv for kk, vv in v.items() if kk != "url"}} for k, v in MODELS.items()])


@app.post("/api/detect")
def api_detect():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No file uploaded (field name must be 'image')."), 400
    if file.mimetype not in ALLOWED:
        return jsonify(error=f"Unsupported file type: {file.mimetype}"), 415

    key = request.form.get("model", "fast")
    if key not in MODELS and key != "custom":
        return jsonify(error=f"Unknown model '{key}'."), 400
    try:
        threshold = min(max(float(request.form.get("threshold", 0.5)), 0.1), 0.95)
        max_det = min(max(int(request.form.get("max_det", 20)), 1), 100)
    except ValueError:
        return jsonify(error="threshold/max_det must be numeric."), 400
    exclude = {s.strip() for s in request.form.get("exclude", "").split(",") if s.strip()}
    want_annotated = request.form.get("annotate", "true").lower() == "true"

    # --- decode with OpenCV -------------------------------------------------
    buf = np.frombuffer(file.read(), np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="Could not decode the image — the file may be corrupt."), 400
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # --- inference ----------------------------------------------------------
    try:
        detector = get_model(key)
    except Exception as exc:                                    # network / hub errors
        return jsonify(error=f"Model could not be loaded: {exc}"), 503

    tensor = tf.convert_to_tensor(rgb[tf.newaxis, ...], dtype=tf.uint8)
    t0 = time.perf_counter()
    out = detector(tensor)
    inference_ms = (time.perf_counter() - t0) * 1000

    boxes = out["detection_boxes"][0].numpy()
    scores = out["detection_scores"][0].numpy()
    classes = out["detection_classes"][0].numpy().astype(int)

    dets = []
    for box, score, cid in zip(boxes, scores, classes):
        if score < threshold:
            continue
        label = COCO_LABELS.get(int(cid), f"class_{int(cid)}")
        if label in exclude:
            continue
        dets.append({"label": label, "class_id": int(cid), "score": float(score),
                     "box": [float(v) for v in box]})
        if len(dets) >= max_det:
            break

    payload = {"detections": dets, "inference_ms": round(inference_ms, 2), "width": w, "height": h}
    if want_annotated:
        ok, png = cv2.imencode(".png", annotate(bgr.copy(), dets))
        if ok:
            payload["annotated"] = "data:image/png;base64," + base64.b64encode(png).decode()
    return jsonify(payload)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="File too large — the limit is 16 MB."), 413


@app.get("/")
def index():
    """Serve the built frontend (copy dist/ into backend/static/)."""
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=True)
