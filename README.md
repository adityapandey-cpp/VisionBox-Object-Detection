# VisionBox — Object Detection (TensorFlow + OpenCV + Flask)

Detect 80 COCO classes in **images, video files and a live webcam**, with colour-coded boxes,
confidence/max-object sliders, class filters, per-class counts and PNG/JSON/CSV export.

## Two inference paths
1. **Browser (default, zero-setup)** — TensorFlow.js SSD/COCO models run locally in the page.
   Great for the webcam and video tabs (no upload latency, privacy friendly).
2. **Server (production)** — the Flask API in `backend/` runs real TensorFlow SavedModels from
   TF-Hub (SSD MobileNet V2, SSD MobileNet V2 FPNLite 320, Faster R-CNN ResNet50) and draws boxes
   with OpenCV.

## Frontend
```bash
npm install
npm run dev     # http://localhost:5173
npm run build   # static bundle in dist/
```

## Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py                                    # dev, http://localhost:8000
gunicorn -w 2 -t 180 -b 0.0.0.0:8000 app:app     # production
```
Serve the frontend from Flask by copying `dist/` into `backend/static/`.

Docker: `docker build -t visionbox backend && docker run -p 8000:8000 visionbox`
(GPU: swap the base image for `tensorflow/tensorflow:2.16.1-gpu` and run with `--gpus all`.)

## API
- `GET /api/models` → `[{key, name, blurb}]`
- `POST /api/detect` (multipart) fields: `image` (jpg/png/webp, ≤16 MB), `model`
  (`fast|balanced|accurate`), `threshold` (0.1–0.95), `max_det`, `exclude` (comma list of labels),
  `annotate` (`true|false`)

```json
{
  "detections": [{"label":"person","class_id":1,"score":0.93,"box":[0.1,0.2,0.8,0.5]}],
  "inference_ms": 142.3, "width": 1280, "height": 720,
  "annotated": "data:image/png;base64,..."
}
```
Boxes are normalized `[ymin, xmin, ymax, xmax]`.

## Green screen with DroidCam / virtual webcams — why, and the built-in fix

A green picture never comes from the detector: the virtual-camera **driver** is handing the browser an
empty/incorrectly-laid-out YUV (NV12/I420) buffer. Usual causes:

| Cause | What the app now does |
|---|---|
| Resolution not supported (we asked 1280×720, DroidCam free is 640×480) | Constraint **ladder**: 640×480 → 1280×720 → 320×240 → plain `video:true`, retrying automatically |
| `facingMode: environment` requested (virtual cams don't implement it) | Removed — only `deviceId` is used |
| Phone app not connected yet, so the driver emits a blank/green plane | Frames are **inspected** (48×36 probe → mean + std-dev per channel); green/blank/frozen feeds are detected and the stream is re-opened with the next profile |
| Wrong device picked ("DroidCam Audio", or the laptop cam) | **Camera dropdown** with auto-selection of the virtual cam (📱 marker) + Refresh |
| Stale buffer after a previous stream | `srcObject = null` → re-attach, plus a warm-up that waits for 8 real frames (`requestVideoFrameCallback`) before inference |
| Chrome hardware-decode bug | Warning banner with step-by-step fixes ("Re-sync camera" / "How to fix") |

Manual checklist shown in the UI: start DroidCam on the phone **first**, set the client to 640×480, choose
the *DroidCam Source 3 / Video* device, close other apps using the camera, disable Chrome's graphics
acceleration (`chrome://settings/system`) and relaunch, or try Firefox / USB mode.
Green frames are also skipped by the inference loop, so no CPU is wasted while the feed is bad.

## Notes
- Models load **lazily and once**, guarded by a lock (double-checked locking).
- Validation: 16 MB cap (HTTP 413), MIME allow-list (415), corrupt-image and model-load errors.
- Custom SavedModel: set `CUSTOM_MODEL_DIR=/path/to/saved_model` and post `model=custom`.
- Accessibility: keyboard-reachable controls, visible focus rings, ARIA tabs/pressed states,
  dark theme, responsive down to mobile.
- Video/webcam frames get **centroid tracking IDs** (`#3 person 91%`) for stable identities.
