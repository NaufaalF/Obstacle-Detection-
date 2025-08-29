# FastDepth Integration for app.py
# Replace the depth estimation section in your existing app.py with this

import base64
import threading
import numpy as np
import cv2
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

# FastDepth import
from depth_estimation_fastdepth import FastDepthEstimator  # NEW

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -------------------------
# Existing configuration (unchanged)
# -------------------------
KNOWN_WIDTHS = {
    0: 175,  # Mobil (class 0)
    1: 70,   # Motor (class 1) 
    2: 50,   # Orang (class 2)
    3: 250   # Truk (class 3)
}

DISTANCE_THRESHOLDS = {
    0: 300,  # Mobil
    1: 300,  # Motor
    2: 50,   # Orang  
    3: 400   # Truk
}

CLASS_NAMES = ["Mobil", "Motor", "Orang", "Truk"]

# -------------------------
# Model initialization with FastDepth
# -------------------------
# Existing YOLO model
model = YOLO("./models/suBest3_float16.tflite")

# FastDepth initialization with fallback
try:
    # Try FastDepth equivalent first - using 320x320 for consistency
    if os.path.exists("./models/mobilenet_depth.tflite"):
        depth_estimator = FastDepthEstimator("./models/mobilenet_depth.tflite", input_size=(320, 320))
        print("✅ FastDepth-equivalent (MobileNet) loaded with 320x320")
    elif os.path.exists("./models/fastdepth.tflite"):
        depth_estimator = FastDepthEstimator("./models/fastdepth.tflite", input_size=(320, 320))
        print("✅ Converted FastDepth loaded with 320x320")
    else:
        raise FileNotFoundError("No FastDepth model found")
    
    USE_DEPTH_ESTIMATION = True
    print("🎯 FastDepth depth estimation enabled (320x320)")
    
except Exception as e:
    print(f"⚠️  FastDepth disabled: {e}")
    print("💡 Falling back to focal length method")
    USE_DEPTH_ESTIMATION = False
    FOCAL_LENGTH = 888  # Fallback to original method

processing_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("connect")
def on_connect():
    print("Client connected")

@socketio.on("disconnect") 
def on_disconnect():
    print("Client disconnected")

@socketio.on("frame")
def handle_frame(data):
    if processing_lock.locked():
        return

    with processing_lock:
        data_url = data.get("image", "") if isinstance(data, dict) else data
        if not data_url:
            return

        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url

        try:
            jpg_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(jpg_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return

            img = cv2.resize(img, (320, 320))

            # YOLO inference (unchanged)
            results = model.predict(img, imgsz=320, conf=0.25)

            # FastDepth estimation
            depth_map = None
            if USE_DEPTH_ESTIMATION:
                depth_map = depth_estimator.estimate_depth(img)

            # Process detections 
            det = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        det.append([x1, y1, x2, y2, conf, cls])

            det = np.array(det) if len(det) else np.empty((0, 6))
            h, w = img.shape[:2]
            detections = []

            for x1, y1, x2, y2, conf, cls in det:
                cls = int(cls)
                alert = False
                
                # Use FastDepth estimation if available
                if USE_DEPTH_ESTIMATION and depth_map is not None:
                    # Get depth from FastDepth
                    distance = depth_estimator.get_object_depth(depth_map, [x1, y1, x2, y2])
                    distance_cm = distance * 100  # Convert meters to cm
                    
                    # FastDepth typically gives good results for 0.1-10m range
                    if distance_cm < 10:  # Less than 10cm seems wrong
                        distance_cm = None  # Fallback to focal length
                else:
                    # Fallback to original focal length method
                    perceived_width = float(x2 - x1)
                    distance_cm = None
                    if cls in KNOWN_WIDTHS and perceived_width > 0:
                        distance_cm = (KNOWN_WIDTHS[cls] * FOCAL_LENGTH) / perceived_width

                # Check alert condition
                if distance_cm and distance_cm <= DISTANCE_THRESHOLDS.get(cls, 999999):
                    alert = True

                class_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
                detections.append({
                    "class_id": cls,
                    "class_name": class_name, 
                    "conf": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "distance_cm": float(distance_cm) if distance_cm is not None else None,
                    "alert": alert,
                    "method": "fastdepth" if USE_DEPTH_ESTIMATION else "focal_length"  # DEBUG info
                })

            emit("detections", {"detections": detections, "width": w, "height": h})
            
        except Exception as e:
            print("Error handling frame:", repr(e))
            return

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)