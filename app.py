import base64
import threading
import numpy as np
import cv2
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
# gunakan threading agar tidak butuh eventlet/gevent
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -------------------------
# Kalibrasi Kamera (ganti dengan milik Anda)
# -------------------------
FOCAL_LENGTH = 888  # hasil kalibrasi Anda

# -------------------------
# Lebar objek nyata (cm)
# -------------------------
KNOWN_WIDTHS = {
    0: 175,  # Mobil (class 0)
    1: 70,   # Motor (class 1)
    2: 50,   # Orang (class 2)
    3: 250   # Truk (class 3)
}

# Ambang batas jarak (cm) untuk peringatan
DISTANCE_THRESHOLDS = {
    0: 300,  # Mobil
    1: 300,  # Motor
    2: 50,   # Orang
    3: 400   # Truk
}

# Nama kelas sesuai urutan dataset
CLASS_NAMES = ["Mobil", "Motor", "Orang", "Truk"]

# -------------------------
# Load model YOLOv5
# -------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = torch.hub.load(
#     'ultralytics/yolov5',
#     'custom',
#     path='sBest2.pt',
#     # path='suBest3.pt',
#     # path='yolov5s-hasbi.pt',
#     # path='best.pt',
#     # path='best.pt',
#     # path='best.pt',
# )
# model.to(device)
# model.eval()

from ultralytics import YOLO

# Load your trained weights
model = YOLO("./models/suBest3_float16.tflite")

# Run inference
# results = model("test.jpg")   # or a video, webcam, etc.
# results.show()                # display results
# results.save("runs/detect")   # save results

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
    # buang frame jika masih memproses frame sebelumnya
    if processing_lock.locked():
        return

    with processing_lock:
        # terima {image: dataURL} atau langsung dataURL
        data_url = data.get("image", "") if isinstance(data, dict) else data
        if not data_url:
            return

        # hapus header "data:image/jpeg;base64," jika ada
        encoded = data_url.split(",", 1)[1] if "," in data_url else data_url

        try:
            jpg_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(jpg_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
            if img is None:
                return

            # resize agar lebih cepat
            img = cv2.resize(img, (320, 320))

            # # inferensi YOLOv5
            # with torch.no_grad():
            #     results = model(img, size=640)

            # # hasil deteksi: (n,6) -> x1, y1, x2, y2, conf, cls
            # det = results.xyxy[0].cpu().numpy() if len(results.xyxy) else np.empty((0, 6))

            # Inference (Ultralytics YOLO API)
            results = model.predict(img, imgsz=320, conf=0.25)

            # results is a list of ultralytics.engine.results.Results
            det = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:  # Boxes object
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        det.append([x1, y1, x2, y2, conf, cls])

            det = np.array(det) if len(det) else np.empty((0, 6))

            h, w = img.shape[:2]
            detections = []

            for x1, y1, x2, y2, conf, cls in det:
                cls = int(cls)
                perceived_width = float(x2 - x1)
                distance = None
                alert = False

                if cls in KNOWN_WIDTHS and perceived_width > 0:
                    distance = (KNOWN_WIDTHS[cls] * FOCAL_LENGTH) / perceived_width
                    # cek apakah terlalu dekat
                    if distance <= DISTANCE_THRESHOLDS.get(cls, 999999):
                        alert = True

                class_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
                detections.append({
                    "class_id": cls,
                    "class_name": class_name,
                    "conf": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "distance_cm": float(distance) if distance is not None else None,
                    "alert": alert
                })

            emit("detections", {"detections": detections, "width": w, "height": h})
        except Exception as e:
            print("Error handling frame:", repr(e))
            return

if __name__ == "__main__":
    # debug=False agar tidak spawn 2x
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
