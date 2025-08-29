import base64
import threading
import numpy as np
import cv2
import torch
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# IMPORT MIDAS/FASTDEPTH
from depth_estimation_fastdepth import FastDepthEstimator

app = Flask(__name__)
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
# Load model YOLOv5 with TensorFlow Lite
# -------------------------
from ultralytics import YOLO

# Load your trained weights
model = YOLO("./models/suBest3_float16.tflite")

# =========================================================================
# 🎯 INITIALIZE MIDAS WITH DEBUG LOGGING
# =========================================================================
print("=" * 60)
print("🚀 INITIALIZING MIDAS/FASTDEPTH")
print("=" * 60)

depth_estimator = None
midas_working = False

try:
    depth_estimator = FastDepthEstimator("models/mobilenet_depth.tflite", input_size=(320, 320))
    midas_working = True
    print("✅ MIDAS INITIALIZED SUCCESSFULLY!")
    print("🎯 MiDaS will provide depth estimation")
    print("📊 Expected depth range: 0.1 - 20.0 meters")
except Exception as e:
    print("❌ MIDAS INITIALIZATION FAILED!")
    print(f"🔧 Error: {e}")
    print("⚡ Will use focal length distance only")
    midas_working = False

print("=" * 60)

processing_lock = threading.Lock()
frame_count = 0  # Global frame counter for debugging

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("connect")
def on_connect():
    print("🔌 Client connected")
    if midas_working:
        print("📡 MiDaS ready for depth estimation")

@socketio.on("disconnect")
def on_disconnect():
    print("🔌 Client disconnected")

@socketio.on("frame")
def handle_frame(data):
    global frame_count
    
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
            frame_count += 1

            # =========================================================================
            # 🎯 MIDAS DEPTH ESTIMATION WITH DEBUG LOGGING
            # =========================================================================
            depth_map = None
            midas_status = "❌ Not Available"
            
            if midas_working and depth_estimator is not None:
                try:
                    print(f"\n🔍 Frame {frame_count}: Starting MiDaS depth estimation...")
                    
                    # Estimate depth using MiDaS
                    depth_map = depth_estimator.estimate_depth(img)
                    
                    # Analyze depth map quality
                    depth_min = np.min(depth_map)
                    depth_max = np.max(depth_map)
                    depth_mean = np.mean(depth_map)
                    depth_std = np.std(depth_map)
                    
                    print(f"📊 MiDaS Results for Frame {frame_count}:")
                    print(f"   🎯 Depth Range: {depth_min:.2f} - {depth_max:.2f} meters")
                    print(f"   📈 Mean Depth: {depth_mean:.2f} ± {depth_std:.2f} meters")
                    
                    # Quality check
                    dynamic_range = depth_max - depth_min
                    if dynamic_range > 0.5:
                        print(f"   ✅ EXCELLENT: Dynamic range = {dynamic_range:.2f}m")
                        midas_status = f"✅ Working (Range: {dynamic_range:.1f}m)"
                    elif dynamic_range > 0.1:
                        print(f"   🟡 MODERATE: Dynamic range = {dynamic_range:.2f}m")
                        midas_status = f"🟡 Limited (Range: {dynamic_range:.1f}m)"
                    else:
                        print(f"   ⚠️  POOR: Dynamic range = {dynamic_range:.2f}m")
                        midas_status = f"⚠️ Poor (Range: {dynamic_range:.1f}m)"
                        
                except Exception as e:
                    print(f"❌ MiDaS estimation failed for frame {frame_count}: {e}")
                    depth_map = None
                    midas_status = f"❌ Error: {str(e)[:30]}"
            else:
                if frame_count % 10 == 1:  # Print occasionally
                    print(f"⚡ Frame {frame_count}: Using focal length only (MiDaS disabled)")
                midas_status = "⚡ Focal Length Only"

            # =========================================================================
            # YOLO INFERENCE
            # =========================================================================
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
            
            print(f"🎯 Frame {frame_count}: Found {len(det)} objects")

            for i, (x1, y1, x2, y2, conf, cls) in enumerate(det):
                cls = int(cls)
                perceived_width = float(x2 - x1)
                
                # =========================================================================
                # 🎯 DISTANCE CALCULATION WITH MIDAS DEBUG
                # =========================================================================
                focal_distance = None
                midas_distance = None
                combined_distance = None
                alert = False

                # 1. Focal length distance (traditional method)
                if cls in KNOWN_WIDTHS and perceived_width > 0:
                    focal_distance = (KNOWN_WIDTHS[cls] * FOCAL_LENGTH) / perceived_width
                    print(f"   📏 Object {i+1} ({CLASS_NAMES[cls]}): Focal distance = {focal_distance:.2f} cm")

                # 2. MiDaS depth distance (if available)
                if depth_map is not None:
                    try:
                        midas_distance_m = depth_estimator.get_object_depth(depth_map, [x1, y1, x2, y2])
                        midas_distance = midas_distance_m * 100  # Convert to cm
                        print(f"   🎯 Object {i+1} ({CLASS_NAMES[cls]}): MiDaS distance = {midas_distance:.2f} cm")
                        
                        # Combine distances (weighted average: 70% focal, 30% MiDaS)
                        if focal_distance is not None:
                            combined_distance = (focal_distance * 0.7) + (midas_distance * 0.3)
                            print(f"   🔄 Object {i+1} ({CLASS_NAMES[cls]}): Combined distance = {combined_distance:.2f} cm")
                        else:
                            combined_distance = midas_distance
                            print(f"   🎯 Object {i+1} ({CLASS_NAMES[cls]}): Using MiDaS only = {combined_distance:.2f} cm")
                            
                    except Exception as e:
                        print(f"   ❌ Object {i+1}: MiDaS depth calculation failed: {e}")
                        combined_distance = focal_distance
                        
                else:
                    combined_distance = focal_distance
                    if frame_count % 5 == 1:  # Occasional reminder
                        print(f"   ⚡ Object {i+1} ({CLASS_NAMES[cls]}): Focal only = {combined_distance:.2f} cm")

                # 3. Alert check
                if combined_distance is not None:
                    threshold = DISTANCE_THRESHOLDS.get(cls, 999999)
                    if combined_distance <= threshold:
                        alert = True
                        print(f"   🚨 ALERT! Object {i+1} ({CLASS_NAMES[cls]}) too close: {combined_distance:.2f} cm < {threshold} cm")

                # Prepare detection data
                class_name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
                detections.append({
                    "class_id": cls,
                    "class_name": class_name,
                    "conf": float(conf),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "distance_cm": float(combined_distance) if combined_distance is not None else None,
                    "focal_distance_cm": float(focal_distance) if focal_distance is not None else None,
                    "midas_distance_cm": float(midas_distance) if midas_distance is not None else None,
                    "alert": alert
                })

            # =========================================================================
            # 🎯 FRAME PROCESSING SUMMARY
            # =========================================================================
            if frame_count % 10 == 0:  # Print summary every 10 frames
                print(f"\n📊 PROCESSING SUMMARY - Frame {frame_count}")
                print(f"   🎯 MiDaS Status: {midas_status}")
                print(f"   🎯 Objects Detected: {len(detections)}")
                print(f"   🚨 Alerts Triggered: {sum(1 for d in detections if d['alert'])}")
                print("─" * 50)

            # Send results to client
            emit("detections", {
                "detections": detections, 
                "width": w, 
                "height": h,
                "frame_count": frame_count,
                "midas_status": midas_status,
                "midas_working": midas_working
            })
            
        except Exception as e:
            print(f"❌ Error handling frame {frame_count}: {repr(e)}")
            return

if __name__ == "__main__":
    print("\n🚀 STARTING APPLICATION WITH MIDAS DEBUG")
    print("=" * 60)
    print("📺 Open browser: http://localhost:5000")
    print("🎬 Click 'Start Camera' to begin")
    print("👀 Watch this terminal for MiDaS debug info!")
    print("=" * 60)
    
    # debug=False agar tidak spawn 2x
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)