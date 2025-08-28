import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cv2
import time
import platform

# -------------------------
# Sound Alert (cross-platform)
# -------------------------
if platform.system() == "Windows":
    import winsound
    def beep():
        winsound.Beep(1000, 500)  # frequency=1000Hz, duration=500ms
else:
    import os
    def beep():
        os.system('play -nq -t alsa synth 0.3 sine 1000')  # Linux beep (requires sox)

# -------------------------
# Load YOLOv5 model
# -------------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# -------------------------
# Camera Calibration Result (replace with your own)
# -------------------------
FOCAL_LENGTH = 888  # <-- your calibrated focal length

# -------------------------
# Real-world object widths (cm)
# -------------------------
KNOWN_WIDTHS = {
    0: 175,  # Mobil (class 0)
    1: 70,   # Motor (class 1)
    2: 50,   # Orang (class 2)
    3: 250   # Truk (class 3)
}

CLASS_NAMES = ["Mobil", "Motor", "Orang", "Truk"]

# -------------------------
# Safety thresholds (cm)
# -------------------------
SAFE_DISTANCE = {
    "Mobil": 300,  
    "Motor": 300,  
    "Orang": 50,   
    "Truk": 400    
}

# -------------------------
# Test source (image or video)
# -------------------------
source = "videotest5.mp4"  # change to 0 for webcam
cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    for x1, y1, x2, y2, conf, cls in detections:
        cls = int(cls)
        label = CLASS_NAMES[cls]
        perceived_width = x2 - x1

        if cls in KNOWN_WIDTHS:
            distance = (KNOWN_WIDTHS[cls] * FOCAL_LENGTH) / perceived_width

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(distance)}cm", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # -------------------------
            # Instant Proximity Warning
            # -------------------------
            if distance < SAFE_DISTANCE[label]:
                cv2.putText(frame, f"WARNING: {label} TOO CLOSE!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                print(f"[ALERT] {label} dangerously close! ({int(distance)} cm)")
                beep()

    cv2.imshow("Distance Estimation with Warning", frame)
    if cv2.waitKey(1) & 0xFF == ord("c"):
        break

cap.release()
cv2.destroyAllWindows()
