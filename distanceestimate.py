import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import cv2

# -------------------------
# Load YOLOv5 model
# -------------------------
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# -------------------------
# Camera Calibration Result (replace with your own)
# -------------------------
FOCAL_LENGTH = 888  # <-- put your calculated focal length here

# -------------------------
# Real-world object widths (cm)
# -------------------------
KNOWN_WIDTHS = {
    0: 175,  # Mobil (class 0)
    1: 70,   # Motor (class 1)
    2: 50,   # Orang (class 2) - average shoulder width
    3: 250   # Truk (class 3)
}

# Class names (match your dataset order!)
CLASS_NAMES = ["Mobil", "Motor", "Orang", "Truk"]

# -------------------------
# Test source (image or video)
# -------------------------
source = "videotest3.mp4"  # can also be "test.jpg" or 0 for webcam

cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

    for x1, y1, x2, y2, conf, cls in detections:
        cls = int(cls)
        label = CLASS_NAMES[cls]

        # Bounding box width in pixels
        perceived_width = x2 - x1

        if cls in KNOWN_WIDTHS:
            distance = (KNOWN_WIDTHS[cls] * FOCAL_LENGTH) / perceived_width

            # Draw bounding box + distance
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(distance)}cm", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Distance Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
