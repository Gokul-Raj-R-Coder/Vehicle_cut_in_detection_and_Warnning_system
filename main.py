import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from scipy.spatial import distance
from sort import Sort

# Load YOLOv8 model
model = YOLO('../Yolo-Weights/yolov8n.pt')

# Open video capture
cap = cv2.VideoCapture("C:\\Users\\govindraj R\\Videos\\Captures\\final testcase.mp4")

# Class names for YOLO
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Parameters
ttc_threshold = 0.7  # Time-to-collision threshold in seconds
fps = cap.get(cv2.CAP_PROP_FPS)

# Polygon ROI coordinates
roi_pts = np.array([[0, 720], [550, 300], [740, 300], [1280, 720]], dtype=np.int32)

# Function to calculate TTC
def calculate_ttc(distance, relative_velocity):
    if relative_velocity <= 0:  # Vehicle is moving away or stationary
        return float('inf')
    return distance / relative_velocity  # TTC in seconds

prev_frame_time = None
prev_positions = {}

while True:
    success, frame = cap.read()
    if not success:
        break

    current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get current time in seconds

    # Draw ROI polygon
    cv2.polylines(frame, [roi_pts], isClosed=True, color=(255, 0, 0), thickness=2)

    results = model(frame, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["person", "car", "truck", "bus", "motorbike", "cow"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    current_positions = {}

    for result in resultsTracker:
        x1, y1, x2, y2, obj_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Check if the vehicle's center is within the ROI polygon
        if (cv2.pointPolygonTest(roi_pts, (x2-(w//4), y2-(h//4)), False) >= 0) or (cv2.pointPolygonTest(roi_pts, (x1+(w//4), y1+(3*h//4)), False) >= 0):

            current_positions[obj_id] = (cx, cy, x1, y1, x2, y2)
            cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    if prev_positions:
        for obj_id, (prev_cx, prev_cy, prev_x1, prev_y1, prev_x2, prev_y2) in prev_positions.items():
            if obj_id not in current_positions:
                continue

            curr_cx, curr_cy, curr_x1, curr_y1, curr_x2, curr_y2 = current_positions[obj_id]

            # Calculate distance and relative velocity
            displacement_vector = (curr_cx - prev_cx, curr_cy - prev_cy)
            dist = distance.euclidean((prev_cx, prev_cy), (curr_cx, curr_cy))
            time_diff = current_frame_time - prev_frame_time
            if time_diff == 0:
                continue



            # Reference direction from camera center to object center
            ref_vector = (curr_cx - 640, curr_cy - 720)
            ref_magnitude = distance.euclidean((640, 720), (curr_cx, curr_cy))

            # Relative velocity calculation (projected on reference direction)
            relative_velocity = (displacement_vector[0] * ref_vector[0] + displacement_vector[1] * ref_vector[1]) / (time_diff * ref_magnitude)

            # Calculate TTC
            distance_from_center = ref_magnitude
            ttc = calculate_ttc(distance_from_center, relative_velocity)
            print(ttc)
            if ttc < ttc_threshold:
                # Generate warning
                cv2.rectangle(frame, (int(curr_x1), int(curr_y1)), (int(curr_x2), int(curr_y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"Collision Warning: {ttc:.2f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cvzone.cornerRect(frame, (curr_x1, curr_y1, curr_x2 - curr_x1, curr_y2 - curr_y1), l=9, rt=2, colorR=(0, 255, 0))

            # Display speed of the vehicle
            speed_kmh = relative_velocity * 3.6  # Convert speed from pixels per second to km/h
            cv2.putText(frame, f"Speed: {speed_kmh:.2f} km/h", (int(curr_x1), int(curr_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_positions = current_positions
    prev_frame_time = current_frame_time

cap.release()
cv2.destroyAllWindows()
