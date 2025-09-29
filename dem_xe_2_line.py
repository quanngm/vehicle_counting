import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Video input
cap = cv2.VideoCapture("Sample_2.mp4")

# Define lines for two directions (459, 545), (475, 12), (1375, 573), (1367, 998)]
LINE_A_START = sv.Point(459, 545)
LINE_A_END   = sv.Point(475, 12)

LINE_B_START = sv.Point(1375, 573)
LINE_B_END   = sv.Point(1367, 998)

# Track history
track_history = defaultdict(list)

# Crossed IDs and class counts
crossed_A = set()
crossed_B = set()
count_A = defaultdict(int)  # e.g., count_A["car"] += 1
count_B = defaultdict(int)

# Class ID to label (based on your training config)
CLASS_MAP = {
    0: "scooter",
    1: "car"
}

# Check if object crossed a line
def is_crossing_line(prev_point, curr_point, line_start, line_end):
    def ccw(A, B, C):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
    A, B = line_start, line_end
    C = sv.Point(*prev_point)
    D = sv.Point(*curr_point)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# Output video
video_info = sv.VideoInfo.from_video_path("Sample_1.mp4")
with sv.VideoSink("output_simple_counte.mp4", video_info) as sink:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, classes=[0, 1], persist=True, tracker="bytetrack.yaml")
        if not results or results[0].boxes.id is None:
            sink.write_frame(frame)
            continue

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            center = (float(x), float(y))
            class_name = CLASS_MAP.get(cls_id, "unknown")

            track = track_history[track_id]
            track.append(center)
            if len(track) > 30:
                track.pop(0)

            # Check Line A
            if len(track) >= 2 and track_id not in crossed_A:
                if is_crossing_line(track[-2], track[-1], LINE_A_START, LINE_A_END):
                    crossed_A.add(track_id)
                    count_A[class_name] += 1

            # Check Line B
            if len(track) >= 2 and track_id not in crossed_B:
                if is_crossing_line(track[-2], track[-1], LINE_B_START, LINE_B_END):
                    crossed_B.add(track_id)
                    count_B[class_name] += 1

        # Draw lines
        cv2.line(frame, (LINE_A_START.x, LINE_A_START.y), (LINE_A_END.x, LINE_A_END.y), (0, 255, 0),4)
        cv2.line(frame, (LINE_B_START.x, LINE_B_START.y), (LINE_B_END.x, LINE_B_END.y), (255, 0, 0), 4)

        # Display counts
        cv2.putText(frame, f"Line A : car: {count_A['scooter']}  scooter: {count_A['car']}",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 4)
        cv2.putText(frame, f"Line B : car: {count_B['scooter']}  scooter: {count_B['car']}",
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 4)

        sink.write_frame(frame)

cap.release()
