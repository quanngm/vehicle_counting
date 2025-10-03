import cv2

points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked point: ({x}, {y})")
        points.append((x, y))

# Load video and read the first frame
cap = cv2.VideoCapture("Video2.mp4")
ret, frame = cap.read()

if ret:
    cv2.imshow("Click to select points", frame)
    cv2.setMouseCallback("Click to select points", click_event)

    print("Left click to get coordinates. Press ESC to exit.")
    while True:
        cv2.imshow("Click to select points", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    cap.release()

    print("All selected points:", points)
else:
    print("Failed to load video")
