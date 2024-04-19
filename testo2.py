import cv2
import numpy as np

video = cv2.VideoCapture(0)  # Capture video from webcam (index 0)

while True:
    ret, orig_frame = video.read()
    if not ret:
        continue

    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    inverted_frame = cv2.subtract(np.full_like(frame, 255), frame)  # Invert colors
    hsv = cv2.cvtColor(inverted_frame, cv2.COLOR_BGR2HSV)  # Convert to HSV
    low_white = np.array([0, 0, 200])  # Define lower range for white color
    up_white = np.array([180, 25, 255])  # Define upper range for white color
    mask = cv2.inRange(hsv, low_white, up_white)  # Create a mask for white color
    edges = cv2.Canny(mask, 75, 150)  # Apply Canny edge detection

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(orig_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("frame", orig_frame)  # Display original frame with detected lines
    cv2.imshow("edges", edges)  # Display edges

    key = cv2.waitKey(1)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
