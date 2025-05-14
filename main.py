import cv2
import numpy as np
import hand
import marker

cap = cv2.VideoCapture(0)
window_name = 'Camera Pose & ArUco Markers'

mk = marker.Marker()
hd = hand.Hand()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mk.detect(frame,visualize=True)
    hd.detect(frame,visualize=True)

    cv2.imshow(window_name, frame)
    
    # BREAK LOOP
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
