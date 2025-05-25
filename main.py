import cv2
import numpy as np
import hand
import camera

cap = cv2.VideoCapture(0)
window_name = 'Camera Pose & ArUco Markers'

cm = camera.Camera()
hd = hand.Hand()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera FPS: {fps}")

stage = "Interval1"
handCalibFrameCount = 0
intervalFrameCount = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    camUpdated = cm.update(frame,visualize=True)
    handDetectedNum = hd.detect(frame,visualize=False)

    if stage == "Interval1":
        intervalFrameCount += 1
        if intervalFrameCount > 120:
            stage = "Hand_Calibration"
            intervalFrameCount = 0

    elif stage == "Hand_Calibration":
        if camUpdated and handDetectedNum == 21:
            hd.calib_hand(cm)
            handCalibFrameCount += 1
        if handCalibFrameCount >= 120:
            print(hd.landmark_len)
            stage = "Interval2"

    elif stage == "Interval2":
        intervalFrameCount += 1
        if intervalFrameCount > 30:
            stage = "Hand_Tracking"

    elif stage == "Hand_Tracking":
        if hd.estimate_depth(cm):
            hd.estimate_index_tip(frame,cm)
        try:
            pass
            hd.visualize_primary_landmark(frame)
            hd.visualize_landmark(frame, 8, threshold=0.005)
        except Exception as e:
            pass
            #print(e)

    cv2.imshow(window_name, frame)
    
    # BREAK LOOP
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
