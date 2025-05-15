import cv2
import numpy as np
import hand
import marker

cap = cv2.VideoCapture(0)
window_name = 'Camera Pose & ArUco Markers'

cm = marker.Camera()
hd = hand.Hand()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cm.update(frame,visualize=True)
    hd.detect(frame,visualize=False)
    if hd.estimate_depth(cm):
        hd.estimate_index_tip(frame,cm)
    try:
        pass
        #hd.visualize_primary_landmark(frame)
        hd.visualize_landmark(frame, 8, threshold=0.02)
    except Exception as e:
        print(e)

    a = hd.landmarks[0]
    #marker.draw_world_line_on_image(frame,a,cm.uv_to_world(a[0],a[1]),cm,-1)

    cv2.imshow(window_name, frame)
    
    # BREAK LOOP
    if cv2.waitKey(1) == 27 or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
