import cv2
import numpy as np

# --- Load calibration parameters ---
params = np.load("camera_params.npz")
K = params["K"]
dist = params["dist"]
K_inv = np.linalg.inv(K)

# 마커 세부 설정
MARKER_LENGTH = 0.05
detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 
    cv2.aruco.DetectorParameters()
    )
# 3D 마커 좌표 (ID 0 기준)
obj_points = np.array([
    [-MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
    [ MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
    [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
], dtype=np.float32)

class Marker:
    def __init__(self):
        self.R_wc = np.eye(3)
        self.t_wc = np.zeros((3,1))
        self.cam_pos = np.array([0.0, 0.0, 0.0])
        self.cam_view = np.array([0, 0, 1])

    def detect(self,frame,visualize=False):
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, id_ in enumerate(ids.flatten()):
                if id_ == 0:
                    image_points = corners[i][0]
                    success, rvec, tvec = cv2.solvePnP(obj_points, image_points, K, dist)

                    if success:
                        R, _ = cv2.Rodrigues(rvec)
                        t = tvec

                        self.R_wc = R.T
                        self.t_wc = -R.T @ t

                        self.cam_pos = self.t_wc.flatten()
                        self.cam_view = self.R_wc.T @ self.cam_pos
                        self.cam_view = self.cam_view / np.linalg.norm(self.cam_view)

                        if visualize:
                            text = f"Camera Pos (m): x={self.cam_pos[0]:.2f}, y={self.cam_pos[1]:.2f}, z={self.cam_pos[2]:.2f}"
                            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)

                        