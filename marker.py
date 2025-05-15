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

class Camera:
    def __init__(self):
        self.R_wc = np.eye(3)
        self.t_wc = np.zeros((3,1))
        self.cam_pos = np.array([0.0, 0.0, 0.0])
        self.cam_view = np.array([0, 0, 1])
        self.K_inv = np.linalg.inv(K)
        self.img_h = 0
        self.img_w = 0

    def update(self,frame,visualize=False):
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.img_h, self.img_w = frame.shape[:2]

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
                        self.cam_view = (self.R_wc @ [0,0,-1]).flatten()
                        self.cam_view = self.cam_view / np.linalg.norm(self.cam_view)

                        if visualize:
                            text = f"Camera Pos (m): x={self.cam_pos[0]:.2f}, y={self.cam_pos[1]:.2f}, z={self.cam_pos[2]:.2f}"
                            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            text = f"Camera View (m): x={self.cam_view[0]:.2f}, y={self.cam_view[1]:.2f}, z={self.cam_view[2]:.2f}"
                            cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, 0.03)

    def uv_to_world_dir(self, u, v, isNormalized=True):
        if isNormalized:
            u *= self.img_w
            v *= self.img_h
        uv_h = np.array([u, v, 1.0])
        ray_dir_cam = self.K_inv @ uv_h
        ray_dir_cam = ray_dir_cam / np.linalg.norm(ray_dir_cam)
        return self.R_wc @ ray_dir_cam
    
def draw_world_line_on_image(image, P0, direction, camera, length=0.1, color=(0, 255, 0)):
    # 선분의 양 끝점 (월드 좌표)
    P1 = P0
    P2 = P0 + direction / np.linalg.norm(direction) * length
    print(P1, P2)

    # 3D 점들을 (3,1) 형식으로 정리
    world_points = np.array([P1, P2], dtype=np.float32).reshape(-1, 3)

    # 월드 좌표 → 카메라 좌표 (R, t는 월드→카메라 변환)
    R_cw = camera.R_wc.T
    t_cw = -R_cw @ camera.t_wc
    rvec, _ = cv2.Rodrigues(R_cw)

    # 이미지 좌표로 투영
    img_pts, _ = cv2.projectPoints(world_points, rvec, t_cw, K, dist)
    p1 = tuple(map(int, img_pts[0].ravel()))
    p2 = tuple(map(int, img_pts[1].ravel()))

    # 이미지에 선 그리기
    cv2.line(image, p1, p2, color, 2)