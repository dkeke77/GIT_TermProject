import cv2
import numpy as np
import os

# === 설정 ===
MARKER_LENGTH = 0.05  # 마커의 한 변 길이 (단위: 미터)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
SAVE_PATH = "calib_images"

os.makedirs(SAVE_PATH, exist_ok=True)

# === 1단계: 이미지 수집 ===
def capture_images():
    cap = cv2.VideoCapture(0)
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
    count = 0

    print("[i] SPACE 키로 이미지 저장 / ESC 키로 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == 32 and ids is not None:
            cv2.imwrite(f"{SAVE_PATH}/img_{count:03d}.png", frame)
            print(f"Saved img_{count:03d}.png")
            count += 1

    cap.release()
    cv2.destroyAllWindows()

# === 2단계: 캘리브레이션 ===
def calibrate():
    obj_points = []
    img_points = []
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

    # 마커의 3D 좌표 정의 (ID 0 기준, 좌상단부터 시계방향)
    marker_obj_pts = np.array([
        [-MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
        [ MARKER_LENGTH/2,  MARKER_LENGTH/2, 0],
        [ MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
        [-MARKER_LENGTH/2, -MARKER_LENGTH/2, 0],
    ], dtype=np.float32)

    for fname in os.listdir(SAVE_PATH):
        img = cv2.imread(os.path.join(SAVE_PATH, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and 0 in ids:
            for i, id_ in enumerate(ids.flatten()):
                if id_ == 0:
                    img_points.append(corners[i][0])
                    obj_points.append(marker_obj_pts)

    if len(obj_points) < 5:
        print("[!] 충분한 이미지가 없습니다 (최소 5장 필요)")
        return

    # 카메라 캘리브레이션
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None)

    print("=== Calibration Result ===")
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs.ravel())

    np.savez("camera_params", K=camera_matrix, dist=dist_coeffs)
    print("[i] camera_params.npz 로 저장됨")

if __name__ == "__main__":
    # capture_images()
    calibrate()
