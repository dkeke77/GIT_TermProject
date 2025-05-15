import mediapipe as mp
from scipy.optimize import minimize
import numpy as np
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def intersect_ray_with_plane(cam_origin, ray_dir, z_plane=0.003):
    t = (z_plane - cam_origin[2]) / ray_dir[2]
    return cam_origin + t * ray_dir

class Hand:
    def __init__(self):
        self.landmarks = [(0.0, 0.0, 0.0)] * 21
        self.landmarks_world = {0:None,5:None,9:None}
        self.landmark_len = {
            "0-5":0.076,
            "0-9":0.072,
            "5-9":0.024,
            # index finger
            "5-6":0.042,
            "6-7":0.027,
            "7-8":0.024
        }
    
    def detect(self, frame, visualize=False):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        for i in range(len(self.landmarks)):
            self.landmarks[i] = (0.0, 0.0, 0.0)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)                
                for idx, lm in enumerate(hand_landmarks.landmark):
                    h, w = frame.shape[:2]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmarks[idx] = (lm.x, lm.y, lm.z)
                    if visualize:
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                        cv2.putText(frame, f"{idx}", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            return True
        else:
            return False

    def estimate_depth(self, cam, z_range=(0.03, 1), num_samples=500):
        """
        cam_origin: 카메라의 월드 위치 (np.array shape (3,))
        uv_vectors: dict, {id: ray_vector (np.array shape (3,))}
        cm: Camera 객체
        z_range: tuple, z 탐색 범위 (최솟값, 최댓값)
        num_samples: int, 브루트포스 샘플 수
        """
        z_candidates = np.linspace(z_range[0], z_range[1], num_samples)
        min_error = float('inf')
        best_z = None
        best_positions = {}

        # 기준: point0
        ray0 = cam.uv_to_world_dir(self.landmarks[0][0],self.landmarks[0][1])
        ray5 = cam.uv_to_world_dir(self.landmarks[5][0],self.landmarks[5][1])
        ray9 = cam.uv_to_world_dir(self.landmarks[9][0],self.landmarks[9][1])

        len_05 = self.landmark_len["0-5"]
        len_09 = self.landmark_len["0-9"]
        len_59 = self.landmark_len["5-9"]

        for z in z_candidates:
            p0 = cam.cam_pos + ray0 * z
            p5 = cam.cam_pos + ray5 * z
            p9 = cam.cam_pos + ray9 * z

            d_05 = np.linalg.norm(p0 - p5)
            d_09 = np.linalg.norm(p0 - p9)
            d_59 = np.linalg.norm(p5 - p9)

            error = (d_05 - len_05)**2 + (d_09 - len_09)**2 + (d_59 - len_59)**2

            if error < min_error:
                min_error = error
                best_z = z
                best_positions = {0: p0, 5: p5, 9: p9}

        for key, value in best_positions.items():
            self.landmarks_world[key] = value
        return best_z
                
    def visualize_primary_landmark(self,frame):
        for i in [0,5,9]:
            lm = self.landmarks[i]
            coord = self.landmarks_world[i]
            h, w = frame.shape[:2]
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            cv2.putText(frame, f"{i}", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})", (cx+5, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    def estimate_index_tip(self,frame,cam):
        try:
            self.estimate_landmark_world_from_base(5, 6, cam)
            self.estimate_landmark_world_from_base(6, 7, cam)
            self.estimate_landmark_world_from_base(7, 8, cam)
        except Exception as e:
            print(e)

        self.visualize_landmark(frame,8,threshold=0.005)

    def visualize_landmark(self,frame,idx,threshold=0.02):
        coord = self.landmarks_world[idx]
        lm = self.landmarks[idx]
        h, w = frame.shape[:2]
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        cv2.putText(frame, f"({coord[0]:.2f}, {coord[1]:.2f}, {coord[2]:.2f})", (cx+5, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if coord[2] < threshold:
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        else:
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    def estimate_landmark_world_from_base(self, base_idx, target_idx, cam):
        if self.landmarks_world[base_idx] is None:
            raise ValueError(f"Base landmark {base_idx} world position is not set.")
        
        # base의 월드 좌표 (구 중심)
        c = self.landmarks_world[base_idx]
        
        # 카메라 위치와 ray
        cam_pos = cam.cam_pos
        d = cam.uv_to_world_dir(self.landmarks[target_idx][0],self.landmarks[target_idx][1])  # 단위벡터
        o = cam_pos
        
        # 거리 정보
        min_idx, max_idx = sorted([base_idx, target_idx])
        key = f"{min_idx}-{max_idx}"
        if key not in self.landmark_len:
            raise ValueError(f"Distance info for landmarks {min_idx}-{max_idx} not found in landmark_len.")
        r = self.landmark_len[key]
        
        # (o + t*d - c)^2 = r^2 → 이차방정식으로 푼다
        oc = o - c
        b = 2 * np.dot(d, oc)
        c_ = np.dot(oc, oc) - r**2
        
        discriminant = b**2 - 4*c_
        
        if discriminant < 0:
            raise ValueError("No intersection between ray and sphere.")
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2
        t2 = (-b + sqrt_disc) / 2

        # 두 후보 중 하나 선택
        # 방법 1: 더 가까운 점
        t = min(t1, t2) if t1 > 0 else t2

        # 최종 월드 좌표
        target_world = o + t2 * d
        self.landmarks_world[target_idx] = target_world

        return target_world

    # def calib_hand(self,marker):
    #     for idx in [0, 5, 9]:
    #         lm = self.landmarks[idx]
    #         u, v = int(lm[0] * marker.w), int(lm[1] * marker.h)
    #         #self.landmarks_2d[idx] = (u, v)

    #         # 정규화된 카메라 좌표계의 ray
    #         uv1 = np.array([[u], [v], [1]], dtype=np.float32)
    #         ray_camera = marker.K_inv @ uv1
    #         ray_camera = ray_camera / np.linalg.norm(ray_camera)

    #         # 월드 좌표계로 변환
    #         ray_world = marker.R_wc @ ray_camera
    #         cam_origin_world = marker.t_wc.flatten()

    #         point_world = intersect_ray_with_plane(cam_origin_world, ray_world.flatten())
    #         self.landmarks_world[idx] = point_world
    #         try:
    #             self.get_distance_ratio()
    #         except:
    #             pass

    # def get_distance_ratio(self):
    #     """캘리브레이션용 거리 비율 반환"""
    #     if len(self.landmarks_world) != 3:
    #         return None
        
    #     p0 = self.landmarks_world[0]
    #     p5 = self.landmarks_world[5]
    #     p9 = self.landmarks_world[9]

    #     d1 = np.linalg.norm(p0 - p5)
    #     d2 = np.linalg.norm(p5 - p9)

    #     ratio = d1 / d2 if d2 != 0 else None
    #     if not(d1 == 0 and d2 == 0):
    #         print(d1,d2)
    #     return ratio, d1, d2
