import mediapipe as mp
from scipy.optimize import minimize
import numpy as np
import cv2
import filter as f

filter_manager = f.LandmarkFilterManager(
    freq=30,         # 카메라 프레임레이트
    min_cutoff=1.0,  # smoothing 정도
    beta=5.0,        # 반응성
    d_cutoff=1.0     # 속도 필터링 cutoff
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def intersect_ray_with_plane(cam_pos, ray_dir, z_plane=0.003):
    t = (z_plane - cam_pos[2]) / ray_dir[2]
    return cam_pos + t * ray_dir

def intersect_ray_sphere(center, radius, ray_origin, ray_dir, approx=False):
    oc = ray_origin - center

    b = np.dot(oc, ray_dir) * 2
    c = np.dot(oc, oc) - radius * radius

    discriminant = b * b - 4 * c

    if discriminant < 0:
        if approx:
            v = center - ray_origin
            t = np.dot(v, ray_dir)
            return [ray_origin + t * ray_dir]
        else:
            return []

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / 2
    t2 = (-b + sqrt_disc) / 2

    if discriminant == 0:
        return [ray_origin + t1 * ray_dir]
    else:
        return [ray_origin + t1 * ray_dir, ray_origin + t2 * ray_dir]

def dist(p1,p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    return np.linalg.norm(p1 - p2)

class Hand:
    def __init__(self):
        self.landmarks = [(0, 0, 0)] * 21
        self.landmarks_world = [(0, 0, 0)] * 21
        self.landmark_len = {
            "0-1": 0, "1-2": 0, "2-3": 0, "3-4": 0,                         # Thumb
            "0-5": 0, "0-9": 0, "0-17": 0, "5-9": 0, "9-13": 0, "13-17": 0, # Palm
            "5-6": 0, "6-7": 0, "7-8": 0,                                   # Index
            "9-10": 0, "10-11": 0, "11-12": 0,                              # Middle
            "13-14": 0, "14-15": 0, "15-16": 0,                             # Ringer
            "17-18": 0, "18-19": 0, "19-20": 0                              # Little
        }
        # self.landmark_len = {
        #     "0-5":0.076,
        #     "0-9":0.072,
        #     "5-9":0.024,
        #     # index finger
        #     "5-6":0.042,
        #     "6-7":0.027,
        #     "7-8":0.024
        # }
    
    def detect(self, frame, visualize=False):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        for i in range(len(self.landmarks)):
            self.landmarks[i] = (0.0, 0.0, 0.0)
        
        if result.multi_hand_landmarks:
            landmarkDetected = 0
            raw_landmarks = [(0.0, 0.0, 0.0)] * 21
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for idx, lm in enumerate(hand_landmarks.landmark):
                    landmarkDetected += 1
                    h, w = frame.shape[:2]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    raw_landmarks[idx] = (lm.x, lm.y, lm.z)
                    if visualize:
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                        cv2.putText(frame, f"{idx}", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            self.landmarks = filter_manager.filter(raw_landmarks)
            return landmarkDetected
        else:
            return 0

    def estimate_depth(self, cam, z_range=(0.01, 0.8), num_samples=50, iterations=3, epsilon=0.001):
        def find_points_on_ray(p0, ray_origin, ray_dir, distance):
            v = p0 - ray_origin
            b = -2 * np.dot(ray_dir, v)
            c = np.dot(v, v) - distance ** 2
            disc = b**2 - 4 * c
            if disc < 0:
                return []
            sqrt_disc = np.sqrt(disc)
            t1 = (-b + sqrt_disc) / 2
            t2 = (-b - sqrt_disc) / 2
            p1 = ray_origin + ray_dir * t1
            p2 = ray_origin + ray_dir * t2
            return [p1, p2] if disc > 0 else [p1]

        ray0 = cam.uv_to_world_dir(*self.landmarks[0])
        ray5 = cam.uv_to_world_dir(*self.landmarks[5])
        ray9 = cam.uv_to_world_dir(*self.landmarks[9])

        best_error = float('inf')
        best_points = {0: None, 5: None, 9: None}
        best_z_idx = -1

        z_min, z_max = z_range

        for step in range(iterations):
            z_candidates = np.linspace(z_min, z_max, num_samples)
            errors = []

            for idx, z in enumerate(z_candidates):
                p0 = cam.cam_pos + ray0 * z
                p5s = find_points_on_ray(p0, cam.cam_pos, ray5, self.landmark_len["0-5"])
                if not p5s:
                    errors.append(float('inf'))
                    continue
                p9s = find_points_on_ray(p0, cam.cam_pos, ray9, self.landmark_len["0-9"])
                if not p9s:
                    errors.append(float('inf'))
                    continue

                min_local_error = float('inf')
                local_best = None
                for p5 in p5s:
                    for p9 in p9s:
                        dist = np.linalg.norm(p5 - p9)
                        err = abs(dist - self.landmark_len["5-9"])
                        if err < min_local_error:
                            min_local_error = err
                            local_best = (p0, p5, p9)

                errors.append(min_local_error)

                if min_local_error < best_error:
                    best_error = min_local_error
                    best_points[0], best_points[5], best_points[9] = local_best
                    best_z_idx = idx

            # coarse to fine: update z_range
            if 0 <= best_z_idx < num_samples:
                lower_idx = max(0, best_z_idx - 2)
                upper_idx = min(num_samples - 1, best_z_idx + 2)
                z_min = z_candidates[lower_idx]
                z_max = z_candidates[upper_idx]

        if best_error < epsilon:
            self.landmarks_world[0] = best_points[0]
            self.landmarks_world[5] = best_points[5]
            self.landmarks_world[9] = best_points[9]
            return True
        return False
                    
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
            #self.visualize_landmark(frame,8,threshold=0.005)
        except Exception as e:
            print(e)

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
        # 카메라 위치와 ray
        ray = cam.uv_to_world_dir(self.landmarks[target_idx][0],self.landmarks[target_idx][1])
        
        # 거리 정보
        min_idx, max_idx = sorted([base_idx, target_idx])
        r = self.landmark_len[f"{min_idx}-{max_idx}"]
        points = intersect_ray_sphere(self.landmarks_world[base_idx],r,cam.cam_pos,ray,approx=False)

        if len(points) == 2:
            if self.landmarks[target_idx][2] > self.landmarks[base_idx][2]:
                self.landmarks_world[target_idx] = points[1]
            else:
                self.landmarks_world[target_idx] = points[0]
        elif len(points) == 1:
            self.landmarks_world[target_idx] = points[0]
        else:
            raise ValueError(f"{min_idx}-{max_idx} : No intersection to sphere")

        return self.landmarks_world[target_idx]
    
    def calib_hand(self,cam):
        for idx in range(21):
            lm = self.landmarks[idx]
            ray = cam.uv_to_world_dir(lm[0], lm[1])
            point_world = intersect_ray_with_plane(cam.cam_pos, ray)
            self.landmarks_world[idx] = point_world
        
        for key in self.landmark_len:
            a, b = key.split('-')
            d = dist(self.landmarks_world[int(a)],self.landmarks_world[int(b)])
            self.landmark_len[key] = (self.landmark_len[key] + d) / 2
