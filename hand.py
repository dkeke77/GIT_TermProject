import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

class Hand:
    def __init__(self):
        self.landmarks = [(0.0, 0.0, 0.0)] * 21
        self.landmark_len = []
    
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
                    self.landmarks.append((lm.x, lm.y, lm.z))
                    if visualize:
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                        cv2.putText(frame, f"{idx}", (cx+5, cy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def calib_hand(self,cam_pos,cam_vec):
        pass