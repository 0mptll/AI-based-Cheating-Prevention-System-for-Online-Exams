import cv2
import mediapipe as mp
import numpy as np

class GazeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Iris and outer eye corners (left and right eyes)
        self.left_eye = {
            "iris": [468, 469, 470, 471],
            "outer": [33, 133]  # left corner, right corner
        }
        self.right_eye = {
            "iris": [473, 474, 475, 476],
            "outer": [362, 263]  # right corner, left corner
        }

    def _iris_relative_position(self, landmarks, eye, frame_shape):
        h, w = frame_shape
        iris_coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye["iris"]])
        eye_left_x = landmarks[eye["outer"][0]].x * w
        eye_right_x = landmarks[eye["outer"][1]].x * w

        iris_x_mean = np.mean(iris_coords[:, 0])
        eye_width = eye_right_x - eye_left_x

        relative_x = (iris_x_mean - eye_left_x) / (eye_width + 1e-6)  # Prevent div by zero
        return relative_x

    def detect_eye_only_gaze_direction(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            left_pos = self._iris_relative_position(landmarks, self.left_eye, frame.shape[:2])
            right_pos = self._iris_relative_position(landmarks, self.right_eye, frame.shape[:2])

            avg_pos = (left_pos + right_pos) / 2

            if avg_pos < 0.35:
                return "Looking Right"
            elif avg_pos > 0.65:
                return "Looking Left"
            else:
                return "Looking Center"
        return "Face or Eyes Not Detected"
