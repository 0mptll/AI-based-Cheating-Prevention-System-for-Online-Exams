import cv2
import mediapipe as mp
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),   # Right eye right corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0)   # Right mouth corner
        ], dtype=np.float32)

    def estimate_pose(self, face_roi):
        h, w = face_roi.shape[:2]
        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_face)

        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),   # Nose tip
            (landmarks[152].x * w, landmarks[152].y * h), # Chin
            (landmarks[263].x * w, landmarks[263].y * h), # Left eye left corner
            (landmarks[33].x * w, landmarks[33].y * h),   # Right eye right corner
            (landmarks[287].x * w, landmarks[287].y * h), # Left mouth corner
            (landmarks[57].x * w, landmarks[57].y * h)    # Right mouth corner
        ], dtype=np.float32)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        pitch, yaw, roll = angles

        return int(pitch), int(yaw), int(roll)
