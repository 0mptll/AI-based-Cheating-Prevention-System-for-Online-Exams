# detect_lip_movement.py
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# Stable sets of lip landmarks
UPPER_LIP_INDICES = [13, 78, 82, 87, 95]
LOWER_LIP_INDICES = [14, 308, 312, 317, 324]

class LipMovementDetector:
    def __init__(self, threshold=0.3):  # Further lowered the threshold
        self.threshold = threshold
        self.prev_avg_distance = None
        self.mp_face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        """
        Detects lip movement in a given video frame.

        Args:
            frame: A BGR video frame (NumPy array).

        Returns:
            True if lip movement is detected, False otherwise.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            self.prev_avg_distance = None
            return False

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        total_distance = 0
        count = min(len(UPPER_LIP_INDICES), len(LOWER_LIP_INDICES))

        for i in range(count):
            upper_y = landmarks[UPPER_LIP_INDICES[i]].y * h
            lower_y = landmarks[LOWER_LIP_INDICES[i]].y * h
            total_distance += abs(lower_y - upper_y)

        avg_distance = total_distance / count

        if self.prev_avg_distance is None:
            self.prev_avg_distance = avg_distance
            return False

        movement_detected = abs(avg_distance - self.prev_avg_distance) > self.threshold
        self.prev_avg_distance = avg_distance

        return movement_detected
