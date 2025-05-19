import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# Initialize pose detector once
pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1)

def detect_upper_body(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return False  # No pose detected

    landmarks = results.pose_landmarks.landmark
    key_ids = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
               mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]

    visibility = [landmarks[k.value].visibility for k in key_ids]
    visible = all(v > 0.5 for v in visibility)  # Visibility threshold

    return visible
