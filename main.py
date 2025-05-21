import cv2
from face_detector import FaceDetector
from head_pose_estimator import HeadPoseEstimator
from detect_upper_body import detect_upper_body
from detect_lip_movement import LipMovementDetector 
# from object_detector import ObjectDetector  # 🔁 NEW
from gaze_tracker import GazeTracker  # ✅ NEW



def main():
    face_detector = FaceDetector("res10_300x300_ssd_iter_140000.caffemodel", "deploy.prototxt.txt")
    head_pose_estimator = HeadPoseEstimator()
    lip_movement_detector = LipMovementDetector()
    # object_detector = ObjectDetector()  # 🔁 NEW
    gaze_tracker = GazeTracker()  



    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return
    print("✅ Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.detect_faces(frame)
        face_count = len(faces)
        alert = False

        # 📦 Detect unauthorized objects
        # unauth_objects = object_detector.detect_unauthorized_objects(frame)
        # for (x1, y1, x2, y2, label, conf) in unauth_objects:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #     cv2.putText(frame, f"⚠️ {label} ({conf:.2f})", (x1, y1 - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # if unauth_objects:
        #     cv2.putText(frame, "🚨 Unauthorized Object Detected!", (10, 230),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


        torso_visible = detect_upper_body(frame)
        torso_status = "✅ Torso Visible" if torso_visible else "⚠️ Torso Not Visible"
        torso_color = (0, 255, 0) if torso_visible else (0, 0, 255)
        cv2.putText(frame, torso_status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, torso_color, 2)

        if face_count == 0:
            cv2.putText(frame, "🚫 Student Absence Warning!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        elif face_count > 1:
            cv2.putText(frame, "⚠️ Multiple Faces Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            pose = head_pose_estimator.estimate_pose(face_roi)
            if pose:
                pitch, yaw, roll = pose
                text = f"Pitch:{pitch}° Yaw:{yaw}° Roll:{roll}°"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)

                if abs(yaw) > 20 or abs(pitch) > 15:
                    alert = True

        # ✅ Lip movement detection
        if lip_movement_detector.detect(frame):
            cv2.putText(frame, "💬 Lip Movement Detected!", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # ✅ Eye-only gaze detection
        gaze_direction = gaze_tracker.detect_eye_only_gaze_direction(frame)
        if gaze_direction != "Face or Eyes Not Detected":
            cv2.putText(frame, f"👁️ Eye Gaze: {gaze_direction}", (10, 270), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            if gaze_direction in ["Looking Left", "Looking Right"]:
                cv2.putText(frame, "⚠️ Eye Movement Detected!", (10, 310), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


        cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if alert:
            cv2.putText(frame, "⚠️ Not Looking Straight!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Face + Head Pose + Torso + Lip Movement Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()