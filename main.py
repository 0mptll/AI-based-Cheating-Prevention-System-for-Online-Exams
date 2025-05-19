import cv2
from face_detector import FaceDetector
from head_pose_estimator import HeadPoseEstimator

def main():
    face_detector = FaceDetector("res10_300x300_ssd_iter_140000.caffemodel", "deploy.prototxt.txt")
    head_pose_estimator = HeadPoseEstimator()

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

        cv2.putText(frame, f"Faces: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        if alert:
            cv2.putText(frame, "⚠️ Not Looking Straight!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Face + Head Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
