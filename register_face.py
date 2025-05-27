import cv2
from face_utils import get_face_descriptor, save_encoding

def register_face(name):
    cap = cv2.VideoCapture(0)
    print("📸 Please look at the camera to register your face...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Register - Press 's' to save, 'q' to quit", frame)

        key = cv2.waitKey(1)
        if key == ord("s"):
            descriptor = get_face_descriptor(frame)
            if descriptor is not None:
                save_encoding(name, descriptor)
                print("✅ Face registered successfully.")
                break
            else:
                print("❌ Make sure only one face is visible.")
        elif key == ord("q"):
            print("❌ Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    name = input("Enter your name or ID: ")
    register_face(name)
