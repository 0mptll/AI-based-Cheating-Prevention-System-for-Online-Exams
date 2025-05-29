from face_utils import get_face_descriptor, load_encoding, compare_encodings

def verify_face_in_frame(frame, registered_name):
    known_encoding = load_encoding(registered_name)
    if known_encoding is None:
        return False, "No registered face found."

    unknown_encoding = get_face_descriptor(frame)
    if unknown_encoding is None:
        return False, "No face or multiple faces detected."

    is_match = compare_encodings(known_encoding, unknown_encoding)
    return is_match, "Match" if is_match else "Mismatch"