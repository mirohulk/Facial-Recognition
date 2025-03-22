import cv2
import face_recognition
import os
from camera import Camera
import numpy as np

def load_all_encodings(path='encodings/'):
    encodings, usernames = [], []
    for file in os.listdir(path):
        if file.endswith("_encodings.npy"):
            username = file.replace("_encodings.npy", "")
            file_path = os.path.join(path, file)
            if os.path.getsize(file_path) > 0:
                user_encodings = np.load(file_path)
                encodings.extend(user_encodings)
                usernames.extend([username]*len(user_encodings))
            else:
                print(f"[WARNING] Skipped empty file: {file}")
    return encodings, usernames

def recognize_faces():
    known_encodings, known_usernames = load_all_encodings()
    if not known_encodings:
        print("[ERROR] No encodings found. Register first.")
        return

    cam = Camera(width=1280, height=720).start()

    print("[INFO] Basic Facial Recognition Started.")

    while True:
        frame = cam.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0,0), fx=0.25, fy=0.25)

        face_locations = face_recognition.face_locations(small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"
            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_usernames[best_match_index]

            top, right, bottom, left = top*4, right*4, bottom*4, left*4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            status = f"{name}"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, status, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if not face_locations:
            cv2.putText(frame, "No Face Detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Reliable Facial Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
