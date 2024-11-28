import cv2
import face_recognition
import numpy as np

# Load known face encodings and names
image_paths = ["/Users/personal/Desktop/Photo.jpg"]
known_face_encodings = []
known_face_names = ["Enter Name"]

for image_path in image_paths:
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)
    if face_encodings:
        known_face_encodings.append(face_encodings[0])

# Start video capture
video_capture = cv2.VideoCapture(0)

# Parameters for recognition
frame_reduction_factor = 0.3
face_match_threshold = 0.49

while True:
    # Capture a frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Reduce frame size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=frame_reduction_factor, fy=frame_reduction_factor)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back up face locations to original size
        top = int(top / frame_reduction_factor)
        right = int(right / frame_reduction_factor)
        bottom = int(bottom / frame_reduction_factor)
        left = int(left / frame_reduction_factor)

        # Compare the face with known encodings
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] <= face_match_threshold:
            name = known_face_names[best_match_index]
            rectangle_color = (0, 255, 0)  # Green for match
            status_text = f"Recognized: {name}"
        else:
            name = "Unknown"
            rectangle_color = (0, 0, 255)  # Red for no match
            status_text = "Unrecognized"

        # Draw a rectangle and label on the frame
        cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)
        cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rectangle_color, 2)

    # Display the resulting frame
    cv2.imshow('Facial Recognition', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()