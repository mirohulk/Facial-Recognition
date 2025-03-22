import cv2
import numpy as np
import face_recognition
from camera import Camera
import time
import math

# Precise SolvePnP head pose estimation
def get_head_pose(landmarks, img_size):
    image_points = np.array([
        landmarks["nose_tip"][len(landmarks["nose_tip"])//2],
        landmarks["chin"][len(landmarks["chin"])//2],
        landmarks["left_eye"][len(landmarks["left_eye"])//2],
        landmarks["right_eye"][len(landmarks["right_eye"])//2],
        landmarks["nose_bridge"][0],
        landmarks["nose_bridge"][-1]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (0.0, 150.0, -125.0),       # Nose bridge (top)
        (0.0, 70.0, -100.0)         # Nose bridge (bottom)
    ])

    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [angle[0] for angle in euler_angles]

    # Adjust pitch and yaw clearly based on your camera orientation
    pitch = -pitch
    yaw = yaw
    roll = roll

    return yaw, pitch, roll


def capture_faces(username, angles_required=5):
    directions = [
        ("Look Straight", lambda yaw, pitch: abs(yaw) < 10 and abs(pitch) < 10),
        ("Turn Left", lambda yaw, pitch: yaw < -20),
        ("Turn Right", lambda yaw, pitch: yaw > 20),
        ("Look Up", lambda yaw, pitch: pitch < -10),
        ("Look Down", lambda yaw, pitch: pitch > 12),
    ]


    cam = Camera(width=1280, height=720).start()
    collected_encodings = []
    captured = 0
    last_capture_time = 0
    feedback_timer = 0

    print("[INFO] Starting advanced facial registration.")

    while captured < angles_required:
        frame = cam.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)

        display_frame = frame.copy()
        h, w = frame.shape[:2]
        overlay = display_frame.copy()

        prompt_text, condition = directions[captured]

        valid_pose = False
        if landmarks_list:
            landmarks = landmarks_list[0]
            yaw, pitch, roll = get_head_pose(landmarks, rgb_frame.shape)
            pitch = -pitch  # fix inversion

            valid_pose = condition(yaw, pitch)
            color = (100, 220, 100) if valid_pose else (230, 230, 230)

            # Face rectangle
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)

            face_encodings = face_recognition.face_encodings(rgb_frame, [face_locations[0]])

            if face_encodings and valid_pose and (time.time() - last_capture_time > 2):
                collected_encodings.append(face_encodings[0])
                captured += 1
                last_capture_time = time.time()
                feedback_timer = time.time()
                print(f"[INFO] Captured '{prompt_text}' successfully.")

        # Modern overlay box
        box_height = 110
        cv2.rectangle(overlay, (0, 0), (w, box_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)

        # Rounded rectangle effect (UI enhancement)
        radius = 20
        cv2.rectangle(display_frame, (radius, 0), (w-radius, box_height), (30, 30, 30), -1)
        cv2.circle(display_frame, (radius, radius), radius, (30, 30, 30), -1)
        cv2.circle(display_frame, (w-radius, radius), radius, (30, 30, 30), -1)

        # Clear antialiased text
        text_size = cv2.getTextSize(prompt_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (w - text_size[0]) // 2
        cv2.putText(display_frame, prompt_text, (text_x, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

        # Real-time angles clear indication
        angle_text = f'Yaw: {int(yaw) if landmarks_list else "--"}  |  Pitch: {int(pitch) if landmarks_list else "--"}  |  Roll: {int(roll) if landmarks_list else "--"}'
        angle_size = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        angle_x = (w - angle_size[0]) // 2
        cv2.putText(display_frame, angle_text, (angle_x, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

        # Feedback for capture (soft flash)
        if time.time() - feedback_timer < 0.3:
            flash_alpha = 0.3 - (time.time() - feedback_timer)
            flash_overlay = display_frame.copy()
            cv2.rectangle(flash_overlay, (0, 0), (w, h), (100, 220, 100), -1)
            cv2.addWeighted(flash_overlay, flash_alpha, display_frame, 1-flash_alpha, 0, display_frame)

        # No face detected message
        if not landmarks_list:
            no_face_text = "Face not detected. Please align your face."
            no_face_size = cv2.getTextSize(no_face_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            no_face_x = (w - no_face_size[0]) // 2
            cv2.putText(display_frame, no_face_text, (no_face_x, box_height + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70,70,250), 2, cv2.LINE_AA)

        cv2.imshow("Modern Face Registration", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

    if len(collected_encodings) == angles_required:
        np.save(f'encodings/{username}_encodings.npy', collected_encodings)
        print("[INFO] Encodings saved successfully.")
    else:
        print("[WARN] Registration incomplete, please retry.")
