import cv2
import mediapipe as mp
import math

# -----------------------------
# MediaPipe Face Mesh Setup
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Webcam Setup
# -----------------------------

cap = cv2.VideoCapture(0)

# -----------------------------
# Main Loop
# -----------------------------

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Get frame dimensions
    height, width, _ = frame.shape

    # ------------------------------------
    # Convert BGR -> RGB
    # ------------------------------------
    # MediaPipe expects RGB images
    # OpenCV captures in BGR
    # ------------------------------------

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    # ------------------------------------
    # Face Landmark Detection
    # ------------------------------------

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ------------------------------------
            # Eye Landmark Indices
            # ------------------------------------
            # Left Eye Outer Corner  -> 33
            # Right Eye Outer Corner -> 263
            # ------------------------------------

            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            # Convert normalized coordinates to pixels

            leftEyeX = int(left_eye.x * width)
            leftEyeY = int(left_eye.y * height)

            rightEyeX = int(right_eye.x * width)
            rightEyeY = int(right_eye.y * height)

            # ------------------------------------
            # Angle Calculation
            # ------------------------------------

            dx = rightEyeX - leftEyeX
            dy = rightEyeY - leftEyeY

            angle = math.degrees(math.atan2(dy, dx))

            # ------------------------------------
            # Alignment Logic
            # ------------------------------------

            if abs(angle) < 5:
                color = (0, 255, 0)
                status = "Aligned"
            else:
                color = (0, 0, 255)
                status = "Tilted"

            # ------------------------------------
            # Draw Eye Points
            # ------------------------------------

            cv2.circle(frame, (leftEyeX, leftEyeY), 3, (255, 0, 0), -1)
            cv2.circle(frame, (rightEyeX, rightEyeY), 3, (255, 0, 0), -1)

            # ------------------------------------
            # Draw Alignment Line
            # ------------------------------------

            cv2.line(
                frame,
                (leftEyeX, leftEyeY),
                (rightEyeX, rightEyeY),
                color,
                2
            )

            # ------------------------------------
            # Display Angle + Status
            # ------------------------------------

            cv2.putText(
                frame,
                f"{status} : {angle:.2f} deg",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

    # ------------------------------------
    # Show Frame
    # ------------------------------------

    cv2.imshow("Face Alignment - MediaPipe", frame)

    # Press X to Exit
    if cv2.waitKey(10) & 0xFF == ord('x'):
        break

# -----------------------------
# Cleanup
# -----------------------------

cap.release()
cv2.destroyAllWindows()