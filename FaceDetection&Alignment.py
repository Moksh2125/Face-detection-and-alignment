import cv2
import math
from mtcnn import MTCNN

detector = MTCNN()
cap = cv2.VideoCapture(0)  # a stream is set for capturing frame from pc webcam

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, width, height = face['box']
        leftEyeX, leftEyeY = face['keypoints']['left_eye']
        rightEyeX, rightEyeY = face['keypoints']['right_eye'] 
        # noseX, noseY = face['keypoints']['nose'] 
        # mouthRightX, mouthRightY = face['keypoints']['mouth_right'] 
        # mouthLeftX, mouthLeftY = face['keypoints']['mouth_left'] 

        dx = rightEyeX - leftEyeX
        dy = rightEyeY - leftEyeY

        angle = math.degrees(math.atan2(dy, dx))



        cv2.rectangle(frame, pt1=(x, y), pt2=(x+width, y+height), color=(255,0,0), thickness=2)
        cv2.circle(frame, center=(leftEyeX, leftEyeY), color=(255, 0, 0), thickness=1, radius=2)
        cv2.circle(frame, center=(rightEyeX, rightEyeY), color=(255, 0, 0), thickness=1, radius=2)
        
        status = ""
        if abs(angle) > 5:
            color = (0, 0, 255)  # Red for non-aligned faces
            status = "Tilted"
        else:
            color = (0, 255, 0)  # Green for aligned faces
            status = "Aligned"

        cv2.line(
            frame,
            (leftEyeX, leftEyeY),
            (rightEyeX, rightEyeY),
            color,
            2
            )
        
         # Angle Text
        cv2.putText(
            frame,
            f"{status} : {angle:.2f} deg",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        # cv2.circle(frame, center=(noseX, noseY), color=(255, 0, 0), thickness=1, radius=2)
        # cv2.circle(frame, center=(mouthRightX, mouthRightY), color=(255, 0, 0), thickness=1, radius=2)
        # cv2.circle(frame, center=(mouthLeftX, mouthLeftY), color=(255, 0, 0), thickness=1, radius=2)


    cv2.imshow('win', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()