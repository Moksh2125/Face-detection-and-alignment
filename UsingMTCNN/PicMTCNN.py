from mtcnn import MTCNN
import cv2
import math

detector = MTCNN()

img = cv2.imread('images/Amitabh Bachchan_20.jpg')

faces = detector.detect_faces(img)

for face in faces:

    # Face Box
    x, y, width, height = face['box']

    # Eye Coordinates
    leftEyeX, leftEyeY = face['keypoints']['left_eye']
    rightEyeX, rightEyeY = face['keypoints']['right_eye']

    # Other Landmarks
    noseX, noseY = face['keypoints']['nose']
    mouthRightX, mouthRightY = face['keypoints']['mouth_right']
    mouthLeftX, mouthLeftY = face['keypoints']['mouth_left']

    # -----------------------------------
    # ALIGNMENT LOGIC
    # -----------------------------------

    dx = rightEyeX - leftEyeX
    dy = rightEyeY - leftEyeY

    angle = math.degrees(math.atan2(dy, dx))

    # Alignment Threshold
    if abs(angle) < 5:
        lineColor = (0, 255, 0)   # Green
        status = "Aligned"
    else:
        lineColor = (0, 0, 255)   # Red
        status = "Tilted"

    # -----------------------------------
    # DRAWINGS
    # -----------------------------------

    # Face Box
    cv2.rectangle(
        img,
        pt1=(x, y),
        pt2=(x + width, y + height),
        color=(255, 0, 0),
        thickness=2
    )

    # Landmarks
    cv2.circle(img, (leftEyeX, leftEyeY), 2, (255, 0, 0), 2)
    cv2.circle(img, (rightEyeX, rightEyeY), 2, (255, 0, 0), 2)
    cv2.circle(img, (noseX, noseY), 2, (255, 0, 0), 2)
    cv2.circle(img, (mouthRightX, mouthRightY), 2, (255, 0, 0), 2)
    cv2.circle(img, (mouthLeftX, mouthLeftY), 2, (255, 0, 0), 2)

    # Eye Alignment Line
    cv2.line(
        img,
        (leftEyeX, leftEyeY),
        (rightEyeX, rightEyeY),
        lineColor,
        2
    )

    # Angle Text
    cv2.putText(
        img,
        f"{status} : {angle:.2f} deg",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        lineColor,
        2
    )

cv2.imshow('window', img)

cv2.waitKey(0)

cv2.destroyAllWindows()