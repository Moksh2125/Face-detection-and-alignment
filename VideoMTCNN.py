import cv2
from mtcnn import MTCNN

detector = MTCNN()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, width, height = face['box']
        leftEyeX, leftEyeY = face['keypoints']['left_eye']
        rightEyeX, rightEyeY = face['keypoints']['right_eye'] 
        noseX, noseY = face['keypoints']['nose'] 
        mouthRightX, mouthRightY = face['keypoints']['mouth_right'] 
        mouthLeftX, mouthLeftY = face['keypoints']['mouth_left'] 

        cv2.rectangle(frame, pt1=(x, y), pt2=(x+width, y+height), color=(255,0,0), thickness=2)
        cv2.circle(frame, center=(leftEyeX, leftEyeY), color=(255, 0, 0), thickness=1, radius=2)
        cv2.circle(frame, center=(rightEyeX, rightEyeY), color=(255, 0, 0), thickness=1, radius=2)
        cv2.circle(frame, center=(noseX, noseY), color=(255, 0, 0), thickness=1, radius=2)
        cv2.circle(frame, center=(mouthRightX, mouthRightY), color=(255, 0, 0), thickness=1, radius=2)
        cv2.circle(frame, center=(mouthLeftX, mouthLeftY), color=(255, 0, 0), thickness=1, radius=2)


    cv2.imshow('win', frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cv2.destroyAllWindows()