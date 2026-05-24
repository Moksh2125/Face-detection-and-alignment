from mtcnn import MTCNN
import cv2

detector = MTCNN()
img = cv2.imread('gp2.jpg')
faces = detector.detect_faces(img)
print(faces)

for face in faces:
    x, y, width, height = face['box']
    leftEyeX, leftEyeY = face['keypoints']['left_eye']
    rightEyeX, rightEyeY = face['keypoints']['right_eye'] 
    noseX, noseY = face['keypoints']['nose'] 
    mouthRightX, mouthRightY = face['keypoints']['mouth_right'] 
    mouthLeftX, mouthLeftY = face['keypoints']['mouth_left'] 

    cv2.rectangle(img, pt1=(x, y), pt2=(x+width, y+height), color=(255,0,0), thickness=2)
    cv2.circle(img, center=(leftEyeX, leftEyeY), color=(255, 0, 0), thickness=1, radius=2)
    cv2.circle(img, center=(rightEyeX, rightEyeY), color=(255, 0, 0), thickness=1, radius=2)
    cv2.circle(img, center=(noseX, noseY), color=(255, 0, 0), thickness=1, radius=2)
    cv2.circle(img, center=(mouthRightX, mouthRightY), color=(255, 0, 0), thickness=1, radius=2)
    cv2.circle(img, center=(mouthLeftX, mouthLeftY), color=(255, 0, 0), thickness=1, radius=2)

cv2.imshow('window', img)

cv2.waitKey(0)