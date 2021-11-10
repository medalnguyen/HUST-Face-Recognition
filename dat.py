import dlib
import os
import cv2 as cv
import numpy as np
import pickle

name = input("Enter your name:")
ref_id = input("Enter id:")

try:
    f = open("data/ref_name.pkl","rb")

    ref_dictt = pickle.load(f)
    f.close()
except:
    ref_dictt = {}
ref_dictt[ref_id] = name


f = open("data/ref_name.pkl","wb")
pickle.dump(ref_dictt,f)
f.close()

try:
    f=open("data/ref_embed.pkl","rb")

    embed_dictt=pickle.load(f)
    f.close()
except:
    embed_dictt={}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:\projects\AI\Face-Recognition\shape_predictor_68_face_landmarks.dat')

cap = cv.VideoCapture(0)

All = list(range(0,68))
RIGHT_EYEBROW = list(range(17,22))
LEFT_EYEBROW = list(range(22,27))
RIGHT_EYE = list(range(36,42))
LEFT_EYE = list(range(42,48))
NOSE = list(range(27, 36))
MOUTH_OUTLINE = list(range(48, 61))
MOUTH_INNER = list(range(61, 68))
JAVLINE = list(range(0,17))

index = All

while True:

    ret, img_frame = cap.read()
    
    img_gray = cv.cvtColor(img_frame, cv.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)
    for face in dets:

        shape = predictor(img_frame, face)

        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])
        list_points = np.array(list_points)

        for i,pt in enumerate(list_points[index]):

            pt_pos = (pt[0], pt[1])
            cv.circle(img_frame, pt_pos, 2, (0,255,0), -1)

        cv.rectangle(img_frame, (face.left(), face.top()),(face.right(),face.bottom()),
        (0,0,255), 3)
    cv.imshow('result', img_frame)
    key = cv.waitKey(1)
    
    if key == ord('q'):
        break
    elif key == ord('1'):
        index = All
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE + MOUTH_INNER
    elif key == ord('6'):
        index = JAVLINE

# os.remove('result/output.avi')
cap.release()