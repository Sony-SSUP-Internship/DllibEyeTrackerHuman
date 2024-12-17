import cv2
import numpy as np 
import dlib


cap = cv2.VideoCapture(0)

fd = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    ret, frame = cap.read()
    flip = cv2.flip(frame,1)
    gray = cv2.cvtColor(flip,cv2.COLOR_BGR2GRAY)

    faces = fd(gray)
    

    for face in faces:
        topl,topr = face.left(),face.top()
        botl,botr = face.right(),face.bottom()

        cv2.rectangle(flip,(topl,topr),(botl,botr),(0,255,0),5)

        for i in range(60):
            markers = predictor(gray,face)
            x = markers.part(i).x
            y = markers.part(i).y

            cv2.circle(flip,(x,y),3,(0,255,0),2)

            #Crop Out eyes
            


    cv2.imshow('feed',gray)



    if(cv2.waitKey(2) == ord('a')):
        break