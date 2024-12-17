import cv2
import dlib
from fc import face
from eye import Eye
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    _face = face('shape_predictor_68_face_landmarks.dat')
    _eye = Eye()

    rats = []


    while True:
        ret,frame = cap.read()
        initial_time = time.time
        grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)



        retn  = _face.GetFacialCoordinates(frame)

        for f in retn:
            markers = _face.getEyeMarkerCoordinates(grey,f)

            eye_left,elr = _eye.isolateEye(grey,np.array(markers[0]),'left')
            eye_right,err = _eye.isolateEye(grey,np.array(markers[1]),'right ')

            

            ration = (elr + err)/2
            rats.append(ration)

            gazeEstimation = _eye.EstimateGaze(ration=ration)

            final_time = time.time

            cv2.putText(frame,gazeEstimation,(50,100),2,2,(0,0,255),3)

        cv2.imshow('frame',frame)

        if(cv2.waitKey(2) == ord('a')):
            break 

    plt.plot(rats)
    plt.show()

    



