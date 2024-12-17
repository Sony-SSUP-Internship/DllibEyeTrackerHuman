import dlib
import numpy as np 
import cv2
import math


class Eye():
    def __init__(self):
        pass

    def isolateEye(self,frame,coords,side):

        height, width = frame.shape[:2]
        black_frame = np.zeros((height, width),dtype=np.uint8)
        mask = np.full((height,width),255,dtype=np.uint8)

        MARGIN_THRESHOLD_VALUE = 3

        cv2.fillPoly(mask,[coords],(0,0,0))
        mask2 = cv2.bitwise_not(mask)
        eye = cv2.bitwise_not(black_frame,frame.copy(),mask=mask2)
        frm = cv2.bitwise_and(frame,frame,mask=mask2)


        #Provide Margins to coordinates
        minx = np.min(coords[:,0]) - MARGIN_THRESHOLD_VALUE
        maxx = np.max(coords[:,0]) + MARGIN_THRESHOLD_VALUE
        miny = np.min(coords[:,1]) - MARGIN_THRESHOLD_VALUE
        maxy = np.max(coords[:,1]) + MARGIN_THRESHOLD_VALUE


        eye2 = frm[miny:maxy,minx:maxx]
        eye_pre_thresh = frame[miny:maxy,minx:maxx]


        mean,std = cv2.meanStdDev(eye2)

        print(mean[0][0])
        print(std[0][0])

        idthresh = mean - (std * 10)


        #OTSU'S BINARIZATION
        eye2 = cv2.GaussianBlur(eye2,(3,3),0)
        eye2 = cv2.resize(eye2 , None,fx=5,fy=5)

        _,threshold = cv2.threshold(eye2,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        #eye2 = cv2.resize(threshold , None,fx=5,fy=5)
        eye_pre_thresh = cv2.resize(eye_pre_thresh,None,fx=5,fy=5)

        thresholdheight,thresholdwidth = threshold.shape

        eps = 0.001

        leftpart = threshold[0:thresholdheight,0:int(thresholdwidth/2)]
        lration = cv2.countNonZero(leftpart) + eps
        rightpart = threshold[0:thresholdheight,int(thresholdwidth/2):thresholdwidth]
        rration = cv2.countNonZero(rightpart) + eps


        print(f"{lration},{rration}")
        nzs = lration/rration
        nzs = round(nzs,2)

        cv2.imshow(f'eye {side}',threshold)
        cv2.imwrite('E:\Projects Continued\DllibEyeTrackerHuman\eyeimgs\imgisolate.png',threshold)


        return eye2,nzs
    
    def EstimateGaze(self,ration):
        if(ration>0.9 and ration<1.8):
            return "Forward"
        
        elif(ration>=1.8):
            return "Left"
        
        elif(ration<0.8):
            return "Right"
        
        else:
            return "Estimating"
        

