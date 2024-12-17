import cv2
import dlib

class face:
    def __init__(self,predictor_path):
        self.predictor = dlib.shape_predictor(predictor_path)
        self.facialDetector = dlib.get_frontal_face_detector() 
        self.LEFT_EYE_COORDINATES = [36,37,38,39,40,41]
        self.RIGHT_EYE_COORDINATES = [42,43,44,45,46,47]

    def GetFacialCoordinates(self,frame):
        faces = self.facialDetector(frame)

        return faces
    
    def getEyeMarkerCoordinates(self,frame,detectionfromFacial):
        markers = self.predictor(frame,detectionfromFacial)

        eye_coords_left = []
        eye_coords_right = []

        for pt in self.LEFT_EYE_COORDINATES:
            
            x = markers.part(pt).x
            y = markers.part(pt).y

            eye_coords_left.append((x,y))

        for pt in self.RIGHT_EYE_COORDINATES:
            x = markers.part(pt).x 
            y = markers.part(pt).y

            eye_coords_right.append((x,y))

        

        

        return [eye_coords_left,eye_coords_right]
    










        



