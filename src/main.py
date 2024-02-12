
from utils import FeatureDetector_Class
import os
import cv2



#CONSTANTS
FRAME_PATH='D:/Alexandre_EyeGazeProject_Extra/Haar_Cascade_Training/test_images_raw/'

if __name__=='__main__':
    FeatureDetector=FeatureDetector_Class.FeatureDetector()

    for filename in os.listdir(FRAME_PATH):
        if filename.endswith(".jpg"): 
            frame=cv2.imread(FRAME_PATH+filename)
            FeatureDetector.detectEyes(frame)


