import os
import cv2

BASE_PATH='D:/Alexandre_EyeGazeProject_Extra/Haar_Cascade_Training/'


with open(BASE_PATH+'neg.txt','w') as f:
    for filename in os.listdir(BASE_PATH+'negative/'):
        f.write(BASE_PATH+'negative/'+filename+'\n')