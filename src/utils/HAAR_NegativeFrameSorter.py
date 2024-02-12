import os
import cv2

#Loop through videos and looks for frames without eyes in them for negative examples
VIDEO_DATA='D:/Alexandre_EyeGazeProject_Extra/Haar_Cascade_Training/negative_videos_raw/NegativeVideo.avi'
DST_PATH='D:/Alexandre_EyeGazeProject_Extra/Haar_Cascade_Training/negative/'

video=cv2.VideoCapture(VIDEO_DATA)
if(video.isOpened()==False):
    print("video cannot be opened")

frame_count=1

success,frame=video.read()

while success:
    cv2.imshow('Current Frame',frame)
    key=cv2.waitKey(0) #Waits for a key press
    if key==ord('t'):
        write_path=DST_PATH+'Frame'+str(frame_count)+'.jpg'
        cv2.imwrite(write_path,frame)
        frame_count+=1
    success,frame=video.read()