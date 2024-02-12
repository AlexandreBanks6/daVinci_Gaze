# Utility to go through eye videos and grab frames for HAAR cascade training
import os
import cv2
import random

VIDEO_DATA_PATH='D:/Alexandre_EyeGazeProject_Extra/eyecorner_userstudy2_converted'
DST_PATH='D:/Alexandre_EyeGazeProject_Extra/Haar_Cascade_Training/test_images_raw'
VIDEO_FILE_NAMES=['eyeVideo_Calib_Init.avi','eyeVideo_Calib_Comp_Lift1_8point.avi','eyeVideo_Calib_Comp_Lift1_dot.avi',\
                  'eyeVideo_Calib_Comp_Lift2_8point.avi','eyeVideo_Calib_Comp_Lift3_8point.avi','eyeVideo_Calib_Comp_Rotate.avi']

FRAMES_PER_VIDEO=2

subdirs=os.listdir(VIDEO_DATA_PATH)

frame_count=10
for entry in subdirs:
    if entry[0]=='P': #We have a participant
        #print(entry)
        for filename in VIDEO_FILE_NAMES:

            vid_path=VIDEO_DATA_PATH+'/'+entry+'/'+'EyeGaze_Data'+'/'+filename
            video=cv2.VideoCapture(vid_path)
            if(video.isOpened()==False):
                print("video "+vid_path+" cannot be opened")
                continue
            num_frames=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            random_frames=random.sample(range(num_frames),FRAMES_PER_VIDEO)

            for frame_num in random_frames:
                video.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
                ret,frame=video.read()
                if not ret:
                    print('frame not opened correctly')
                    continue
                new_image_name=DST_PATH+'/'+'Frame'+str(frame_count)+'.jpg'
                frame_count+=1
                cv2.imwrite(new_image_name,frame)

