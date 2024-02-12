import cv2
import numpy as np
import scipy.stats as ss
import math

CASCADE_PATH='resources/cascade.xml'
FRAME_WIDTH=1280
FRAME_HEIGHT=280
IMAGE_SIZE_SCALE=2 #Factor to downsample frames to increase processing speed
DIFF_FACTOR=1.5 #Maximum differences between confidences where we take box closest to center

#[100,42]

class FeatureDetector:
    def __init__(self):
        #Constructor

        ######Load the cascade
        print("Entered Constructor")
        self.eye_model=cv2.CascadeClassifier(CASCADE_PATH)

        ######Erosion/Dilation kernel
        self.erode_kernel=np.ones((3,3),np.uint8)
        self.dilate_kernel=np.ones((3,3),np.uint8)

        #####Blob detection parameters setup
        blob_params=cv2.SimpleBlobDetector_Params()
        # Set Circularity filtering parameters 
        blob_params.filterByCircularity = True 
        blob_params.minCircularity = 0.76
        
        # Set Convexity filtering parameters 
        blob_params.filterByConvexity = True
        blob_params.minConvexity = 0.13
            
        # Set inertia filtering parameters 
        blob_params.filterByInertia = True
        blob_params.minInertiaRatio = 0.01
        self.blob_detector=cv2.SimpleBlobDetector_create(blob_params)



    def detectEyes(self,frame):

        #####Pre-processing steps#####
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame_gray_eq=cv2.equalizeHist(frame_gray)

        #Resize Image to speed up processing
        frame_small=cv2.resize(frame_gray,(int(FRAME_WIDTH/IMAGE_SIZE_SCALE),int(FRAME_HEIGHT/IMAGE_SIZE_SCALE)),interpolation=cv2.INTER_LINEAR)


        #####Haar cascade detection####
        eye_obj=self.eye_model.detectMultiScale3(frame_gray_eq,scaleFactor=1.1,minNeighbors=3,minSize=(150,63),outputRejectLevels=True)
        eye_rects=eye_obj[0]
        eye_weights=eye_obj[2] #The weights are the confidences of the corresponding bounding boxes
        

        ###Pupil detection to refine eye detection###
        #Thresholding image first to detect pupils
        _,thresh=cv2.threshold(frame_small,38,255,cv2.THRESH_BINARY)

        #Dialating then eroding the image
        thresh=cv2.erode(thresh,self.erode_kernel,iterations=1)
        thresh=cv2.dilate(thresh,self.dilate_kernel,iterations=1)

        #Blob detection
        blobs=self.blob_detector.detect(thresh)
        blank = np.zeros((1, 1))
        blob_img=cv2.drawKeypoints(thresh,blobs,blank,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('Blobs',blob_img)     


        #Rescale blobs to correct screen size
        blobs_list=[]
        for blob in blobs:
            blobs_list.append([blob.pt[0]*IMAGE_SIZE_SCALE,blob.pt[1]*IMAGE_SIZE_SCALE])

        boxes=self.filterDetections(eye_rects,eye_weights,blobs_list)








        #frame_blurred=cv2.medianBlur(frame_gray,7)
        #frame_blurred=cv2.blur(frame_gray,(3,3))
        #Detect circles with the Hough transform
        #rows=frame_gray.shape[0]
        #circles = cv2.HoughCircles(frame_blurred, cv2.HOUGH_GRADIENT, 1, rows / 8, param1=50, param2=30,minRadius=1, maxRadius=30)
        '''
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(frame, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(frame, center, radius, (255, 0, 255), 3)
        '''


        print("Worked")
    
        for ((x, y, w, h),weight) in zip(eye_rects,eye_weights): 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2) 
            cv2.putText(frame,str(weight),(int(x), int(y) - 10),fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale = 0.6,color = (255, 0, 0),thickness=2)
        for (x, y, w, h) in boxes: 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0,255), thickness=2) 
  
        cv2.imshow('Detected faces', frame) 
        cv2.waitKey(0)  


    def filterDetections(self,eye_rects,eye_weights,blobs_list):
        right_boxes=[]
        left_boxes=[]
        right_weights=[]
        left_weights=[]

        count=-1

        for (x, y, w, h) in eye_rects:
            count+=1
            if x<(FRAME_WIDTH/2) and (x+w)>(FRAME_WIDTH/2): #The box is invalid as it overlays over both left/right sides
                continue
            else:
                if x<=(FRAME_WIDTH/2):
                    left_boxes.append([x,y,w,h])
                    left_weights.append(eye_weights[count])
                elif x>(FRAME_WIDTH/2):
                    right_boxes.append([x,y,w,h])
                    right_weights.append(eye_weights[count])
        


        #Find blob closest to center of highest confidence box
        left_highest=[]
        right_highest=[]
        if left_boxes:
            max_left_ind=np.argmax(left_weights)
            left_highest=left_boxes[max_left_ind]
        if right_boxes:
            max_right_ind=np.argmax(right_weights)
            right_highest=right_boxes[max_right_ind]

        left_blob_distances=[]
        right_blob_distances=[]
        right_blob=[]
        left_blob=[]
        left_blobs=[]
        right_blobs=[]

        if blobs_list:
            for blob in blobs_list:
                x=blob[0]
                y=blob[1]

                if left_highest and x<=(FRAME_WIDTH/2):
                    left_blob_distances.append(math.sqrt((x-(left_highest[0]+left_highest[2]/2))**2+(y-(left_highest[1]+left_highest[3]/2))**2))
                    left_blobs.append(blob)
                if right_highest and x>(FRAME_WIDTH/2):
                    right_blob_distances.append(math.sqrt((x-(right_highest[0]+right_highest[2]/2))**2+(y-(right_highest[1]+right_highest[3]/2))**2))
                    right_blobs.append(blob)
            if left_blob_distances:
                left_blob_ind=np.argmin(left_blob_distances)
                left_blob=left_blobs[left_blob_ind]
            if right_blob_distances:
                right_blob_ind=np.argmin(right_blob_distances)
                right_blob=right_blobs[right_blob_ind]


        #Find distance of each candidate box to blob center
        right_distances=[]
        left_distances=[]
        if left_boxes and left_blob:
            for (x, y, w, h) in left_boxes:
                left_distances.append(math.sqrt((left_blob[0]-(x+w/2))**2+(left_blob[1]-(y+h/2))**2))

        if right_boxes and right_blob:
            for (x, y, w, h) in right_boxes:
                right_distances.append(math.sqrt((right_blob[0]-(x+w/2))**2+(right_blob[1]-(y+h/2))**2))
        

    
        #Finding rank of each the distances and weights, and adding them to find the appropriate box
        boxes=[]
        #Doing left box first
        if left_boxes:
            if left_distances:
                dist_rank=len(left_distances)+1-ss.rankdata(left_distances).astype(int) #Returns array with values corresponding to distances with SMALLEST values
                weight_rank=ss.rankdata(left_weights).astype(int) #Returns array with highest integer equal to heights confidence
                total_score=dist_rank+weight_rank
                total_score=np.array(total_score)
                max_inds=np.where(total_score==max(total_score))   #Get max score
                if len(max_inds[0])==1:
                    print("Entered Contemplation")
                    boxes.append(left_boxes[max_inds[0][0]])
                else: #We have more than 1 candidate, we take the box closest to center if it is within DIFF_FACTOR of the highest confedence score
                    #Find distances index sorted from least to greatest
                    cond_sort_inds=np.argsort(left_distances)
                    bool_check=False
                    for ind in cond_sort_inds:
                        curr_weight=left_weights[ind]
                        if abs(max(left_weights)-curr_weight)<=DIFF_FACTOR:
                            boxes.append(left_boxes[ind])
                            bool_check=True
                            break
                    if bool_check==False:
                        boxes.append(left_highest)

            else:
                boxes.append(left_highest)

        if right_boxes:
            if right_distances:
                dist_rank=len(right_distances)+1-ss.rankdata(right_distances).astype(int) #Returns array with values corresponding to distances with SMALLEST values
                weight_rank=ss.rankdata(right_weights).astype(int) #Returns array with highest integer equal to heights confidence
                total_score=dist_rank+weight_rank
                total_score=np.array(total_score)
                max_inds=np.where(total_score==total_score.max())   #Get max score
                #max_inds=list(max_inds)
                if len(max_inds[0])==1:
                    print("Entered Contemplation")
                    boxes.append(right_boxes[max_inds[0][0]])
                else:
                    cond_sort_inds=np.argsort(right_distances)
                    bool_check=False
                    for ind in cond_sort_inds:
                        curr_weight=right_weights[ind]
                        if abs(max(right_weights)-curr_weight)<=DIFF_FACTOR:
                            boxes.append(right_boxes[ind])
                            bool_check=True
                            break
                    if bool_check==False:
                        boxes.append(right_highest)

            else:
                boxes.append(right_highest)

        return boxes


        