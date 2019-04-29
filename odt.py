#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 18:29:39 2019

@author: pivotalit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:10:45 2019

@author: pivotalit
"""

import cv2
import numpy as np
from sort import *

weights='/users/pivotalit/downloads/MHA_Demo/yolov3.weights'
config='/users/pivotalit/downloads/MHA_Demo/yolov3.cfg'
labels='/users/pivotalit/downloads/MHA_Demo/yolov3.txt'

classes = None
with open(labels, 'r') as f:
     classes = [line.strip() for line in f.readlines()]
     
# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#create instance of SORT
#mot_tracker = Sort() 

def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)
  
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, obj_id, confidence, x, y, x_plus_w, y_plus_h):
    
    label = str(classes[class_id])+"-"+str(int(obj_id))

    color = COLORS[int(obj_id)]
  
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

input_movie = cv2.VideoCapture("/users/pivotalit/downloads/Walking_Office_People.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('/users/pivotalit/downloads/Walking_Office_People_mine.avi', fourcc, 25.0, (1280, 720))

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# read pre-trained model and config file
net = cv2.dnn.readNet(weights, config)

# cv2 multitracker initialization
multiTracker = cv2.MultiTracker_create()
tracker = OPENCV_OBJECT_TRACKERS["csrt"]()

strg=list()
number_of_people=1
frame_number = 0
while frame_number<10:
    
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
    if ret:    
        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392
    
        # create input blob 
        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=True)
    
        # set input blob for the network
        net.setInput(blob)
    
        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))
    
        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
    
        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence >= conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                
        # apply non-max suppression
        detections = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in detections:
            i=i[0]
            if (str(classes[class_ids[i]])=='person'):
               box=list(boxes[i])
               ibox=list(box)
               ibox[2]=ibox[0]+ibox[2]
               ibox[3]=ibox[1]+ibox[3]

               bbox=(box[0],box[1],box[2],box[3])
               retval=multiTracker.getObjects()
               tracked=False
               for k in range(len(retval)):
                   rbox=list(retval[k])
                   rbox[2]=rbox[0]+rbox[2]
                   rbox[3]=rbox[1]+rbox[3]
                   iou_res=iou(ibox,rbox)
                   if iou_res>=0.3:
                      tracked=True
               if not tracked:       
                  multiTracker.add(cv2.TrackerBoosting_create(), frame, bbox)
                  strg.append(classes[class_ids[i]]+'_'+str(number_of_people))
                  number_of_people+=1
#print(number_of_people)
#cv2.imshow('img',frame)
#cv2.waitKey(0)
#        count_of_people=0
#        for i in detections:
#            i=i[0]
#            if (str(classes[class_ids[i]])=='person'):
#                count_of_people+=1
#                
#        
#        if count_of_people==0:
#           count_of_people=1
           
#        if count_of_people==2:
#           for i in detections:
#            i=i[0]
#            if (str(classes[class_ids[i]])=='person'):
#               print(class_ids,confidences,detections)
#               
#        dets=np.zeros(shape=(count_of_people,6))
#        j=0
#        for i in detections:
#            i=i[0]
#            box = boxes[i]
#            x = box[0]
#            y = box[1]
#            w = box[2]
#            h = box[3]
#            classid=class_ids[i]
#            sortinput=[x,y,x+w,y+h,confidences[i],classid]
#            if (str(classes[class_ids[i]])=='person'):
#               dets[j]=sortinput
#               j+=1
#        
#        tracked_objects = mot_tracker.update(dets)    
#        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
#            draw_bounding_box(frame, int(cls_pred), obj_id, confidences[i], int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
#
#print(strg)              
while frame_number<(length-1):
    ret, frame = input_movie.read()
    frame_number += 1
    if ret: 
      (success, newboxes) = multiTracker.update(frame)
      if success:
         for i,nbox in enumerate(newboxes):
              (x, y, w, h) = [int(v) for v in nbox]
              cv2.rectangle(frame, (x,y), (x+w , y+h), COLORS[i], 1)
              font = cv2.FONT_HERSHEY_DUPLEX
              cv2.putText(frame, strg[i], (x, w+100), font, .4, (0, 0, 0), 1)
      print("Writing frame {} / {}".format(frame_number, length))
      output_movie.write(frame)

# release resources
cv2.destroyAllWindows()

            
    
    