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
COLORS = np.random.uniform(0, 255, size=(300, 3))

#create instance of SORT
mot_tracker = Sort() 

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, obj_id, confidence, x, y, x_plus_w, y_plus_h):
    
    #label = str(classes[class_id])+"-"+str(int(obj_id))
    label = str(int(obj_id))

    color = COLORS[int(obj_id)]
  
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

input_movie = cv2.VideoCapture("/users/pivotalit/downloads/CC_30secs.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('/users/pivotalit/downloads/CC_30secs.avi', fourcc, 25.0, (1280, 720))

# read pre-trained model and config file
net = cv2.dnn.readNet(weights, config)

frame_number = 0
while frame_number<(length-1):
    
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
    if ret:    
        Width = frame.shape[1]
        Height = frame.shape[0]
        scale = 0.00392
        #scale = 0.01
      
        # create input blob 
        blob = cv2.dnn.blobFromImage(frame, scale, (416,416), (0,0,0), True, crop=False)
    
        # set input blob for the network
        net.setInput(blob)
    
        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))
    
        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.4
        nms_threshold = 0.4
    
        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
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
        
        count_of_people=0
        for i in detections:
            i=i[0]
            if (str(classes[class_ids[i]])=='person'):
                count_of_people+=1
                
        
        if count_of_people==0:
           count_of_people=1
           
        dets=np.zeros(shape=(count_of_people,5))
        j=0
        for i in detections:
            i=i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            classid=class_ids[i]
            sortinput=[x,y,x+w,y+h,confidences[i]]
            if (str(classes[class_ids[i]])=='person'):
               dets[j]=sortinput
               j+=1
        print('Count of People:',count_of_people)
        tracked_objects = mot_tracker.update(dets)    
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            draw_bounding_box(frame, int(cls_pred), obj_id, confidences[i], int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
 
        # save output image to disk
        print("Writing frame {} / {}".format(frame_number, length))
        output_movie.write(frame)

# release resources
cv2.destroyAllWindows()

            
    
    