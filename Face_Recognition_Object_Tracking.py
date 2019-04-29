#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:13:44 2019

@author: pivotalit
"""
import imutils
import face_recognition
import cv2


# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("/users/pivotalit/downloads/trump_walking_Around_12fps.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('/users/pivotalit/downloads/output.avi', fourcc, 12, (1280, 720))

# Load some sample pictures and learn how to recognize them.

lmm_image = face_recognition.load_image_file("/users/pivotalit/downloads/trump_try.jpg")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]
#lmm_image = imutils.resize(lmm_image, width=600)
#face_loc=face_recognition.face_locations(lmm_image)
#(top,right,bottom,left)=face_loc[0]
#cv2.imshow('trump',lmm_image[top:bottom,left:right])
#cv2.waitKey(1)
#print(lmm_face_encoding)
#al_image = face_recognition.load_image_file("/users/pivotalit/downloads/alex-lacamoire.png")
#al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
    lmm_face_encoding
#    al_face_encoding
]

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

trackers = cv2.MultiTracker_create()


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0
name=" "
tracked=False
matched=False
length_of_movie=input_movie.get(cv2.CAP_PROP_FRAME_COUNT)
while frame_number<(length_of_movie-1):
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
    

    # Quit when the input video file ends
    #if not ret:
    #    break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    #frame = imutils.resize(frame, width=600)
    #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
#    for i in range(len(face_locations)):
#       (top,right,bottom,left)=face_locations[i]
#       cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),1)
#       cv2.imshow('img1',frame)
#       cv2.waitKey(1)
    face_encodings = face_recognition.face_encodings(rgb_frame,face_locations)

    #print(face_encodings)
    face_names = []
    if not matched:
        #face_locations = face_recognition.face_locations(rgb_frame)
        #face_encodings = face_recognition.face_encodings(rgb_frame,face_locations)

        j=0
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
 #           (top,right,bottom,left)=face_locations[0]
 #           cv2.rectangle(frame,(top,left),(bottom,right),(0,0,255),2)
               
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.66)
    
            
            #for i in range(len(match)):
            if match[0]:
                    matched=True
                    name="Trump"
                    (top,right,bottom,left)=face_locations[2]
                    #face_image = frame[top:bottom, left:right]
                    box=(left,top,(right-left),(bottom-top)) #reformat to x,y,w,h
                    if not tracked:
                        tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
                        trackers.add(tracker, frame, box)
                        tracked = True
            j+=1            
                   #frame[dlib.rectangle(left, top, right, bottom)]
#              
    if tracked:
        (success, boxes) = trackers.update(frame)
        if success:
            for box in boxes:
                   (x, y, w, h) = [int(v) for v in box]
                   cv2.rectangle(frame, (x,y), (x+w , y+h), (0, 255, 0), 2)
                   font = cv2.FONT_HERSHEY_DUPLEX
                   cv2.putText(frame, name, (x+w, y+h), font, .5, (255, 255, 255), 2)
                   #cv2.imshow('trump',frame)
               #cv2.waitKey(3)
    
   # cv2.imshow('Video', frame)   
   # if cv2.waitKey(1) & 0xFF == ord('q'):
   #     break        
    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

 #All done!
input_movie.release()
cv2.destroyAllWindows()
