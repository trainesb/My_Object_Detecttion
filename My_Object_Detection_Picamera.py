########################################################################
#
# Author: Ben Traines
# Date: 11/22/2019
# Description: Object detection using a tensorflow model and a Raspberry
# Pi Camera.
#
########################################################################

import os
import cv2
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import RPi.GPIO as GPIO

from PCA9685 import PCA9685

from picamera import PiCamera
from picamera.array import PiRGBArray

from utils import label_map_util
from utils import visualization_utils as vis_util

# Camera constants
CAMERA_IMAGE_WIDTH = 1280
CAMERA_IMAGE_HEIGHT = 720
ZOOM = 50

def zoom(n):
	camera.zoom=(n/100., n/100., 0.5, 0.5)

# Directory of object_detection
sys.path.append('..')

# Name of Model
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Path of the current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph that contains the model
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')

# Path of label map
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

# Map the Label Map indexes to category names
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load TensorFlow model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
		
	sess = tf.Session(graph=detection_graph)
	
##### Define Input and Output Tensors #####

# Input Tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output Tensors are detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Score represents the level of confidence, and shown with the class label
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of detected objects
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize the frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Define Left Box (top left & bottom right)
TL_LEFT = (int(CAMERA_IMAGE_WIDTH*0), int(CAMERA_IMAGE_HEIGHT))
BR_LEFT = (int(CAMERA_IMAGE_WIDTH*0.5), int(CAMERA_IMAGE_HEIGHT*0))

# Define Right Box (top left & bottom right)
TL_RIGHT = (int(CAMERA_IMAGE_WIDTH*0.5), int(CAMERA_IMAGE_HEIGHT))
BR_RIGHT = (int(CAMERA_IMAGE_WIDTH), int(CAMERA_IMAGE_HEIGHT*0))

# Define Bottom Box (top left & bottom right)
TL_BOTTOM = (int(CAMERA_IMAGE_WIDTH*0), int(CAMERA_IMAGE_HEIGHT))
BR_BOTTOM = (int(CAMERA_IMAGE_WIDTH), int(CAMERA_IMAGE_HEIGHT*0.5))

# Define Top Box (top left & bottom right)
TL_TOP = (int(CAMERA_IMAGE_WIDTH*0), int(CAMERA_IMAGE_HEIGHT*0.5))
BR_TOP = (int(CAMERA_IMAGE_WIDTH), int(CAMERA_IMAGE_HEIGHT*0))

# Initialize control variables for arm
detected_left = False
detected_right = False
detected_top = False
detected_bottom = False

left_counter = 0
right_counter = 0
top_counter = 0
bottom_counter = 0

pause = 0
pause_counter = 0

horizontal_angle = 0
vertical_angle = 0

pwm = PCA9685()
pwm.setPWMFreq(50)
pwm.setServoPulse(1, 500)
pwm.setRotationAngle(1, 0)
pwm.setRotationAngle(0, 55)

def arm_detector(frame):
	
	global detected_left, detected_right, detected_top, detected_bottom
	global left_counter, right_counter, top_counter, bottom_counter
	global pause, pause_counter
	global horizontal_angle, vertical_angle, pwm
	
	frame_expanded = np.expand_dims(frame, axis=0)
	
	# Perform detection by running the model
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: frame_expanded})

	# Draw the results (Vizualize)
	vis_util.visualize_boxes_and_labels_on_image_array(
		frame, 
		np.squeeze(boxes), 
		np.squeeze(classes).astype(np.int32), 
		np.squeeze(scores), 
		category_index, 
		use_normalized_coordinates=True, 
		line_thickness=8, 
		min_score_thresh=0.40)
		
	# Draw boxes defining "left", "right", "top", and "bottom"
	cv2.rectangle(frame, TL_LEFT, BR_LEFT, (0, 128, 116), 3)
	cv2.putText(frame, "Left", (TL_LEFT[0]+50, (TL_LEFT[1]-180)), font, 1, (0, 128, 116), 3, cv2.LINE_AA)
		
	cv2.rectangle(frame, TL_RIGHT, BR_RIGHT, (128, 3, 0), 3)
	cv2.putText(frame, "Right", (TL_RIGHT[0]+180, (TL_RIGHT[1]-180)), font, 1, (128, 3, 0), 3, cv2.LINE_AA)
		
	cv2.rectangle(frame, TL_TOP, BR_TOP, (38, 128, 0), 3)
	cv2.putText(frame, "Top", ((TL_TOP[0]+int(CAMERA_IMAGE_WIDTH /2)), TL_TOP[1]-180), font, 1, (38, 128, 0), 3, cv2.LINE_AA)
		
	cv2.rectangle(frame, TL_BOTTOM, BR_BOTTOM, (92, 0, 128), 3)
	cv2.putText(frame, "Bottom", ((TL_BOTTOM[0]+int(CAMERA_IMAGE_WIDTH/2)), TL_BOTTOM[1]-80), font, 1, (92, 0, 128), 3, cv2.LINE_AA)
	
	cv2.putText(frame, "Vertical Angle: {}".format(vertical_angle), (30, 140), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
	cv2.putText(frame, "Horizontal Angle: {}".format(horizontal_angle), (80, 140), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
	
		
	# Check the top detected object (classes[0][0])
	# If the object is a person (1), find its center boxes[0][0]
	if((int(classes[0][0]) == 1) and (pause == 0)):
		
		x = int(((boxes[0][0][1] + boxes[0][0][3]) / 2) * CAMERA_IMAGE_WIDTH)
		y = int(((boxes[0][0][0] + boxes[0][0][2]) /2) * CAMERA_IMAGE_HEIGHT)
			
		# Draw a circle at center of object
		cv2.circle(frame, (x,y), 5, (75,13,180), -1)
		cv2.putText(frame, "x: {} -- y: {}".format(x, y), (30, 110), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
		
			
		# If object is in left box
		if((x > TL_LEFT[0]) and (x < BR_LEFT[0]) and (y < TL_LEFT[1]) and (y > BR_LEFT[1])):
			left_counter += 1
			detected_left = True
			
			### Motor Control ####
			cv2.putText(frame, "Move Left", (30 ,170), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
			
			if(horizontal_angle > 0):
				horizontal_angle -= 1
				pwm.setRotationAngle(0, horizontal_angle)
			else:
				cv2.putText(frame, "Move right!", (30, 200), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

				
		# If object is in right box, increment right counter
		if((x > TL_RIGHT[0]) and (x < BR_RIGHT[0]) and (y < TL_RIGHT[1]) and (y > BR_RIGHT[1])):
			right_counter += 1
			detected_right = True
		
			### Motor Control ####
			cv2.putText(frame, "Move Right", (30, 170), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
			
			if(horizontal_angle < 181):
				horizontal_angle += 1
				pwm.setRotationAngle(0, horizontal_angle)
			else:
				cv2.putText(frame, "Move Left!", (30, 200), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

				
		# If object is in top box, increment top counter
		if((x > TL_TOP[0]) and (x < BR_TOP[0]) and (y < TL_TOP[1]) and (y > BR_TOP[1])):
			top_counter += 1
			detected_top = True
		
			### Motor Control ####
			cv2.putText(frame, "Move Up", (30, 170), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
			
			if(vertical_angle < 20):
				vertical_angle += 1
				pwm.setRotationAngle(0, vertical_angle)
			else:
				cv2.putText(frame, "Move Down!", (30, 200), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

				
		# If object is in bottom box, increment bottom counter
		if((x > TL_BOTTOM[0]) and (x < BR_BOTTOM[0]) and (y < TL_BOTTOM[1]) and (y > BR_BOTTOM[1])):
			bottom_counter += 1
			detected_bottom = True
		
			### Motor Control ####
			cv2.putText(frame, "Move Down", (30, 230), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
			
			if(vertical_angle > 91):
				vertical_angle -= 1
				pwm.setRotationAngle(0, vertical_angle)
			else:
				cv2.putText(frame, "Move up!", (30, 200), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

				
		
	# Draw counter info
	#cv2.putText(frame, 'Detection Counter: ' + str(max(left_counter, right_counter, bottom_counter, top_counter)), (10,100), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
	
	return frame

# Initialize the Picamera and perform object detection
camera = PiCamera()
camera.resolution = (CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT))
rawCapture.truncate(0)

for frame1 in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
	t1 = cv2.getTickCount()
	
	# Get frame and expand it to have the shape: [1, None, None, 3]
	# -> A single-column array, where each item has the pixel RGB value
	frame = np.copy(frame1.array)
	frame.setflags(write=1)
	
	# Pass frame to arm_detectior(frame)
	frame = arm_detector(frame)
	
	# Draw FPS
	cv2.putText(frame, "FPS: {0:.2f}".format(frame_rate_calc), (30,50), font, 1, (255,255,0), 2, cv2.LINE_AA)
	
	# Draw Zoom
	cv2.putText(frame, "Zoom: {}".format(ZOOM), (30, 80), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
	
	# Results have been drawn on the frame, so present it
	cv2.imshow('Object detector', frame)
	
	t2 = cv2.getTickCount()
	time1 = (t2 - t1) / freq
	frame_rate_calc = 1 / time1
	
	
	key = cv2.waitKey(1)
	
	# Zoom in press '+'
	if key == 43:
		if ZOOM < 100:
			ZOOM += 10
			zoom(ZOOM)
			
	# Zoom out pres '-'
	elif key == 45:
		if ZOOM > -1:
			ZOOM -= 10
			zoom(ZOOM)
			
	# Angle up with up arrow
	if key == 82:
		vertical_angle += 5
		pwm.setRotationAngle(0, vertical_angle)
		
	# Angle down with down arrow
	elif key == 84:
		vertical_angle -= 5
		pwm.setRotationAngle(0, vertical_angle)
		
	# Turn right with right arrow
	if key == 83:
		horizontal_angle += 5
		pwm.setRotationAngle(1, horizontal_angle)
		
	# Turn left with left arrow
	elif key == 81:
		horizontal_angle -= 5
		pwm.setRotationAngle(1, horizontal_angle)
			
	# Press 'q' to quit
	elif key == ord('q'):
		break
	
	rawCapture.truncate(0)
	
camera.close()
pwm.exit_PCA9685()

cv2.destroyAllWindows()
