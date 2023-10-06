#!/usr/bin/env python3

import rospy
import cv2
import sys
import numpy as np
from sensor_msgs.msg import Image
import os

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45
 
# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
 
# Colors.
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
      # Create a 4D blob from a frame.
      blob = cv2.dnn.blobFromImage(input_image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)
 
      # Sets the input to the network.
      net.setInput(blob)
 
      # Run the forward pass to get output of the output layers.
      outputs = net.forward(net.getUnconnectedOutLayersNames())
      return outputs

def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    # Rows.
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT
    # Iterate through detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
                classes_scores = row[5:]
                # Get the index of max class score.
                class_id = np.argmax(classes_scores)
                #  Continue if the class score is above threshold.
                if (classes_scores[class_id] > SCORE_THRESHOLD):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    print(len(boxes))
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]             
        # Draw bounding box.             
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        # Class label.                      
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])             
        # Draw label.             
        draw_label(input_image, label, left, top)
    return input_image

# def imgmsg_to_cv2(img_msg):
#     if img_msg.encoding != "bgr8":
#         rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
#     dtype = np.dtype("uint8") # Hardcode to 8 bits...
#     dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
#     image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
#                     dtype=dtype, buffer=img_msg.data)
#     # If the byt order is different between the message and the system.
#     if img_msg.is_bigendian == (sys.byteorder == 'little'):
#         image_opencv = image_opencv.byteswap().newbyteorder()
#     return image_opencv

def callback( data):
    # print(data.encoding)
    try:
        # cv_image = self.imgmsg_to_cv2(data)
        # cv_image = cv2.cvtColor(self.imgmsg_to_cv2(data), cv2.COLOR_RGB2BGR)
        frame = np.frombuffer(data.data, dtype = np.uint8).reshape(data.height,data.width,-1)
    except:
        rospy.logerr("conversion error!!")
        return
    
    # Give the weight files to the model and load the network using       them.
    # modelWeights = "YOLOv5s.onnx"
    # Process image.
    detections = pre_process(frame, net)
    img = post_process(frame.copy(), detections)
    # """
    # Put efficiency information. The function getPerfProfile returns       the overall time for inference(t) 
    # and the timings for each of the layers(in layersTimes).
    # """
    # t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
    # print(label)
    # cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE,  (0, 0, 255), THICKNESS, cv2.LINE_AA)
    cv2.imshow('Output', img)
    cv2.waitKey()

if __name__ == '__main__':
    rospy.init_node('object_detection', anonymous=False)

    classes=['person','apple','lemon']
    script_directory = os.path.dirname(os.path.abspath(__file__))
    model_filename = "best.onnx"
    model_path = os.path.join(script_directory, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")
    net = cv2.dnn.readNet(model_path)
    
    image_sub = rospy.Subscriber("/image_raw", Image, callback)
    

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
      
