#!/usr/bin/env python3

import rospy
import cv2
import onnxruntime as ort
import os
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class camera_1:

  def __init__(self,model_path):
    self.net = cv2.dnn.readNetFromONNX(model_path)
    self.image_sub = rospy.Subscriber("/image_raw", Image, self.callback)

  def callback(self,data):
    bridge = CvBridge()

    try:
      cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
    except CvBridgeError as e:
      rospy.logerr(e)
    
    image = cv_image

    input_blob = cv2.dnn.blobFromImage(
            cv_image, 1.0, (640, 640), (127.5, 127.5, 127.5), swapRB=True, crop=False
        )
    self.net.setInput(input_blob)
    outputs = self.net.forward()
    class_ids = []
    confidences = []
    boxes = []
    no_of_detections = outputs[0].shape[0]
    image_height, image_width = cv_image.shape[:2]
    x_factor = image_width / 640
    y_factor = image_height / 640
    for idx in range(no_of_detections):
        detection = outputs[0][idx]
        confidence = detection[4]
        if confidence >= 0.9:
            classes_scores = detection[5:]
            class_id = np.argmax(classes_scores)
            if classes_scores[class_id] > 0.8:
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = detection[0], detection[1], detection[2], detection[3]
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
    # i = 0
    # print(len(boxes))
    # print(len(confidences))
    confidences = np.array(confidences)
    # print(confidences)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.9, 0.1)
    # print(len(indices))
    for i in indices:
        # print(i[0])
        x, y, w, h = boxes[i[0]]
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(
            img=cv_image,
            text=f"{class_ids[i[0]]}:{confidences[i[0]]}",
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_DUPLEX,
            fontScale=1.0,
            color=(125, 246, 55),
            thickness=1,
        )
    #     i = i + 1

    image = cv_image
    # resized_image = cv2.resize(image, (360, 640)) 

    cv2.imshow("Camera output normal", image)
    # cv2.imshow("Camera output resized", resized_image)

    cv2.waitKey(1)

def main():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    model_filename = "yolov5n-face.onnx"
    model_path = os.path.join(script_directory, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")
    # model = ort.InferenceSession(
    #     model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    # )
    camera_1(model_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('camera_read', anonymous=False)
    main()