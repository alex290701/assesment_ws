#!/usr/bin/env python3
import cv2
import numpy as np
import onnxruntime as ort
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge,CvBridgeError


def callback(data):
    cv_image = np.frombuffer(data.data, dtype=np.uint8).reshape(
        data.height, data.width, -1)
    # cv_image = bridge.imgmsg_to_cv2(data)
    # input_shape = model.get_inputs()[0].shape
    # cam = cv2.VideoCapture(0)
    while True:
        # cv_image = cv2.imread("test.jpg")
        # ret, cv_image = cam.read()
        # if not ret:
        #     continue
        input_blob = cv2.dnn.blobFromImage(
            cv_image, 1.0, (640, 640), (127.5, 127.5, 127.5), swapRB=True, crop=False
        )
        net.setInput(input_blob)
        outputs = net.forward()
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
            if confidence >= 1:
                classes_scores = detection[5:]
                class_id = np.argmax(classes_scores)
                if classes_scores[class_id] > 0.9:
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = detection[0], detection[1], detection[2], detection[3]
                    left = int((x - w / 2) * x_factor)
                    top = int((y - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        i = 0
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.9, 0.9)
        # print(len(indices))
        for i in indices:
            x, y, w, h = boxes[i]
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.putText(
                img=cv_image,
                text="{class_ids[i]}:{confidences[i]}",
                org=(x, y),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=1.0,
                color=(125, 246, 55),
                thickness=1,
            )
            i = i + 1
        cv2.imshow("Object Detection", cv_image)
        key = cv2.waitKey(1)
        if key % 256 == 27:
            print("Closing the window (Esc key pressed)")
            break
    # cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    script_directory = os.path.dirname(os.path.abspath(__file__))
    model_filename = "best.onnx"
    model_path = os.path.join(script_directory, model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found")
    model = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    net = cv2.dnn.readNetFromONNX(model_path)

    rospy.init_node('object_detection', anonymous=False)
    # bridge = CvBridge()
    image_sub = rospy.Subscriber("/image_raw", Image, callback)

    rospy.spin()
