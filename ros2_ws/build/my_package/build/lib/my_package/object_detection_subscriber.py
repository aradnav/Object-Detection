# object_detectionros2/objectdetection_ros2/object_detection_subscriber.py

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv

class ObjectDetectionSubscriber(Node):
    def __init__(self):
        super().__init__('object_detection_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'camera_frame',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.model = YOLO("yolov5s.pt")
        self.model.model.to('cuda')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detections = self.detect_objects(frame)
        annotated_frame = self.annotate_objects(frame, detections)
        cv2.imshow("Object Detection", annotated_frame)
        cv2.waitKey(1)

    def detect_objects(self, frame):
        data = self.model(frame)
        detections = sv.Detections.from_ultralytics(data[0])
        return detections

    def annotate_objects(self, frame, detections):
        for det in detections:
            label = f"{self.model.model.names[det[3]]} {det[2]:.2f}"
            box = det[0]
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()