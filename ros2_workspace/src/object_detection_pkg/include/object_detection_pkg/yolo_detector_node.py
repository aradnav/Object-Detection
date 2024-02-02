# yolo_detector_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import cv2

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.bridge = CvBridge()
        self.image_subscription = self.create_subscription(
            Image, 'image', self.image_callback, 10)
        self.yolo_model = YOLO('')  # Update with correct path
        self.detection_publisher = self.create_publisher(
            Image, 'yolo_detection', 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(e)
            return

        # Perform object detection using YOLO
        results = self.yolo_model.predict(cv_image)

        # Draw bounding boxes and labels on the image
        for box in results.xyxy:
            x1, y1, x2, y2, conf, cls = map(int, box)
            label = results.names[cls]
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the image back to a ROS message and publish it
        try:
            detection_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.detection_publisher.publish(detection_msg)
        except CvBridgeError as e:
            self.get_logger().error(e)

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    rclpy.spin(node)

if __name__ == '__main__':
    main()