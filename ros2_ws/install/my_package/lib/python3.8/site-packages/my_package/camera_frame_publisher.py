# object_detection_ros2/object_detection_ros2/camera_frame_publisher.py

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraFramePublisher(Node):
    def __init__(self):
        super().__init__('camera_frame_publisher')
        self.publisher_ = self.create_publisher(Image, 'camera_frame', 10)
        self.timer = self.create_timer(1.0 / 30, self.publish_frame)
        self.bridge = CvBridge()

        # Initialize RealSense camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraFramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
