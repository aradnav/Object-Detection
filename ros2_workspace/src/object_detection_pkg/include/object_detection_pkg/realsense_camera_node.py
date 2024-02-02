# realsense_camera_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs
import numpy as np

class RealsenseCamera(Node):
    def __init__(self):
        super().__init__('realsense_camera')
        self.bridge = CvBridge()
        self.image_publisher = self.create_publisher(Image, 'image', 10)
        self.depth_image_publisher = self.create_publisher(Image, 'depth_image', 10)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)

    def publish_image(self, frame):
        try:
            cv_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_publisher.publish(cv_image)
        except CvBridgeError as e:
            self.get_logger().error(e)

    def publish_depth_image(self, frame):
        try:
            cv_image = self.bridge.cv2_to_imgmsg(frame, "mono16")
            self.depth_image_publisher.publish(cv_image)
        except CvBridgeError as e:
            self.get_logger().error(e)

    def capture_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        self.publish_image(color_image)
        self.publish_depth_image(depth_image)

def main(args=None):
    rclpy.init(args=args)
    node = RealsenseCamera()
    rclpy.spin(node)

if __name__ == '__main__':
    main()