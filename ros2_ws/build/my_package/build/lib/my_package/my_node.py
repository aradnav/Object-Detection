import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

class YOLORealSenseNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.bridge = CvBridge()
        self.detection = YOLORealSenseDetection()
        self.image_publisher = self.create_publisher(
            Image,
            'yolo_realSense/output',
            10
        )

    def color_callback(self, data):
        try:
            color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        annotated_image, center_x_red, center_x_blue, depth_red, depth_blue = self.detection.detect(color_image)

        if center_x_red is not None and center_x_blue is not None:
            angle = self.detection.calculate_angle(center_x_red, center_x_blue, color_image.shape[1], self.detection.fov)
            self.get_logger().info(f"Rotate camera by {math.degrees(angle):.2f} degrees")

        if center_x_red is not None:
            middle_angle = self.detection.calculate_middle_angle(center_x_red, color_image.shape[1])
            self.get_logger().info(f"Adjust camera to middle by {math.degrees(middle_angle):.2f} degrees")

        try:
            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(annotated_image, "bgr8"))
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")

        # Display the annotated image
        cv2.imshow("YOLO Annotated Image", annotated_image)
        cv2.waitKey(1)  # Needed for OpenCV to refresh the window

    def depth_callback(self, data):
        pass  # No need to process depth data here


def main(args=None):
    rclpy.init(args=args)

    node = YOLORealSenseNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()