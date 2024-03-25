import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import supervision as sv
from ultralytics import YOLO
import pyrealsense2 as rs
import math

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()
        self.model = YOLO("/home/cadt-02/Object-Detection/ros2_ws/model_- 31 january 2024 11_41.pt")
        self.depth_scale = None
        self.fov = math.radians(69.4)
        self.grid_color = (255, 255, 255)
        self.grid_thickness = 1
        self.num_rows = 3
        self.num_cols = 3
        self.publisher = self.create_publisher(Image, 'object_detection_output', 10)
        self.subscription = self.create_subscription(Image, 'camera/color/image_raw', self.image_callback, 10)
        self.subscription  # prevent unused variable warning

    def image_callback(self, msg):
        color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        annotated_image = color_image.copy()

        data = self.model(color_image)
        detections = sv.Detections.from_ultralytics(data[0])

        for det in detections:
            label = f"{self.model.model.names[det[3]]} {det[2]:.2f}"
            box = det[0]
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if self.model.model.names[det[3]] == "red ball":
                center_x_red = center_x
            elif self.model.model.names[det[3]] == "blue ball":
                center_x_blue = center_x

        if center_x_red is not None and center_x_blue is not None:
            angle = self.calculate_angle(center_x_red, center_x_blue, color_image.shape[1], self.fov)
            self.get_logger().info(f"Rotate camera by {math.degrees(angle):.2f} degrees")

        if center_x_red is not None:
            middle_angle = self.calculate_middle_angle(center_x_red, color_image.shape[1])
            self.get_logger().info(f"Adjust camera to middle by {math.degrees(middle_angle):.2f} degrees")

        self.draw_grid(annotated_image, self.num_rows, self.num_cols, self.grid_color, self.grid_thickness)

        output_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        self.publisher.publish(output_msg)

    def calculate_angle(self, center_x1, center_x2, frame_width, fov):
        distance1 = abs(center_x1 - frame_width / 2)
        distance2 = abs(center_x2 - frame_width / 2)
        angle = math.atan(distance2 / (frame_width / 2) * math.tan(fov / 2)) - math.atan(distance1 / (frame_width / 2) * math.tan(fov / 2))
        return angle

    def calculate_middle_angle(self, center_x, frame_width):
        distance = center_x - frame_width / 2
        angle = math.atan(distance / (frame_width / 2))
        return angle

    def draw_grid(self, image, num_rows, num_cols, color, thickness):
        height, width = image.shape[:2]
        cell_width = width // num_cols
        cell_height = height // num_rows

        for i in range(1, num_rows):
            cv2.line(image, (0, i * cell_height), (width, i * cell_height), color, thickness)

        for i in range(1, num_cols):
            cv2.line(image, (i * cell_width, 0), (i * cell_width, height), color, thickness)

def main(args=None):
    rclpy.init(args=args)
    object_detection_node = ObjectDetectionNode()
    rclpy.spin(object_detection_node)
    object_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
