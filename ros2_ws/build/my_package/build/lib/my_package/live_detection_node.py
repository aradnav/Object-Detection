import rclpy
from rclpy.node import Node
import cv2
import os 
import time
import pyrealsense2 as rs
import torch
import numpy as np
from ultralytics import YOLO
import math
import supervision as sv

class LiveObjectDetectionNode(Node):

    def __init__(self):
        super().__init__('live_object_detection_node')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO("/home/cadt-02/Object-Detection/ros2_ws/model_- 18 march 2024 15_19.pt")
        self.initialize_camera()

    def initialize_camera(self):
        # Initialize the RealSense D455 camera
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)

    def determine_movement(self, center_x, frame_width):
        """
        Determine the direction in which the camera needs to move.
        """
        if center_x < frame_width // 3:
            return "left"
        elif center_x > 2 * frame_width // 3:
            return "right"
        else:
            return "center"

    def calculate_angle(self, center_x1, center_x2, frame_width, fov):
        """
        Calculate the angle by which the camera needs to rotate.
        """
        # Calculate the distance from the center of the frame to the center of each ball
        distance1 = abs(center_x1 - frame_width / 2)
        distance2 = abs(center_x2 - frame_width / 2)

        # Calculate the angle based on the distances and field of view (fov)
        angle = math.atan(distance2 / (frame_width / 2) * math.tan(fov / 2)) - math.atan(distance1 / (frame_width / 2) * math.tan(fov / 2))
        return angle

    def calculate_middle_angle(self, center_x, frame_width):
        """
        Calculate the angle to adjust the camera to the middle position.
        """
        distance = center_x - frame_width / 2
        angle = math.atan(distance / (frame_width / 2))
        return angle

    def draw_red_dot(self, image):
        """
        Draw a red dot in the middle of the camera's field of view.
        """
        height, width, _ = image.shape
        # Calculate the coordinates of the middle point
        center_x = width // 2
        center_y = height // 2
        # Draw a red dot at the middle point
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

    def draw_line_to_middle(self, image, center_x_obj, center_y_obj):
        """
        Draw a line from the detected object's center to the red dot in the middle of the camera's field of view.
        """
        height, width, _ = image.shape
        # Calculate the coordinates of the middle point
        center_x_middle = width // 2
        center_y_middle = height // 2
        # Draw a line from the object's center to the middle point
        cv2.line(image, (int(center_x_obj), int(center_y_obj)), (center_x_middle, center_y_middle), (255, 0, 0), 2)

    def get_detection_position(self, box, frame_width, frame_height, distance, fov):
        """
        Get the position of the detection relative to the center of the frame.
        Returns the (x, y) offset from the center of the frame in cm.
        """
        x1, y1, x2, y2 = box.astype(int)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        offset_x = center_x - frame_width / 2
        offset_y = center_y - frame_height / 2

        # Convert offset from pixels to cm
        # This assumes the camera's field of view is the same in the x and y directions
        sensor_width = 2 * distance * math.tan(math.radians(fov / 2))
        pixel_size = sensor_width / frame_width
        offset_x_cm = offset_x * pixel_size
        offset_y_cm = offset_y * pixel_size

        return offset_x_cm, offset_y_cm


    def draw_grid(self, image, num_rows, num_cols, color, thickness):
        """
        Draw a grid overlay on the image.
        """
        height, width = image.shape[:2]
        cell_width = width // num_cols
        cell_height = height // num_rows

        # Draw horizontal lines
        for i in range(1, num_rows):
            cv2.line(image, (0, i * cell_height), (width, i * cell_height), color, thickness)

        # Draw vertical lines
        for i in range(1, num_cols):
            cv2.line(image, (i * cell_width, 0), (i * cell_width, height), color, thickness)

    def update(self):
        '''main Function'''
        while rclpy.ok():
            # Get depth scale
            depth_sensor = self.profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # Define camera parameters
            fov = math.radians(69.4)  # Horizontal field of view in radians

            # Define grid parameters
            grid_color = (255, 255, 255)
            grid_thickness = 1
            num_rows = 3
            num_cols = 3

            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            data = self.model(color_image)
            detections = sv.Detections.from_ultralytics(data[0])

            annotated_image = color_image.copy()

            # Initialize variables to store the center positions and depths of the detected balls
            center_x_red = None
            center_x_blue = None
            depth_red = None
            depth_blue = None

            for det in detections:
                label = f"{self.model.model.names[det[3]]} {det[2]:.2f}"  # Accessing class index and confidence from tuple
                print("Detected:", label)  # Print detected object names and confidence scores for debugging
                box = det[0]
                # Get center pixel of the detection box
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                # Get depth value at the center pixel
                depth_value = depth_image[center_y, center_x]
                distance = depth_value
                # Convert depth value to meters using depth scale
                depth = depth_value * depth_scale
                label += f" Depth: {depth:.2f} meters"

                print(f"Distance to {self.model.model.names[det[3]]}: {depth:.2f} meters")
                cv2.circle(annotated_image, (center_x, center_y), 5, (0, 255, 255), -1)
                
                # Get position of the detection
                offset_x, offset_y = self.get_detection_position(box, color_image.shape[1], color_image.shape[0], distance, fov)
                print(f"{self.model.model.names[det[3]]} Detection Position (Offset):", offset_x, offset_y)

                # Draw bounding box on the image
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                self.draw_red_dot(annotated_image)
                # Store center positions and depths of red and blue balls
                if self.model.model.names[det[3]] == "red":
                    center_x_red = center_x
                    depth_red = depth
                elif self.model.model.names[det[3]] == "blue":
                    center_x_blue = center_x
                    depth_blue = depth

            if center_x_red is not None and center_x_blue is not None:
                print("Inside angle calculation block")
                angle = self.calculate_angle(center_x_red, center_x_blue, color_image.shape[1], fov)
                print(f"Rotate camera by {math.degrees(angle):.2f} degrees")
                movement = self.determine_movement((center_x_red + center_x_blue) / 2, color_image.shape[1])
                print(f"Move camera {movement}")

            if center_x_red is not None:
                print("Inside middle angle calculation block")
                middle_angle = self.calculate_middle_angle(center_x_red, color_image.shape[1])
                print(f"Adjust camera to middle by {math.degrees(middle_angle):.2f} degrees")

            if center_x_red is not None:
                self.draw_line_to_middle(annotated_image, center_x_red, center_y)
            if center_x_blue is not None:
                self.draw_line_to_middle(annotated_image, center_x_blue, center_y)

            # Draw grid overlay
            self.draw_grid(annotated_image, num_rows, num_cols, grid_color, grid_thickness)

            # Display the live camera feed with object detection annotations
            cv2.imshow("Live Object Detection", annotated_image)

        # Release the camera and close the OpenCV window
        self.pipeline.stop()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    live_object_detection_node = LiveObjectDetectionNode()
    try:
        live_object_detection_node.update()
    except KeyboardInterrupt:
        pass
    finally:
        live_object_detection_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
