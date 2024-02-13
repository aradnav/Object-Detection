import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
import math
import pyrealsense2 as rs
import supervision as sv
from ultralytics import YOLO
from my_package.utils import draw_grid

class LiveObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('live_object_detection_node')
        self.create_timer(0.5, self.update)  # Adjust the update rate to 2 frames per second

        # Initialize camera and YOLO model
        self.initialize_camera()
        self.model = None
        try:
            self.model = YOLO("model_- 31 january 2024 11_41.pt")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")

        # Define grid parameters
        self.num_rows = 3
        self.num_cols = 3
        self.grid_color = (255, 255, 255)
        self.grid_thickness = 1

        # Initialize time for frame rate control
        self.last_update_time = time.time()

    def initialize_camera(self):
        try:
            # Initialize the RealSense D455 camera
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.profile = self.pipeline.start(self.config)
        except Exception as e:
            self.get_logger().error(f"Failed to initialize camera: {e}")
            self.destroy_node()
            return

    def update(self):
        current_time = time.time()
        if current_time - self.last_update_time < 0.5:  # Throttle update rate to 2 frames per second
            return

        self.last_update_time = current_time

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=10000)  # Adjust the timeout value as needed
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                self.get_logger().warn("No valid frames received")
                return

            color_image = np.asanyarray(color_frame.get_data())

            annotated_image = color_image.copy()

            # Perform object detection if YOLO model is loaded
            if self.model:
                # Resize image to reduce computation
                resized_image = cv2.resize(color_image, (416, 416))
                data = self.model(resized_image)
                detections = sv.Detections.from_ultralytics(data[0])

                for det in detections:
                    label = f"{self.model.model.names[det[3]]} {det[2]:.2f}"  # Accessing class index and confidence from tuple
                    box = det[0]
                    # Scale box back to original size
                    x1, y1, x2, y2 = (box * np.array([color_image.shape[1], color_image.shape[0], color_image.shape[1], color_image.shape[0]])).astype(int)
                    # Draw bounding box on the image
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw grid overlay
            self.draw_grid(annotated_image)

            # Display the live camera feed with object detection annotations
            cv2.imshow("Live Object Detection", annotated_image)

            # Exit the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.pipeline.stop()
                cv2.destroyAllWindows()
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error in update loop: {e}")

    def draw_grid(self, image):
        """
        Draw a grid overlay on the image.
        """
        height, width = image.shape[:2]
        cell_width = width // self.num_cols
        cell_height = height // self.num_rows

        # Draw horizontal lines
        for i in range(1, self.num_rows):
            cv2.line(image, (0, i * cell_height), (width, i * cell_height), self.grid_color, self.grid_thickness)

        # Draw vertical lines
        for i in range(1, self.num_cols):
            cv2.line(image, (i * cell_width, 0), (i * cell_width, height), self.grid_color, self.grid_thickness)


def main(args=None):
    rclpy.init(args=args)
    live_object_detection_node = LiveObjectDetectionNode()
    try:
        rclpy.spin(live_object_detection_node)
    except KeyboardInterrupt:
        pass
    finally:
        live_object_detection_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
