import rclpy
from rclpy.node import Node
import cv2
import os 
import numpy as np
import time
import pyrealsense2 as rs
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from my_package.utils import Utils

class LiveObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('live_object_detection_node')
        self.create_timer(0.5, self.update)  # Adjust the update rate to 2 frames per second

        # Initialize camera and TensorRT model
        self.initialize_camera()
        self.initialize_tensorrt_model()
        self.allocate_buffers()

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

    def initialize_tensorrt_model(self):
        try:
            # Load TensorRT optimized model
            TRT_MODEL_PATH = "/home/cadt-02/Object-Detection/model.trt"
            # Check if the model file exists
            if not os.path.exists(TRT_MODEL_PATH):
                raise FileNotFoundError(f"TensorRT model file not found: {TRT_MODEL_PATH}")          
            self.runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))      
            with open(TRT_MODEL_PATH, "rb") as f:
                engine_data = f.read()
            # Check if the engine data is not empty
            if not engine_data:
                raise ValueError("The TensorRT model file is empty.")          
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)     
            if self.engine is None:
                raise ValueError("Failed to deserialize the TensorRT engine.")        
            self.context = self.engine.create_execution_context()
            # Set optimization profile dimensions
            if self.engine.has_implicit_batch_dimension:
                raise ValueError("The TensorRT engine has an implicit batch dimension. This code is designed for explicit batch dimensions.")
            if self.engine.num_optimization_profiles == 0:
                raise ValueError("The TensorRT engine does not have any optimization profiles defined.")
            self.context.active_optimization_profile = 0
            for binding in range(self.engine.num_bindings):
                if self.engine.binding_is_input(binding):
                    input_name = self.engine.get_binding_name(binding)
                    min_shape, opt_shape, max_shape = self.engine.get_profile_shape(0, input_name)
                    print(f"Binding: {binding}, Optimal Shape: {opt_shape}")
                    if not self.context.set_binding_shape(binding, opt_shape):
                        raise ValueError(f"Failed to set the binding shape for input {input_name}.")
        except Exception as e:
            self.get_logger().error(f"Failed to load TensorRT model: {e}")
            self.destroy_node()
            return

    def allocate_buffers(self):
        # Allocate device memory for inputs and outputs
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        for binding in range(self.engine.num_bindings):
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append((host_mem, device_mem))
            else:
                self.outputs.append((host_mem, device_mem))

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

            # Perform object detection using TensorRT
            preprocessed_image = preprocess_image(color_image)
            np.copyto(self.inputs[0][0], preprocessed_image.ravel())

            # Perform inference
            self.context.execute_async(bindings=self.bindings, stream_handle=self.stream.handle)

            # Transfer predictions back from device
            cuda.memcpy_dtoh_async(self.outputs[0][0], self.outputs[0][1], self.stream)

            # Synchronize the stream
            self.stream.synchronize()

            # Post-process output
            detections = postprocess_output(self.outputs[0][0])

            # Draw bounding boxes
            for detection in detections:
                label, confidence, box = detection
                draw_bounding_box(annotated_image, box, label, confidence)
            
            # Draw lines
            self.draw_lines(annotated_image)

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

    def draw_lines(self, image):
        line_length = 100
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        line1_start = (center_x - line_length, center_y)
        line1_end = (center_x + line_length, center_y)
        line2_start = (center_x, center_y - line_length)
        line2_end = (center_x, center_y + line_length)

        # Draw lines
        cv2.line(image, line1_start, line1_end, (0, 255, 0), 2)
        cv2.line(image, line2_start, line2_end, (0, 255, 0), 2)

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
