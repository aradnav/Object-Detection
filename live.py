import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch

# Function to initialize RealSense camera
def initialize_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline

# Function to capture color frame from RealSense camera
def get_color_frame(pipeline):
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if color_frame:
        color_image = np.asanyarray(color_frame.get_data())
        return color_image
    else:
        return None

# Function to perform object detection with YOLO model
def detect_objects(image):
    # Load the YOLO model
    model = YOLO("/home/cadt-02/Object-Detection/ros2_ws/model_- 31 january 2024 11_41.pt")
    # Set the input size for the model
    input_size = (416, 416)
    # Preprocess the image
    blob = cv2.dnn.blobFromImage(image, 1/255, input_size, swapRB=True, crop=False)
    # Perform forward pass and get the output
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blob = torch.from_numpy(blob).to(device)
    outputs = model(blob)
    # Process the output to get the detected objects
    # Your code to process the output and draw bounding boxes goes here
    pass

# Main function to run object detection with RealSense camera and YOLO model
def run_object_detection():
    pipeline = initialize_camera()
    
    while True:
        color_image = get_color_frame(pipeline)
        if color_image is None:
            print("No valid color frame received")
            continue

        # Perform object detection on the color image
        detected_image = detect_objects(color_image)

        # Display the detected image
        cv2.imshow("Object Detection", detected_image)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()

# Run the object detection
if __name__ == "__main__":
    run_object_detection()
