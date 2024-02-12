import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import pyrealsense2 as rs
import math

def determine_movement(center_x, frame_width):
    """
    Determine the direction in which the camera needs to move.
    """
    if center_x < frame_width // 3:
        return "left"
    elif center_x > 2 * frame_width // 3:
        return "right"
    else:
        return "center"

def calculate_angle(center_x1, center_x2, frame_width, fov):
    """
    Calculate the angle by which the camera needs to rotate.
    """
    # Calculate the distance from the center of the frame to the center of each ball
    distance1 = abs(center_x1 - frame_width / 2)
    distance2 = abs(center_x2 - frame_width / 2)

    # Calculate the angle based on the distances and field of view (fov)
    angle = math.atan(distance2 / (frame_width / 2) * math.tan(fov / 2)) - math.atan(distance1 / (frame_width / 2) * math.tan(fov / 2))
    return angle

def calculate_middle_angle(center_x, frame_width):
    """
    Calculate the angle to adjust the camera to the middle position.
    """
    distance = center_x - frame_width / 2
    angle = math.atan(distance / (frame_width / 2))
    return angle

def draw_grid(image, num_rows, num_cols, color, thickness):
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

def live_object_detection():
    '''main Function'''
    # Initialize the RealSense D455 camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)

    model = YOLO("model_- 31 january 2024 11_41.pt")

    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Define camera parameters
    fov = math.radians(69.4)  # Horizontal field of view in radians

    # Define grid parameters
    grid_color = (255, 255, 255)
    grid_thickness = 1
    num_rows = 3
    num_cols = 3

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        data = model(color_image)
        detections = sv.Detections.from_ultralytics(data[0])

        annotated_image = color_image.copy()

        # Initialize variables to store the center positions and depths of the detected balls
        center_x_red = None
        center_x_blue = None
        depth_red = None
        depth_blue = None

        for det in detections:
            label = f"{model.model.names[det[3]]} {det[2]:.2f}"  # Accessing class index and confidence from tuple
            box = det[0]
            # Get center pixel of the detection box
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            # Get depth value at the center pixel
            depth_value = depth_image[center_y, center_x]
            # Convert depth value to meters using depth scale
            depth = depth_value * depth_scale
            label += f" Depth: {depth:.2f} meters"

            # Draw bounding box on the image
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store center positions and depths of red and blue balls
            if model.model.names[det[3]] == "red ball":
                center_x_red = center_x
                depth_red = depth
            elif model.model.names[det[3]] == "blue ball":
                center_x_blue = center_x
                depth_blue = depth

        if center_x_red is not None and center_x_blue is not None:
            # Calculate the angle to rotate the camera based on the depths of the balls
            angle = calculate_angle(center_x_red, center_x_blue, color_image.shape[1], fov)
            print(f"Rotate camera by {math.degrees(angle):.2f} degrees")

        if center_x_red is not None:
            # Calculate the angle to adjust the camera to the middle position
            middle_angle = calculate_middle_angle(center_x_red, color_image.shape[1])
            print(f"Adjust camera to middle by {math.degrees(middle_angle):.2f} degrees")

        # Draw grid overlay
        draw_grid(annotated_image, num_rows, num_cols, grid_color, grid_thickness)

        # Display the live camera feed with object detection annotations
        cv2.imshow("Live Object Detection", annotated_image)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the OpenCV window
    pipeline.stop()
    cv2.destroyAllWindows()

# Run the live object detection function
live_object_detection()
