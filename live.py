import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import pyrealsense2 as rs

def live_object_detection():
    '''main Function'''
    # Initialize the RealSense D455 camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    model = YOLO("model_- 31 january 2024 11_41.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())

        data = model(frame)
        detection = sv.Detections.from_ultralytics(data[0])

        label = [f"{model.model.names[ci]} {con:0.2f}"
                 for _, _, con, ci, _ in detection]

        if frame is None:
            print("Error: Failed to grab frame.")
            break

        frame = box_annotator.annotate(scene=frame, detections=detection, labels=label)

        # Display the live camera feed with object detection annotations
        cv2.imshow("Live Object Detection", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the OpenCV window
    pipeline.stop()
    cv2.destroyAllWindows()

# Run the live object detection function
live_object_detection()