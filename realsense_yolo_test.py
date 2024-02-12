import cv2
import torch
import threading
import supervision as sv
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
from queue import Queue
class RealSenseCapture:

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config=self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.queue = Queue()
        self.running = True

        def capture_loop(self):
            while self.running:
                frames = self.pipeline.wait_for_frames(timeout_ms=10000)
                if not frames:
                    continue

                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                self.queue.put((color_image, depth_image))

        self.thread = threading.Thread(target=capture_loop, args=(self,))
        self.thread.start()

    def read(self):
        return self.queue.get()

    def release(self):
        self.running = False
        self.thread.join()
        self.pipeline.stop()

def live_object_detection():
    cap = RealSenseCapture()
    model = YOLO("model_- 23 january 2024 15_03.pt")  # Corrected the model file path

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    frame_count = 0
    object_detection_counter = 0
    frame = None
    while True:
        color_image, depth_image = cap.read()


        if frame_count % 8 == 0:
                data = model(color_image)
                print(f"Raw model output: {data}")
                detections = sv.Detections.from_ultralytics(data[0])

                label = [f"{model.model.names[ci]} {con:0.2f}"
                         for _, _, con, ci, _ in detections]
                print(f"Data from model: {data}")  # Debug line
                print(f"Detections: {detections}")  # Debug line
                print(f"Labels: {label}")  # Debug line

                frame = box_annotator.annotate(scene=color_image, detections=detections, labels=label)
                print(f"Number of detections: {len(detections)}")  # Debug line
                print(f"Number of labels: {len(label)}")  # Debug line
        if frame_count % 20 == 0:
                data = model(color_image)
                detection = sv.Detections.from_ultralytics(data[0])

                label = [f"{model.model.names[ci]} {con:0.2f}"
                         for _, _, con, ci, _ in detection]

                frame = box_annotator.annotate(scene=color_image, detections=detection, labels=label)

                object_detection_counter += 1

        cv2.imshow("Live Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Run the live object detection function
live_object_detection()
