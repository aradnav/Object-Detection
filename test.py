import cv2
import torch
import threading
import supervision as sv
from ultralytics import YOLO
from queue import Queue

class VideoCapture:
    def __init__(self, index=0, queue_size=1):
        self.cap = cv2.VideoCapture(index)
        self.queue = Queue(queue_size)
        self.running = True

        def capture_loop(self):
            while self.running:
                ret, frame = self.cap.read()
                if ret:
                    self.queue.put((ret, frame))

        self.thread = threading.Thread(target=capture_loop, args=(self,))
        self.thread.start()

    def read(self):
        return self.queue.get()

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def live_object_detection():
    '''main Function'''
    # Initialize the camera
    cap = VideoCapture(0, queue_size=2)
    model = YOLO("model_- 22 january 2024 10_45.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    frame_count = 0
    object_detection_counter = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 240))  # Reduce frame size

        if frame_count % 20 == 0:  # Perform object detection every 20 frames
            if object_detection_counter % 2 == 0:
                data = model(frame)
                detection = sv.Detections.from_ultralytics(data[0])

                label = [f"{model.model.names[ci]} {con:0.2f}"
                         for _, _, con, ci, _ in detection]

                if not ret:
                    print("Error: Failed to grab frame.")
                    break

                frame = box_annotator.annotate(scene=frame, detections=detection, labels=label)

                object_detection_counter += 1

        # Display the live camera feed with object detection annotations
        cv2.imshow("Live Object Detection", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

# Run the live object detection function
live_object_detection()