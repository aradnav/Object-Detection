import cv2
import numpy as np
import pyrealsense2 as rs
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def interpret_output(output):
    # Adjust these values to match your model's output format
    grid_size = 13
    num_anchors = 5
    num_classes = 20

    # Reshape the output tensor
    output = output.reshape((grid_size, grid_size, num_anchors, 5 + num_classes))

    # Extract the bounding box coordinates and dimensions
    box_xy = output[..., :2]
    box_wh = output[..., 2:4]

    # Compute the box corners (used to draw the bounding box)
    box_mins = box_xy - box_wh / 2
    box_maxes = box_xy + box_wh / 2

    # Extract the object confidence
    object_confidence = output[..., 4:5]

    # Extract the class probabilities
    class_probs = output[..., 5:]

    # Compute the class scores
    class_scores = object_confidence * class_probs

    # Find the class with the highest score
    classes = np.argmax(class_scores, axis=-1)
    scores = np.max(class_scores, axis=-1)

    # Create a list of detections
    detections = []
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(num_anchors):
                if scores[i, j, k] > 0.5:  # adjust this to your desired confidence threshold
                    x, y, w, h = box_mins[i, j, k, 0], box_mins[i, j, k, 1], box_maxes[i, j, k, 0], box_maxes[i, j, k, 1]
                    detections.append([classes[i, j, k], scores[i, j, k], x, y, w, h])

    return detections
def interpret_output(output):
    # Print the shape of the output
    print("Output shape:", output.shape)

    # Adjust these values to match your model's output format
    grid_size = 13
    num_anchors = 5
    num_classes = 20

    # Calculate the total size of the expected shape
    expected_size = grid_size * grid_size * num_anchors * (5 + num_classes)

    # Check if the output size matches the expected size
    if output.size != expected_size:
        print(f"Warning: output size ({output.size}) does not match expected size ({expected_size}).")
        return []

    # Reshape the output tensor
    output = output.reshape((grid_size, grid_size, num_anchors, 5 + num_classes))

# Initialize RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Load TensorRT model
TRT_MODEL_PATH = "model.trt"
runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))

with open(TRT_MODEL_PATH, "rb") as f:
    engine_data = f.read()
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# Allocate buffers
input_name = engine.get_binding_name(0)
output_name = engine.get_binding_name(1)
input_shape = engine.get_binding_shape(input_name)
output_shape = engine.get_binding_shape(output_name)
d_input = cuda.mem_alloc(trt.volume(input_shape) * np.dtype(np.float32).itemsize)
d_output = cuda.mem_alloc(trt.volume(output_shape) * np.dtype(np.float32).itemsize)
bindings = [int(d_input), int(d_output)] 
context.set_binding_shape(0, input_shape)
context.set_binding_shape(1, output_shape)
context.execute_v2(bindings=bindings)

try:
    while True:
        # Get a new frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # Create a separate variable for the image to display
        display_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Preprocess the frame
        frame = cv2.resize(frame, (224, 224))  # adjust this to your model's input size
        frame = frame.transpose((2, 0, 1))  # adjust this to your model's input format
        frame = frame.astype(np.float32) / 255.0  # adjust this to your model's input format

        # Ensure the frame data is contiguous in memory before copying it to the GPU
        frame = np.ascontiguousarray(frame)

        # Calculate the size of the input and output in bytes
        input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
        output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize

        # Allocate buffers
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)

        # Print debugging information
        print("Frame size:", frame.size)
        print("GPU input memory size:", input_size)
        print("GPU output memory size:", output_size)
        print("Frame is contiguous:", frame.flags['C_CONTIGUOUS'])
        print("Current CUDA context:", cuda.Context.get_current())

        # Run the frame through the model
        cuda.memcpy_htod(d_input, frame.ravel())
        context.execute(batch_size=1, bindings=bindings)
        output = np.empty(output_shape, dtype=np.float32)  # Replace output_size with output_shape
        cuda.memcpy_dtoh(output, d_output)
        # Interpret the output
        detections = interpret_output(output)

        # Display the results
        print("Detections:", detections)
        cv2.imshow('RealSense', display_image)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()

    # Free the GPU memory
    del d_input, d_output  # Replace cuda.mem_free with del