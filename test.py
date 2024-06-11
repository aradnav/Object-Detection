import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Step 1: Record Video in MP4 Format
def record_video(video_filename='output.mp4'):
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires a camera with a Color sensor")
        return

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    # Create an OpenCV video writer
    out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Show images
            cv2.imshow('RealSense', color_image)

            # Write the color frame to the video file
            out.write(color_image)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()

# Step 2: Extract Frames from Video
def extract_frames(video_filename='output.mp4', frames_dir='frames'):
    # Open the video file
    video_capture = cv2.VideoCapture(video_filename)

    # Check if the video file opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        return

    # Create a directory to save frames
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Save the frame as an image file
        frame_filename = os.path.join(frames_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    # Release the video capture object
    video_capture.release()

    print(f"Extracted {frame_count} frames to '{frames_dir}' directory.")

# Main function to record video and then extract frames
def main():
    video_filename = 'output.mp4'
    frames_dir = 'frames'

    record_video(video_filename)
    extract_frames(video_filename, frames_dir)

if __name__ == "__main__":
    main()
