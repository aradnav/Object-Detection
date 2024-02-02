# object_detection.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='object_detection_pkg',
            executable='realsense_camera_node.py',
            name='realsense_camera_node'
        ),
        Node(
            package='object_detection_pkg',
            executable='live_detection_node.py',
            name='live_detection_node'
        ),
        Node(
            package='object_detection_pkg',
            executable='yolo_detector_node.py',
            name='yolo_detector_node'
        ),
    ])
if __name__ == '__main__':
    generate_launch_description()
