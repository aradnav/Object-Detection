# object_detection/object_detection/main.py

import rclpy
from my_package.live_detection_node import LiveObjectDetectionNode

def main(args=None):
    rclpy.init(args=args)
    live_object_detection_node = LiveObjectDetectionNode()
    rclpy.spin(live_object_detection_node)
    live_object_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
