# object_detection/object_detection/main.py

import rclpy
from my_package.live_detection_node import LiveObjectDetectionNode

def main(args=None):
    rclpy.init(args=args)
    live_object_detection_node = LiveObjectDetectionNode()
    try:
        live_object_detection_node.update()
    except KeyboardInterrupt:
        pass
    finally:
        live_object_detection_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
