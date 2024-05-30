# object_detection/object_detection/main.py

import rclpy
from my_package.live_detection_node import LiveObjectDetectionNode
from my_package.reinforcement_r2 import ObjectDetectionNode

def main(args=None):
    rclpy.init(args=args)
    live_object_detection_node = LiveObjectDetectionNode()
    reinforcement_r2 = ObjectDetectionNode()
    try:
        live_object_detection_node.update()
        reinforcement_r2.update()
    except KeyboardInterrupt:
        pass
    finally:
        live_object_detection_node.destroy_node()
        reinforcement_r2.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
