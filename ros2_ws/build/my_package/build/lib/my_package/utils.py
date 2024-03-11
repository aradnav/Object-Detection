# object_detection/object_detection/utils.py

import cv2
import numpy as np
import math
from rclpy.node import Node

class Utils(Node):

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
