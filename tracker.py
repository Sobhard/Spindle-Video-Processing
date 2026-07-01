"""
This class process individual frames in an image
"""

import cv2
import numpy as np
from enum import Enum


class Color(Enum):
    RED = 0
    GREEN = 1
    YELLOW = 2
    # WHITE = 3


class DotTracker:
    """This class contains methods to process individual frames from the spindle video"""

    # Brush Size for Cleaning Up Artifacts (larger brush = larger artifacts are cleaned up)
    BRUSH_SIZE = 15

    # HSV Values for the masks

    # Red needs 2 sets of masks due to the way HSV works
    RED_MASK_LOWER_1 = np.array([0, 124, 104])
    RED_MASK_UPPER_1 = np.array([10, 255, 255])
    RED_MASK_LOWER_2 = np.array([167, 114, 154])
    RED_MASK_UPPER_2 = np.array([179, 255, 255])

    GREEN_MASK_LOWER = np.array([60, 50, 20])
    GREEN_MASK_UPPER = np.array([95, 255, 182])

    YELLOW_MASK_LOWER = np.array([20, 77, 140])
    YELLOW_MASK_UPPER = np.array([30, 255, 255])

    # LARGE artifacts when using the white mask
    WHITE_MASK_LOWER = np.array([81, 0, 208])
    WHITE_MASK_UPPER = np.array([179, 60, 255])

    @staticmethod
    def apply_mask(frame: np.ndarray, color: Color) -> np.ndarray:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masked_frame = None
        match color:

            case Color.RED:
                masked_frame_1 = cv2.inRange(
                    hsv_frame, DotTracker.RED_MASK_LOWER_1, DotTracker.RED_MASK_UPPER_1
                )
                masked_frame_2 = cv2.inRange(
                    hsv_frame, DotTracker.RED_MASK_LOWER_2, DotTracker.RED_MASK_UPPER_2
                )

                masked_frame = cv2.bitwise_or(masked_frame_1, masked_frame_2)

            case Color.GREEN:
                masked_frame = cv2.inRange(
                    hsv_frame, DotTracker.GREEN_MASK_LOWER, DotTracker.GREEN_MASK_UPPER
                )

            case Color.YELLOW:
                masked_frame = cv2.inRange(
                    hsv_frame,
                    DotTracker.YELLOW_MASK_LOWER,
                    DotTracker.YELLOW_MASK_UPPER,
                )

            # Need to implement motion tracking for white
            # case Color.WHITE:
            # masked_frame = cv2.inRange(
            # hsv_frame, DotTracker.WHITE_MASK_LOWER, DotTracker.WHITE_MASK_UPPER
            # )

        kernal = np.ones(
            (DotTracker.BRUSH_SIZE, DotTracker.BRUSH_SIZE), dtype=np.uint8
        )  # Brush size for cleanup (artifacts smaller than the brush are deleted)

        masked_frame = cv2.morphologyEx(masked_frame, cv2.MORPH_CLOSE, kernal)
        masked_frame = cv2.morphologyEx(masked_frame, cv2.MORPH_OPEN, kernal)

        return masked_frame

    @staticmethod
    def _get_dot_position(
        masked_frame: np.ndarray, previous_positons: np.ndarray | None = None
    ) -> list:
        """
        Runs an algorithm on a clean masked frame to find the center of the dot
        1. Finds contours in the masked image
        2. Fits a circle to the contours
        3. Finds the center of these circles

        Args:
            clean masked image
        Returns:
            two dimensional array of x, y positions of the center of each dot and the radius
            empty array if no contours found
        """

        contours, _ = cv2.findContours(
            masked_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )  # TODO: See if changing to chain_approx_none improves data

        coordinates = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 50:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                coordinates.append([int(x), int(y), int(radius)])

        return coordinates

    @staticmethod
    def process_frame(frame: np.ndarray) -> np.ndarray:
        coordinates = []

        for color in Color:
            masked_frame = DotTracker.apply_mask(frame, color)
            coordinates += DotTracker._get_dot_position(masked_frame)

        return np.array(coordinates)
