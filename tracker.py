"""
This class processes individual frames and locates dots of a specific color
"""

import cv2
import numpy as np
from enum import Enum


class Filters(Enum):
    """This enum contains tuned filters that work for the test data, not sure if they will generalize"""

    RED_MASK_LOWER = [0, 124, 104]
    RED_MASK_UPPER = [10, 255, 255]

    # RED_MASK_LOWER_2 = [167, 114, 154] #Secondary Red filters which are not needed
    # RED_MASK_UPPER_2 = [179, 255, 255]

    GREEN_MASK_LOWER = [60, 50, 20]
    GREEN_MASK_UPPER = [95, 255, 182]

    YELLOW_MASK_LOWER = [20, 77, 140]
    YELLOW_MASK_UPPER = [30, 255, 255]

    # LARGE artifacts with the white mask
    WHITE_MASK_LOWER = [81, 0, 208]
    WHITE_MASK_UPPER = [179, 60, 255]

    @property
    def np_array(self):
        """Returns the enum as a numpy array"""
        return np.array(self.value, dtype=np.uint8)


class DotTracker:
    """This class contains methods to process individual frames from the spindle video"""

    def __init__(
        self, brush_size: int, lower_filter: np.ndarray, upper_filter: np.ndarray
    ):
        # Brush Size for Cleaning Up Artifacts (larger brush = larger artifacts are cleaned up)
        self.BRUSH_SIZE = brush_size
        self.LOWER_FILTER = lower_filter
        self.UPPER_FILTER = upper_filter
        self.prev_dot_A = None
        self.prev_dot_B = None

    def _apply_mask(self, frame: np.ndarray) -> np.ndarray:
        """Applies a the image mask to the frame and then cleans up the mask"""

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masked_frame = cv2.inRange(hsv_frame, self.LOWER_FILTER, self.UPPER_FILTER)

        kernal = np.ones(
            (self.BRUSH_SIZE, self.BRUSH_SIZE), dtype=np.uint8
        )  # Brush size for cleanup (artifacts smaller than the brush are deleted)

        masked_frame = cv2.morphologyEx(masked_frame, cv2.MORPH_CLOSE, kernal)
        masked_frame = cv2.morphologyEx(masked_frame, cv2.MORPH_OPEN, kernal)

        return masked_frame

    def _get_dot_positions(self, masked_frame: np.ndarray) -> list:
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
                coordinates.append([float(x), float(y), int(radius)])

        return coordinates

    def process_frame(self, frame: np.ndarray, show_debug_frame: bool = False):
        """
        This function takes in a frame in BGR format and runs through the entire pipeline to produce 2 lists, one for each dot
        The lists are in the format, [x, y, radius]. It uses the previous positions of the dots to ensure
        the values returned always correspond to the same dot on repeated uses.
        """
        masked_frame = self._apply_mask(frame)
        unlabeled_coordinates = self._get_dot_positions(masked_frame)
        if len(unlabeled_coordinates) != 2:
            # print("More or less than 2 unlabeled coordinates")
            return None, None

        p1, p2 = unlabeled_coordinates

        if self.prev_dot_A is None or self.prev_dot_B is None:
            if p1[0] < p2[0]:  # Leftmost dot becomes A
                self.prev_dot_A, self.prev_dot_B = p1, p2
            else:
                self.prev_dot_A, self.prev_dot_B = p2, p1

            # print(
            # f"p1: {p1}, p2: {p2}\nPrev Dot A: {self.prev_dot_A}, Prev Dot B: {self.prev_dot_B}\n"
            # )
            return self.prev_dot_A, self.prev_dot_B

        # arrangement 1
        dist_p1_to_A = np.linalg.norm(np.array(p1[:2]) - np.array(self.prev_dot_A[:2]))
        dist_p2_to_B = np.linalg.norm(np.array(p2[:2]) - np.array(self.prev_dot_B[:2]))
        max_dist_arrangement_1 = max(dist_p1_to_A, dist_p2_to_B)

        # arrangement 2
        dist_p2_to_A = np.linalg.norm(np.array(p2[:2]) - np.array(self.prev_dot_A[:2]))
        dist_p1_to_B = np.linalg.norm(np.array(p1[:2]) - np.array(self.prev_dot_A[:2]))
        max_dist_arrangement_2 = max(dist_p2_to_A, dist_p1_to_B)

        if show_debug_frame:
            # Shows a frame that helps debug any tracking errors

            debug_frame = frame.copy()
            cv2.namedWindow("Tracker Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tracker Debug", 300, 400)
            cv2.circle(debug_frame, (int(p1[0]), int(p1[1])), 5, (255, 255, 255), -1)
            cv2.putText(
                debug_frame,
                "p1",
                (int(p1[0]) + 10, int(p1[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.circle(debug_frame, (int(p2[0]), int(p2[1])), 5, (255, 255, 255), -1)
            cv2.putText(
                debug_frame,
                "p2",
                (int(p2[0]) + 10, int(p2[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.circle(
                debug_frame,
                (int(self.prev_dot_A[0]), int(self.prev_dot_A[1])),
                12,
                (0, 0, 255),
                2,
            )
            cv2.circle(
                debug_frame,
                (int(self.prev_dot_B[0]), int(self.prev_dot_B[1])),
                12,
                (255, 0, 0),
                2,
            )
            if max_dist_arrangement_1 < max_dist_arrangement_2:
                cv2.line(
                    debug_frame,
                    (int(self.prev_dot_A[0]), int(self.prev_dot_A[1])),
                    (int(p1[0]), int(p1[1])),
                    (0, 255, 0),
                    2,
                )
                cv2.line(
                    debug_frame,
                    (int(self.prev_dot_B[0]), int(self.prev_dot_B[1])),
                    (int(p2[0]), int(p2[1])),
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.line(
                    debug_frame,
                    (int(self.prev_dot_A[0]), int(self.prev_dot_A[1])),
                    (int(p2[0]), int(p2[1])),
                    (0, 255, 0),
                    2,
                )
                cv2.line(
                    debug_frame,
                    (int(self.prev_dot_B[0]), int(self.prev_dot_B[1])),
                    (int(p1[0]), int(p1[1])),
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Tracker Debug", debug_frame)

        if max_dist_arrangement_1 < max_dist_arrangement_2:
            self.prev_dot_A = p1
            self.prev_dot_B = p2
        else:
            self.prev_dot_A = p2
            self.prev_dot_B = p1

        return self.prev_dot_A, self.prev_dot_B
