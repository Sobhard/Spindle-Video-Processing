"""
This class processes individual frames and locates dots of a specific color
"""

import cv2
import numpy as np
from enum import Enum
import csv


class Filters(Enum):
    """This enum contains tuned filters that work for the test data, not sure if they will generalize"""

    RED_MASK_LOWER = [0, 124, 104]  # 124
    RED_MASK_UPPER = [15, 255, 255]

    # RED_MASK_LOWER_2 = [167, 114, 154] #Secondary Red filters which are not needed
    # RED_MASK_UPPER_2 = [179, 255, 255]

    GREEN_MASK_LOWER = [60, 40, 30]
    GREEN_MASK_UPPER = [95, 255, 182]

    YELLOW_MASK_LOWER = [20, 77, 140]
    YELLOW_MASK_UPPER = [30, 255, 255]

    # LARGE artifacts with the white mask
    WHITE_MASK_LOWER = [81, 0, 208]
    WHITE_MASK_UPPER = [179, 60, 255]

    CENTER_MASK_LOWER = [44, 0, 0]
    CENTER_MASK_UPPER = [163, 71, 255]

    @property
    def np_array(self):
        """Returns the enum as a numpy array"""
        return np.array(self.value, dtype=np.uint8)


class DotTracker:
    """This class contains methods to process individual frames from the spindle video

    **min_contour_area**: Any objects smaller than this area (in pixels^2) are filtered out (default=1000)\n
    **brush_size**: Any fragments or artifacts left after the mask that are smaller than the brush_size are filtered out\n
    **circularity**: Any fragments with a circularity less than this threshold are filtered out (default=0.84)
    """

    def __init__(
        self,
        brush_size: int,
        lower_filter: np.ndarray,
        upper_filter: np.ndarray,
        name: str,
        num_dots: int = 2,
        min_contour_area: int = 1000,
        min_circularity: float = 0.80,
    ):
        # Brush Size for Cleaning Up Artifacts (larger brush = larger artifacts are cleaned up)
        self.BRUSH_SIZE = brush_size
        self.MIN_CONTOUR_AREA = min_contour_area
        self.MIN_CIRCULARITY = min_circularity
        self.LOWER_FILTER = lower_filter
        self.UPPER_FILTER = upper_filter
        self.prev_dot_A = None
        self.prev_dot_B = None
        self.contour = 0
        self.num_dots = num_dots
        self.name = name

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

    def _find_circle_center(self, masked_frame: np.ndarray) -> list:
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
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                coordinates.append([float(x), float(y), int(radius)])

        return coordinates

    def _calc_circularity(self, contour, area) -> float:
        """Simple function to calculate circularity"""
        perimeter = cv2.arcLength(contour, True)
        return 4 * np.pi * area / (perimeter**2)

    def _find_centroid(
        self,
        masked_frame: np.ndarray,
        min_contour_area: int,
        min_circularity: float,
        use_hull: bool = True,
        show_debug_frame: bool = False,
    ) -> list:
        """
        Runs an algorithm on a clean masked frame to find the centriod of a dot
        1. Finds the contours in a masked image
        2. Any contours smaller than the min contour area will be discarded
        3. Finds the convex hull of the contours (if use_hull=False we just use the raw contours)
        4. Finds the centroid of the contour

        Args:
            Clean masked image
        Returns:
            Two dimensional list of x, y positions of the centroid and the area of the convex hull
            [[x1, y1, area1, contour], [x2, y2, area2, contour]]
            Returns an empty list if no centroids found

        """

        contours, _ = cv2.findContours(
            masked_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        coordinates = []

        for contour in contours:

            if use_hull:
                contour = cv2.convexHull(contour)

            contour_area = cv2.contourArea(contour)
            circularity = self._calc_circularity(contour, contour_area)

            if contour_area > min_contour_area and circularity > min_circularity:

                # Use moments to find centroid
                moments = cv2.moments(contour)

                if abs(moments["m00"]) < 1e-8:
                    continue

                c_x = moments["m10"] / moments["m00"]
                c_y = moments["m01"] / moments["m00"]

                coordinates.append(
                    [float(c_x), float(c_y), float(contour_area), contour]
                )

        if show_debug_frame:
            debug_img = np.zeros_like(masked_frame)
            for obj in coordinates:
                cv2.drawContours(debug_img, obj[3], -1, 255, 2)
                cv2.circle(debug_img, (int(obj[0]), int(obj[1])), 4, 128, -1)
                cv2.imshow("Contour Drawing", debug_img)

        return coordinates

    def process_frame_center(self, frame: np.ndarray):
        """
        This function processes the frame, but only for the center, so it does not need tracking logic. Throws an error when used in a non-single dot pipeline.

        **For best results, use with a very high circularity**

        1. Masks frame
        2. finds contours
        3. Returns biggest contour
        """

        try:
            assert self.num_dots == 1
        except AssertionError:
            print("process_frame_center should only be used when tracking a single dot")
            return

        masked_frame = self._apply_mask(frame)
        unlabeled_coords = self._find_centroid(
            masked_frame, self.MIN_CONTOUR_AREA, self.MIN_CIRCULARITY, True
        )

        if not unlabeled_coords:
            return None

        max_area = unlabeled_coords[0]

        for n in unlabeled_coords:
            if n[2] > max_area[2]:
                max_area = n

        return max_area

    def process_frame_two_dots(
        self,
        frame: np.ndarray,
        show_tracking_debug: bool = False,
        show_centroid_debug: bool = False,
    ):
        """
        This function takes in a frame in BGR format and runs through the entire pipeline to produce 2 lists, one for each dot
        The lists are in the format, [x, y, area, contour]. It uses the previous positions of the dots to ensure
        the values returned always correspond to the same dot on repeated uses.
        """
        masked_frame = self._apply_mask(frame)
        unlabeled_coordinates = self._find_centroid(
            masked_frame,
            self.MIN_CONTOUR_AREA,
            self.MIN_CIRCULARITY,
            True,
            show_centroid_debug,  # TODO: Rework the centroid debug frame so it is actually useful
        )

        if len(unlabeled_coordinates) != 2:
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
        dist_p1_to_B = np.linalg.norm(np.array(p1[:2]) - np.array(self.prev_dot_B[:2]))
        max_dist_arrangement_2 = max(dist_p2_to_A, dist_p1_to_B)

        if show_tracking_debug:
            # Shows a frame that helps debug any tracking errors

            debug_frame = frame.copy()
            cv2.namedWindow("Tracker Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tracker Debug", 600, 800)
            cv2.circle(debug_frame, (int(p1[0]), int(p1[1])), 5, (255, 255, 255), -1)
            cv2.putText(
                debug_frame,
                f"p1: ({int(p1[0])}, {int(p1[1])}), p2: ({int(p2[0])}, {int(p2[1])})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                debug_frame,
                f"prev_dot_A: ({int(self.prev_dot_A[0])}, {int(self.prev_dot_A[1])}) prev_dot_B: ({int(self.prev_dot_B[0])}, {int(self.prev_dot_B[1])})",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                debug_frame,
                f"dist_p1_to_A: {int(dist_p1_to_A)}, dist_p2_to_B {int(dist_p1_to_B)}, max_dist_arrangement_1: {int(max_dist_arrangement_1)}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                debug_frame,
                f"dist_p1_to_B: {int(dist_p1_to_B)}, dist_p2_to_A {int(dist_p1_to_A)}, max_dist_arrangement_2: {int(max_dist_arrangement_2)}",
                (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                3,
            )
            cv2.putText(
                debug_frame,
                "p1",
                (int(p1[0]) + 10, int(p1[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

            cv2.circle(debug_frame, (int(p2[0]), int(p2[1])), 5, (255, 255, 255), -1)
            cv2.putText(
                debug_frame,
                "p2",
                (int(p2[0]) + 10, int(p2[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
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


class SpindleVideoProcessor:
    """
    Manages the video reading, coordinates the color trackers,
    and handles exporting the data to a CSV file.

    **brush_size**:  Any fragments or artifacts left after the mask that are smaller than the brush_size are filtered out
    **precision**: Decimal places that values are rounded to
    """

    def __init__(self, brush_size: int = 5, precision: int = 3):
        self.red_tracker = DotTracker(
            brush_size=brush_size,
            lower_filter=Filters.RED_MASK_LOWER.np_array,
            upper_filter=Filters.RED_MASK_UPPER.np_array,
            name="RED",
        )
        self.green_tracker = DotTracker(
            brush_size=brush_size,
            lower_filter=Filters.GREEN_MASK_LOWER.np_array,
            upper_filter=Filters.GREEN_MASK_UPPER.np_array,
            name="GREEN",
        )
        self.yellow_tracker = DotTracker(
            brush_size=brush_size,
            lower_filter=Filters.YELLOW_MASK_LOWER.np_array,
            upper_filter=Filters.YELLOW_MASK_UPPER.np_array,
            name="YELLOW",
        )

        self.center_tracker = DotTracker(
            brush_size=brush_size,
            lower_filter=Filters.CENTER_MASK_LOWER.np_array,
            upper_filter=Filters.CENTER_MASK_UPPER.np_array,
            name="CENTER",
            num_dots=1,
            min_circularity=0.95,
            min_contour_area=10000,
        )

        self.decimal_precision = precision
        self.missing_values = 0

    def _extract_xyArea(self, dot):
        """Helper method to safely extract (x, y, area) and handle None values."""
        if dot is None:
            self.missing_values += 3
            return ["", "", ""]  # Leave CSV cell blank if dot was lost
        return [round(val, self.decimal_precision) for val in dot[:3]]

    def format_data(self, output_path: str, timestamp: float, coordinates: list):
        """
        Writes a single row of tracked coordinates to the CSV file.
        """
        with open(output_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            row = [round(timestamp, self.decimal_precision)] + coordinates
            writer.writerow(row)

    def process_video(self, video_path: str, output_csv_path: str, frequency: int = 30):
        """
        Processes the video frame-by-frame. Tracks colors at native framerate
        to maintain mathematical state, but exports data at the requested frequency.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine the time intervals for our requested output frequency
        output_interval = 1.0 / frequency
        next_output_time = 0.0
        frame_count = 0

        with open(output_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "timestamp",
                    "red_x1",
                    "red_y1",
                    "red_area1",
                    "red_x2",
                    "red_y2",
                    "red_area2",
                    "green_x1",
                    "green_y1",
                    "green_area1",
                    "green_x2",
                    "green_y2",
                    "green_area2",
                    "yellow_x1",
                    "yellow_y1",
                    "yellow_area1",
                    "yellow_x2",
                    "yellow_y2",
                    "yellow_area2",
                    "center_x",
                    "center_y",
                    "center_area",
                ]
            )

        print(
            f"Processing video at {fps} native FPS. Outputting data at {frequency} Hz..."
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            red_A, red_B = self.red_tracker.process_frame_two_dots(frame)
            green_A, green_B = self.green_tracker.process_frame_two_dots(frame)
            yellow_A, yellow_B = self.yellow_tracker.process_frame_two_dots(frame)
            center = self.center_tracker.process_frame_center(frame)

            current_time = frame_count / fps

            if current_time >= next_output_time:
                row_coords = []
                row_coords.extend(self._extract_xyArea(red_A))
                row_coords.extend(self._extract_xyArea(red_B))
                row_coords.extend(self._extract_xyArea(green_A))
                row_coords.extend(self._extract_xyArea(green_B))
                row_coords.extend(self._extract_xyArea(yellow_A))
                row_coords.extend(self._extract_xyArea(yellow_B))
                row_coords.extend(self._extract_xyArea(center))

                self.format_data(output_csv_path, current_time, row_coords)

                next_output_time += output_interval

            frame_count += 1

            # Optional: Print progress to terminal
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")

        cap.release()
        print(
            f"Processing complete! Data saved to {output_csv_path}\nMissing Values = {self.missing_values}"
        )
