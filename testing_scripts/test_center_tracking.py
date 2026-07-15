"""Tests the tracker.py class"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from tracker import DotTracker, Filters

PATH_TO_VIDEO = "test_data/spindle_video.mp4"
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def main():

    video = cv2.VideoCapture(PATH_TO_VIDEO)

    try:
        assert video.isOpened()
    except AssertionError:
        print("Could not find video")

    cv2.namedWindow("Masked Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Masked Frame", 600, 800)

    cv2.namedWindow("Contour Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contour Frame", 600, 800)

    center_tracker = DotTracker(
        10,
        Filters.CENTER_MASK_LOWER.np_array,
        Filters.CENTER_MASK_UPPER.np_array,
        "center",
        1,
        10000,
        0.95,
    )

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            masked_frame = center_tracker._apply_mask(frame)
            cv2.imshow("Masked Frame", masked_frame)

            coords = center_tracker.process_frame_center(frame)

            if coords is not None:
                cv2.drawContours(frame, [coords[3]], -1, GREEN, 4)
                cv2.circle(frame, (int(coords[0]), int(coords[1])), 2, GREEN, 8)

            cv2.imshow("Contour Frame", frame)

            key = cv2.waitKey(0)
            if key == ord("q"):
                break

        else:
            break

    video.release()


if __name__ == "__main__":
    main()
