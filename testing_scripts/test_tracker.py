"""Tests the tracker.py class"""

import numpy as np
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

    red_dot_tracker = DotTracker(
        10, Filters.RED_MASK_LOWER.np_array, Filters.RED_MASK_UPPER.np_array
    )
    yellow_dot_tracker = DotTracker(
        10, Filters.YELLOW_MASK_LOWER.np_array, Filters.YELLOW_MASK_UPPER.np_array
    )
    green_dot_tracker = DotTracker(
        10, Filters.GREEN_MASK_LOWER.np_array, Filters.GREEN_MASK_UPPER.np_array
    )
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            dots = red_dot_tracker.process_frame(frame, show_debug_frame=True)
            dots += yellow_dot_tracker.process_frame(frame)
            dots += green_dot_tracker.process_frame(frame)

            swap = False
            for dot in dots:
                if dot != None:
                    if swap:
                        cv2.circle(frame, (int(dot[0]), int(dot[1])), dot[2], GREEN, 4)
                        cv2.circle(frame, (int(dot[0]), int(dot[1])), 2, GREEN, 8)
                        swap = False

                    else:
                        cv2.circle(frame, (int(dot[0]), int(dot[1])), dot[2], BLUE, 4)
                        cv2.circle(frame, (int(dot[0]), int(dot[1])), 2, BLUE, 8)
                        swap = True

                    cv2.imshow("Masked Frame", frame)

            key = cv2.waitKey(0)
            if key == ord("q"):
                break

        else:
            break

    video.release()


if __name__ == "__main__":
    main()
