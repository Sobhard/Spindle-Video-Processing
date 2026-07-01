"""Tests the tracker.py class"""

import numpy as np
import cv2
from tracker import DotTracker

PATH_TO_VIDEO = "test_data/spindle_video.mp4"
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def main():

    video = cv2.VideoCapture(PATH_TO_VIDEO)

    try:
        assert video.isOpened()
    except AssertionError:
        print("Could not find video")

    cv2.namedWindow("Original Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original Frame", 300, 400)
    cv2.moveWindow("Original Frame", -20, 120)

    cv2.namedWindow("Masked Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Masked Frame", 300, 400)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            cv2.imshow("Original Frame", frame)

            circle_positions = DotTracker.process_frame(frame)
            for i, coordinates in enumerate(circle_positions):
                x = coordinates[0]
                y = coordinates[1]
                radius = coordinates[2]

                if i == 0:
                    cv2.circle(frame, (x, y), radius, GREEN, 4)
                    cv2.circle(frame, (x, y), 2, GREEN, 8)

                elif i == 1:
                    cv2.circle(frame, (x, y), radius, BLUE, 4)
                    cv2.circle(frame, (x, y), 2, BLUE, 8)

                else:
                    cv2.circle(frame, (x, y), radius, BLUE, 4)
                    cv2.circle(frame, (x, y), 2, BLUE, 8)

            cv2.imshow("Masked Frame", frame)

            key = cv2.waitKey(0)
            if key == ord("q"):
                break

        else:
            break

    video.release()


if __name__ == "__main__":
    main()
