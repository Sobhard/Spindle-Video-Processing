"""Simple script for tuning the HSV Filter"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from tracker import DotTracker

PATH_TO_VIDEO = "test_data/spindle_video.mp4"


def main():
    """Creates 3 windows, one with bars to tune the filter, and two with the frames before and after the mask is applied"""

    video = cv2.VideoCapture(PATH_TO_VIDEO)

    assert video.isOpened()  # Throws an error if the video isnt loaded

    # print(f"Frame Count: {video.get(cv2.CAP_PROP_FRAME_COUNT)}")
    # print(f"FPS: {video.get(cv2.CAP_PROP_FPS)}")

    cv2.namedWindow("Trackbars")

    cv2.createTrackbar("L - H", "Trackbars", 0, 179, lambda: None)
    cv2.createTrackbar("L - S", "Trackbars", 124, 255, lambda: None)
    cv2.createTrackbar("L - V", "Trackbars", 104, 255, lambda: None)
    cv2.createTrackbar("U - H", "Trackbars", 12, 179, lambda: None)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, lambda: None)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, lambda: None)

    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Masked Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 600, 800)
    cv2.resizeWindow("Masked Frame", 600, 800)
    cv2.moveWindow("Frame", 0, 0)

    while video.isOpened():
        ret, frame = video.read()
        if ret:

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lh = cv2.getTrackbarPos("L - H", "Trackbars")
            ls = cv2.getTrackbarPos("L - S", "Trackbars")
            lv = cv2.getTrackbarPos("L - V", "Trackbars")
            uh = cv2.getTrackbarPos("U - H", "Trackbars")
            us = cv2.getTrackbarPos("U - S", "Trackbars")
            uv = cv2.getTrackbarPos("U - V", "Trackbars")

            lowerBound = np.array([lh, ls, lv])
            upperBound = np.array([uh, us, uv])

            mask = cv2.inRange(hsv_frame, lowerBound, upperBound)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("Frame", frame)
            cv2.imshow("Masked Frame", result)

            key = cv2.waitKey(0)
            if key == ord("q"):
                break
        else:
            break

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    main()
