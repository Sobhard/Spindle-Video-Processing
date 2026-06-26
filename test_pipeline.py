"""Runs the video processing pipeline locally from a specific test video file"""

import cv2
import numpy as np

PATH_TO_VIDEO = "test_data/spindle_video.mp4"


def main():
    """Tests the dot tracker by feeding it frames from the test video. This method is only meant for testing"""

    video = cv2.VideoCapture(PATH_TO_VIDEO)

    assert video.isOpened()  # Throws an error if the video isnt loaded

    print(f"Frame Count: {video.get(cv2.CAP_PROP_FRAME_COUNT)}")
    print(f"FPS: {video.get(cv2.CAP_PROP_FPS)}")

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            cv2.imshow("Frame", frame)

            key = cv2.waitKey(20)

            if key == ord("q"):
                break
        else:
            break

    video.release()


if __name__ == "__main__":
    main()
