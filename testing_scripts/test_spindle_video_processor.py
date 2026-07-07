from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tracker import SpindleVideoProcessor

if __name__ == "__main__":
    pipeline = SpindleVideoProcessor(brush_size=10)

    pipeline.process_video(
        video_path="test_data/spindle_video.mp4",
        output_csv_path="results.csv",
        frequency=30,
    )
