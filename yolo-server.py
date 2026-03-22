from pathlib import Path
from src.utils import crop_head, CropHeadParams

data_dir = Path("/share/Share/valera/veeg_june19_2025_vzuev/vzuev_june18_2025_Video_2")
for video_path in data_dir.glob("*s.mp4"):
    crop_head(CropHeadParams(
        in_path=video_path,
        yolo_path="notebooks/yolo11m-pose.pt",
        plot_centers=False,
    ))