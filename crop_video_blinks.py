import logging
from pathlib import Path

from src.utils import VideoProcessor

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # data_dir = Path("data/real_data/big_edfs")
    # video_dir = data_dir / "videos"

    # data_dir = Path("/share/Share/valera/veeg_june19_2025_vzuev")
    # video_dir = data_dir / "vzuev_june18_2025_Video_2"

    data_dir = Path("/share/Share/valera/veeg_july19_2025_vkuklin")
    video_dir = data_dir / "vkuklin_july19_2025_Video_2"

    # prefix = "vzuev_june18_2025 17-JUN-2025_22h05m27.475s_eog_"
    prefix = "vkuklin_july19_2025 18-JUL-2025_21h28m18.382s_eog_"
    consensus_path = data_dir / f"{prefix}consensus.txt"
    fp1_path = data_dir / f"{prefix}FP1-nomatch.txt"
    fp2_path = data_dir / f"{prefix}FP2-nomatch.txt"

    output_dir = data_dir / "videos_cropped"
    processor = VideoProcessor(video_dir, output_dir, consensus_path,
                               fp1_path, fp2_path, logger)
    processor.process_all_videos()


if __name__ == "__main__":
    main()
