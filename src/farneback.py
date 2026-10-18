import sys
from pathlib import Path

from tqdm import tqdm

from utils_farneback import calc_intensities_farneback, save_txt


def process_directory(directory: Path) -> None:
    """Process all .mp4 files in the given directory.

    For each video, computes Farneback intensities and writes two text files
    (times and intensities) into a "farneback" subdirectory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"no such dir: {directory}")

    if not directory.is_dir():
        raise FileNotFoundError(f"not a dir: {directory}")
    
    output_dir = directory / "farneback"
    output_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() == ".mp4"
    )

    for video_path in tqdm(mp4_files, desc="Processing videos", unit="video"):
        try:
            times, intensities = calc_intensities_farneback(video_path)
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to process {video_path.name}: {e}")
            continue

        times_path = output_dir / f"{video_path.stem}_times.txt"
        intensities_path = output_dir / f"{video_path.stem}_intensities.txt"

        save_txt(times, times_path)
        save_txt(intensities, intensities_path)

        tqdm.write(f"Saved: {times_path.name}, {intensities_path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(sys.argv[0]).name} <directory>")
        sys.exit(1)

    process_directory(Path(sys.argv[1]))
