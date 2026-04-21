#!/usr/bin/env python3
"""
Video Annotation Tool for Blink Detection Dataset
Allows manual classification of videos into accept/unsure/discard categories
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Video annotation tool for blink detection dataset')
    parser.add_argument('-i', '--input-dir', type=str, default='.',
                        help='Input directory containing .mp4 files (default: current directory)')
    parser.add_argument('-o', '--out-dir', type=str, default='.',
                        help='Output directory for classified videos (default: current directory)')
    parser.add_argument('-s', '--speed-ratio', type=float, default=0.5,
                        help='Playback speed ratio (default: 0.5, meaning half speed)')
    return parser.parse_args()

def setup_output_dirs(out_dir):
    """Create output subdirectories if they don't exist"""
    categories = ['accepted', 'unsure', 'discarded']
    dirs = {}
    for cat in categories:
        cat_dir = Path(out_dir) / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        dirs[cat] = cat_dir
    return dirs

def get_mp4_files(input_dir):
    """Get all .mp4 files from input directory (non-recursive)"""
    input_path = Path(input_dir)
    mp4_files = list(input_path.glob('*.mp4'))
    mp4_files.extend(input_path.glob('*.MP4'))  # Also handle uppercase extension
    return sorted(mp4_files)


def process_video(video_path, output_dirs, speed_ratio):
    """Process a single video file with annotation interface"""
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate delay between frames (ms)
    # Speed ratio: 0.5 means half speed (longer delay)
    original_delay = int(1000 / fps) if fps > 0 else 33
    frame_delay = int(original_delay / speed_ratio) if speed_ratio > 0 else original_delay
    
    # Get window dimensions
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Cannot read frames from {video_path}")
        cap.release()
        return None
    
    height, width = frame.shape[:2]
    
    # Create window
    window_name = f"Annotating: {video_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width * 3, height * 3)
    

    # Main video loop
    result = None
    current_frame_pos = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    while result is None:
        ret, frame = cap.read()
        if not ret:
            # Loop video - restart from beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        

        cv2.imshow(window_name, frame)
        
        # Handle keyboard input
        key = cv2.waitKey(frame_delay) & 0xFF
        
        # Check for key presses
        if key == ord('q'):
            result = 'accepted'
            break
        elif key == ord('w'):
            result = 'unsure'
            break
        elif key == ord('e'):
            result = 'discarded'
            break
        elif key == 27 or key == ord('x'):  # ESC or X to exit program
            print("\nAnnotation interrupted by user")
            return 'exit'
    
    # Cleanup
    cap.release()
    cv2.destroyWindow(window_name)
    
    return result

def move_video(video_path, category, output_dirs):
    """Move video to the appropriate output directory"""
    dest_dir = output_dirs[category]
    dest_path = dest_dir / video_path.name
    
    # Handle duplicate filenames
    if dest_path.exists():
        stem = video_path.stem
        suffix = video_path.suffix
        counter = 1
        while dest_path.exists():
            dest_path = dest_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    
    shutil.move(str(video_path), str(dest_path))
    print(f"  ☑️ Moved to {category}/")

def main():
    args = parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dirs = setup_output_dirs(args.out_dir)
    
    # Get MP4 files
    mp4_files = get_mp4_files(input_dir)
    
    if not mp4_files:
        print(f"No .mp4 files found in {input_dir}")
        return
    
    print(f"\nFound {len(mp4_files)} .mp4 files in {input_dir}")
    print(f"Playback speed: {args.speed_ratio}x")
    print("\nControls:")
    print("  Q - Accept video")
    print("  W - Mark as unsure")
    print("  E - Discard video")
    print("  ESC or X - Exit program\n")
    print("Starting annotation...\n")
    
    # Process each video
    for i, video_path in enumerate(mp4_files, 1):
        print(f"\n[{i}/{len(mp4_files)}] Processing: {video_path.name}")
        
        result = process_video(video_path, output_dirs, args.speed_ratio)
        
        if result == 'exit':
            print("Exiting...")
            break
        elif result:
            move_video(video_path, result, output_dirs)
        else:
            print(f"  😿 Failed to process {video_path.name}")
    
    print("\nAnnotation completed!")
    print(f"Results saved to: {Path(args.out_dir).absolute()}")
    for category, dir_path in output_dirs.items():
        count = len(list(dir_path.glob('*.mp4')))
        print(f"  {category}: {count} videos")

if __name__ == "__main__":
    main()