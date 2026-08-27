from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import pickle

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from ultralytics.models.yolo import YOLO


def nw_smoothing(x: Sequence[int], y: Sequence[np.float32], sigma: float) -> list[np.float32]:
    x_np = np.array(x, dtype=np.float32).reshape(-1, 1)
    y_np = np.array(y, dtype=np.float32)

    dist_matrix = cdist(x_np, x_np, metric='euclidean')
    
    # Apply Gaussian kernel: K(u) = exp(-u^2/(2 sigma^2))
    weights = np.exp(-(dist_matrix ** 2) / (2 * sigma ** 2))
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_normalized = weights / weights_sum
    
    smoothed = np.sum(weights_normalized * y_np.reshape(1, -1), axis=1)
    return smoothed.tolist()

def smooth(frms: Sequence[int], xy:Sequence[tuple[np.float32, np.float32]], sigma:float) -> list[tuple[np.float32, np.float32]]:
     xs = nw_smoothing(frms, [xy_i[0] for xy_i in xy], sigma=sigma)
     ys = nw_smoothing(frms, [xy_i[1] for xy_i in xy], sigma=sigma)
     return list(zip(xs, ys))


class KeypointsInfo:
    head_centers: dict[int, list[tuple[int, int]]] = defaultdict(list)  # {person_idx: [(x, y), ... (for each frame)] }
    shoulder_distances: dict[int, list[np.float32]] = defaultdict(list)  # {person_idx: distances for each frame }
    frame_indices: dict[int, list[int]] = defaultdict(list)

    max_frame_idx: int = 0

    def __init__(self) -> None:
        self.head_centers = defaultdict(list)
        self.shoulder_distances = defaultdict(list)
        self.frame_indices = defaultdict(list)

@dataclass(frozen=True)
class CropHeadParams:
    in_path: Path
    yolo_path: str
    plot_centers: bool = False
    sigma: float = 1.

def track_people(params: CropHeadParams) -> KeypointsInfo:
    model = YOLO(str(params.yolo_path))
    result = KeypointsInfo()

    frame_idx = -1
    for tracked in model.track(source=str(params.in_path), stream=True, tracker="bytetrack.yaml"):
        frame_idx += 1

        if tracked.keypoints is not None and tracked.keypoints.data is not None:
            keypoints_arr = tracked.keypoints.data.cpu().numpy()
            for i_kp, keypoints in enumerate(keypoints_arr):
            
                left_eye = keypoints[1][:2] if keypoints[1][2] > 0 else None
                right_eye = keypoints[2][:2] if keypoints[2][2] > 0 else None
                nose = keypoints[0][:2] if keypoints[0][2] > 0 else None

                head_points = []
                if left_eye is not None:
                    head_points.append(left_eye)
                if right_eye is not None:
                    head_points.append(right_eye)
                if nose is not None:
                    head_points.append(nose)
                
                if head_points:
                    head_center = np.mean(head_points, axis=0).astype(int)
                    result.head_centers[i_kp].append(tuple(head_center.astype(int)))
                    result.frame_indices[i_kp].append(frame_idx)
                
                left_shoulder = keypoints[5][:2] if keypoints[5][2] > 0 else None
                right_shoulder = keypoints[6][:2] if keypoints[6][2] > 0 else None

                if left_shoulder is not None and right_shoulder is not None:
                    distance = np.linalg.norm(left_shoulder - right_shoulder)
                    result.shoulder_distances[i_kp].append(distance)

    result.max_frame_idx = frame_idx
    return result


def crop_head(params: CropHeadParams) -> None:  # TODO unify code with `VideoProcessor.crop_video_segment`
    cap = cv2.VideoCapture(str(params.in_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    data_dir = params.in_path.parent
    file_prefix = params.in_path.stem
    fourcc = cv2.VideoWriter_fourcc(*"avc1")

    kp_info: KeypointsInfo = track_people(params)
    with open(data_dir /f'{file_prefix}.keypoints.pkl', 'wb') as f:
        pickle.dump(kp_info, f)

    for person_idx in kp_info.frame_indices.keys():
        if len(kp_info.frame_indices[person_idx]) < kp_info.max_frame_idx / 2:
            print(f"insufficient number of frames for person {person_idx} (less than half)")
            continue

        shoulder_distances: list = kp_info.shoulder_distances[person_idx]
        head_centers: list[tuple] = kp_info.head_centers[person_idx]
        frame_indices: list[int] = kp_info.frame_indices[person_idx]

        smoothed_centers: list[tuple[np.float32, np.float32]] = smooth(frame_indices, head_centers, params.sigma)
        shoulder_distance_ave = np.mean(shoulder_distances)

        output_yolo_path = data_dir / f"{file_prefix}.yolo{person_idx}.mp4"
        crop_size = int(shoulder_distance_ave * 0.7)
        writer = cv2.VideoWriter(str(output_yolo_path), fourcc, fps, (crop_size, crop_size), isColor=True)

        cap = cv2.VideoCapture(str(params.in_path))

        for i, frame_idx in enumerate(sorted(frame_indices)):
            print(f"Processing frame {frame_idx} ({i+1}/{len(frame_indices)}) for person {person_idx}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            current_smoothed = smoothed_centers[i]
            center_x, center_y = current_smoothed

            x_start = int(max(0, center_x - crop_size // 2))
            x_end = int(min(w, x_start + crop_size))
            y_start = int(max(0, center_y - crop_size // 2))
            y_end = int(min(h, y_start + crop_size))

            # Adjust if crop window goes out of bounds
            if x_end - x_start < crop_size:
                if x_start == 0:
                    x_end = min(w, crop_size)
                else:
                    x_start = max(0, w - crop_size)
            if y_end - y_start < crop_size:
                if y_start == 0:
                    y_end = min(h, crop_size)
                else:
                    y_start = max(0, h - crop_size)

            cropped_frame = frame[y_start:y_end, x_start:x_start + crop_size]
            cropped_frame = cv2.resize(cropped_frame, (crop_size, crop_size))

            if params.plot_centers:
                # Plot head center on the cropped frame
                head_x, head_y = head_centers[i]
                head_x_rel = int(head_x - x_start)
                head_y_rel = int(head_y - y_start)
                if 0 <= head_x_rel < crop_size and 0 <= head_y_rel < crop_size:
                    cv2.circle(cropped_frame, (head_x_rel, head_y_rel), 5, (0, 255, 0), -1)  # Green for original

                # Plot smoothed center
                smooth_x_rel = int(center_x - x_start)
                smooth_y_rel = int(center_y - y_start)
                if 0 <= smooth_x_rel < crop_size and 0 <= smooth_y_rel < crop_size:
                    cv2.circle(cropped_frame, (smooth_x_rel, smooth_y_rel), 5, (0, 0, 255), -1)  # Red for smoothed
                    cv2.drawMarker(cropped_frame, (smooth_x_rel, smooth_y_rel), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

            writer.write(cropped_frame)

        cap.release()
        writer.release()


def load_keypoints(pkl_file: Path) -> KeypointsInfo:
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)

def parse_timestamp_from_filename(filename: str) -> tuple[str, float]:
        """Extract timestamp from video filename."""
        # Pattern: vzuev_june18_2025__17-JUN-2025_22h05m27.475s.mp4
        pattern = r'(\d{2}h\d{2}m\d{2}\.\d{3}s)'
        match = re.search(pattern, filename)
        if match:
            timestamp_str = match.group(1)
            # Convert to seconds
            hours = int(timestamp_str[:2])
            if hours < 12:  # after midnight
                  hours += 24
            minutes = int(timestamp_str[3:5])
            seconds = float(timestamp_str[6:-1])
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return timestamp_str, total_seconds
        return "", 0.0

def get_average_head_position_and_shoulder_distance(kp_info, start_time: float, 
                                                    end_time: float, fps: float = 30.0) -> tuple[tuple[int, int], float]:
        """Get average head position and shoulder distance within time window."""
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Collect all head positions and shoulder distances within the window
        all_head_positions = []
        all_shoulder_distances = []
        
        for person_idx in kp_info.head_centers:
            # Find frames for this person within the window
            frames = kp_info.frame_indices[person_idx]
            for i, frame_idx in enumerate(frames):
                if start_frame <= frame_idx <= end_frame:
                    if i < len(kp_info.head_centers[person_idx]):
                        all_head_positions.append(kp_info.head_centers[person_idx][i])
                    if i < len(kp_info.shoulder_distances[person_idx]):
                        all_shoulder_distances.append(kp_info.shoulder_distances[person_idx][i])
        
        if not all_head_positions or not all_shoulder_distances:
            return (0, 0), 100.0  # Default values
        
        # Average head position
        avg_head_x = int(np.mean([pos[0] for pos in all_head_positions]))
        avg_head_y = int(np.mean([pos[1] for pos in all_head_positions]))
        
        # Average shoulder distance
        avg_shoulder_distance = float(np.mean(all_shoulder_distances))
        
        return (avg_head_x, avg_head_y), avg_shoulder_distance

class VideoProcessor:
    def __init__(self, video_dir: Path, output_dir: Path, 
                 consensus_file: Path, fp1_file: Path, fp2_file: Path, logger: logging.Logger):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        
        self.start_offset: float = parse_timestamp_from_filename(consensus_file.stem)[1]
        
        # Load blink times from the three files
        assert consensus_file,exists() and fp1_file.exists() and fp2_file.exists()
        self.consensus_times = np.loadtxt(consensus_file) + self.start_offset
        self.fp1_times = np.loadtxt(fp1_file) + self.start_offset
        self.fp2_times = np.loadtxt(fp2_file) + self.start_offset
        
        # Combine all blink times
        self.all_blink_times = np.sort(np.unique(np.concatenate([
            self.consensus_times, self.fp1_times, self.fp2_times
        ])))
        
        logger.info(f"Loaded {len(self.consensus_times)} consensus blinks")
        logger.info(f"Loaded {len(self.fp1_times)} FP1 blinks")
        logger.info(f"Loaded {len(self.fp2_times)} FP2 blinks")
        logger.info(f"Total unique blinks: {len(self.all_blink_times)}")
        self.logger = logger
    
    def crop_video_segment(self, video_path: Path, start_time_rel: float, end_time_rel: float,
                          head_center: Tuple[int, int], half_size: int, output_path: Path):
        """Crop a segment from video around the head.
        
        Args:
            start_time_rel: Start time relative to video start (seconds)
            end_time_rel: End time relative to video start (seconds)
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time_rel * fps)
        end_frame = int(end_time_rel * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        crop_size = half_size * 2
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_size, crop_size), isColor=True)
        
        frame_count = 0
        while frame_count <= (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            center_x, center_y = head_center
            
            x_start = int(max(0, center_x - half_size))
            x_end = int(min(w, x_start + crop_size))
            y_start = int(max(0, center_y - half_size))
            y_end = int(min(h, y_start + crop_size))
            
            # Adjust if crop window goes out of bounds
            if x_end - x_start < crop_size:
                if x_start == 0:
                    x_end = min(w, crop_size)
                else:
                    x_start = max(0, w - crop_size)
            if y_end - y_start < crop_size:
                if y_start == 0:
                    y_end = min(h, crop_size)
                else:
                    y_start = max(0, h - crop_size)
            
            cropped_frame = frame[y_start:y_end, x_start:x_start + crop_size]
            cropped_frame = cv2.resize(cropped_frame, (crop_size, crop_size))
            writer.write(cropped_frame)
            
            frame_count += 1
        
        cap.release()
        writer.release()
    
    def find_blinks_in_video(self, video_start: float, video_end: float) -> np.ndarray:
        """Find consensus blinks within video time range (excluding first/last 0.5s)."""
        # Exclude first and last 0.5 seconds
        valid_start = video_start + 0.5
        valid_end = video_end - 0.5
        
        blinks = self.consensus_times[
            (self.consensus_times >= valid_start) & 
            (self.consensus_times <= valid_end)
        ]
        return blinks
    
    def find_non_overlapping_intervals(self, video_start: float, video_end: float,
                                       interval_duration: float = 3.0,
                                       min_gap: float = 3.0) -> List[Tuple[float, float]]:
        """Find non-overlapping 3-second intervals without any blinks"""
        # Get all blinks within video range
        all_blinks = self.all_blink_times[
            (self.all_blink_times >= video_start) & 
            (self.all_blink_times <= video_end)
        ]
        
        if len(all_blinks) == 0:
            # No blinks in entire video, return middle of video
            mid = (video_start + video_end) / 2
            return [(mid - interval_duration/2, mid + interval_duration/2)]
        
        intervals = []
        current_time = video_start
        
        for blink_time in all_blinks:
            # Check if there's a gap before this blink
            if blink_time - current_time >= interval_duration:
                # Add an interval in the middle of the gap
                mid_point = (current_time + blink_time) / 2
                interval_start = mid_point - interval_duration/2
                interval_end = interval_start + interval_duration
                
                # Ensure interval is within video bounds
                if interval_start >= video_start and interval_end <= video_end:
                    intervals.append((interval_start, interval_end))
            
            # Move current time past the blink plus min_gap
            current_time = blink_time + min_gap
        
        # Check after the last blink
        if video_end - current_time >= interval_duration:
            mid_point = (current_time + video_end) / 2
            interval_start = mid_point - interval_duration/2
            interval_end = interval_start + interval_duration
            if interval_start >= video_start and interval_end <= video_end:
                intervals.append((interval_start, interval_end))
        
        return intervals[:5]  # Limit to 5 intervals to avoid imbalance
    
    def process_video(self, video_file: Path):
        """Process a single video file."""
        self.logger.info(f"Processing video: {video_file.name}")
        
        # Get video prefix (without extension)
        video_prefix = video_file.stem
        pkl_file = self.video_dir / f"{video_prefix}.keypoints.pkl"
        
        if not pkl_file.exists():
            self.logger.error(f"Keypoints file not found: {pkl_file}")
            return
        
        # Load keypoints
        kp_info = load_keypoints(pkl_file)
        
        # Get video start timestamp and calculate video time range
        _, video_start_offset = parse_timestamp_from_filename(video_file.name)
        video_end_offset = video_start_offset + 300  # 5 minutes = 300 seconds
        
        # Process blink videos (only from consensus file)
        blink_times = self.find_blinks_in_video(video_start_offset, video_end_offset)
        
        for i, blink_time in enumerate(blink_times):
            # Crop around blink: +/- 0.5 seconds
            crop_start_abs = blink_time - 0.5
            crop_end_abs = blink_time + 0.5
            
            # Convert to relative time within video
            crop_start_rel = crop_start_abs - video_start_offset
            crop_end_rel = crop_end_abs - video_start_offset
            
            # Get average head position and shoulder distance in this 1-second window
            head_center, avg_shoulder = get_average_head_position_and_shoulder_distance(
                kp_info, crop_start_rel, crop_end_rel, fps=30.0
            )

            if head_center == (0, 0):
                logging.warning("head center is 0, skipping")
                continue
            
            size_fraction = int(avg_shoulder * 0.3)
            if size_fraction < 10:  # Minimum size
                size_fraction = 50
            
            # Output filename
            output_filename = f"b_{blink_time:.3f}s_{video_prefix}.mp4"
            output_path = self.output_dir / output_filename
            
            self.logger.info(f"Processing blink at {blink_time:.3f}s: {output_filename}")
            self.crop_video_segment(video_file, crop_start_rel, crop_end_rel, 
                                   head_center, size_fraction, output_path)
        
        # Find and process non-overlapping intervals
        intervals = self.find_non_overlapping_intervals(video_start_offset, video_end_offset)
        
        for i, (interval_start_abs, interval_end_abs) in enumerate(intervals):
            # Use middle 1 second of the 3-second interval
            mid_point = (interval_start_abs + interval_end_abs) / 2
            crop_start_abs = mid_point - 0.5
            crop_end_abs = mid_point + 0.5
            
            # Convert to relative time within video
            crop_start_rel = crop_start_abs - video_start_offset
            crop_end_rel = crop_end_abs - video_start_offset
            
            # Get average head position and shoulder distance
            head_center, avg_shoulder = get_average_head_position_and_shoulder_distance(
                kp_info, crop_start_rel, crop_end_rel, fps=30.0
            )
            if head_center == (0, 0):
                self.logger.warning("head_center is 0, skipping")
                continue
            
            size_fraction = int(avg_shoulder * 0.3)
            if size_fraction < 10:
                size_fraction = 50
            
            # Output filename
            output_filename = f"nonblink_{i}_{video_prefix}.mp4"
            output_path = self.output_dir / output_filename
            
            self.logger.info(f"Processing non-blink interval {i}: {output_filename}")
            self.crop_video_segment(video_file, crop_start_rel, crop_end_rel, 
                                   head_center, size_fraction, output_path)
    
    def process_all_videos(self):
        """Process all videos in the video directory."""
        # Find all video files
        video_files = sorted(self.video_dir.glob("*.mp4"))
        
        if not video_files:
            self.logger.error(f"No video files found in {self.video_dir}")
            return
        
        for video_file in video_files:
            # Skip files that don't match the pattern
            if not video_file.stem.endswith("s"):
                continue
            self.process_video(video_file)
