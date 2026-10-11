from collections import defaultdict
import re

import numpy as np


class KeypointsInfo:
    head_centers: dict[int, list[tuple[int, int]]] = defaultdict(list)  # {person_idx: [(x, y), ... (for each frame)] }
    shoulder_distances: dict[int, list[np.float32]] = defaultdict(list)  # {person_idx: distances for each frame }
    frame_indices: dict[int, list[int]] = defaultdict(list)

    max_frame_idx: int = 0

    def __init__(self) -> None:
        self.head_centers = defaultdict(list)
        self.shoulder_distances = defaultdict(list)
        self.frame_indices = defaultdict(list)

def parse_timestamp_from_filename(filename: str) -> tuple[str, float]:
        """Extract timestamp from video filename."""
        # Pattern example: <...>22h05m27.475s.mp4
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

def get_average_head_position_and_shoulder_distance(kp_info: KeypointsInfo, start_time: float, 
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
