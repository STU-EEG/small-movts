from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import pickle

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from ultralytics.models.yolo import YOLO
from ultralytics.engine.results import Keypoints


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
            keypoints_arr = np.array(tracked.keypoints.data)
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


def crop_head(params: CropHeadParams) -> None:
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
    
