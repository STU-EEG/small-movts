from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def calc_intensities_farneback(video_path: Path) -> tuple[list[float], list[float]]:
    """
    :returns: (times, intensities)
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    times = []
    intensities = []
    frame_idx = 1
    ret, frame = cap.read()
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude = np.linalg.norm(flow, axis=2)

        # Ignore tiny displacements likely caused by noise
        magnitude[magnitude < 0.3] = 0

        # Mean motion intensity, in pixels per second
        intensity = magnitude.mean() * fps

        times.append(frame_idx / fps)
        intensities.append(intensity)

        prev = curr
        frame_idx += 1

    cap.release()
    return times, intensities


def save_txt(arr: Iterable, out_path: Path) -> None:
    np.savetxt(str(out_path), np.array(arr), fmt="%.6f")
