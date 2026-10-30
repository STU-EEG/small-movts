from pathlib import Path
from typing import Final, Sequence

import numpy as np
import mne
from scipy.signal import welch

from src.video_utils import parse_timestamp_from_filename


def smoothen(intensities: Sequence | np.ndarray, period: int, time_delta: float) -> tuple[int, list[float]]:
    n_points = int(period // time_delta)
    
    averages = []
    for i in range(len(intensities) - n_points):
        ave = np.average(intensities[i:i + n_points])
        averages.append(ave)

    return n_points, averages

def load_and_merge_intensities(
    directory: Path,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, float]]]:
    """Load all *_intensities.txt files in `directory`, shift each by the
    video's start time (parsed from the filename), merge, and sort by time.

    Returns
    -------
    times : np.ndarray
        Sorted 1D array of absolute times (seconds).
    intensities : np.ndarray
        Corresponding 1D array of intensity values.
    timestamp_pairs : list[tuple[str, float]]
        List of (timestamp_str, total_seconds) for each processed file,
        in the order the files were processed.
    """
    directory = Path(directory)

    intensities_files = sorted(directory.glob("*_intensities.txt"))
    if not intensities_files:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            [],
        )

    all_times = []
    all_intensities = []
    timestamp_pairs: list[tuple[str, float]] = []

    for int_path in intensities_files:
        # Corresponding times file: replace _intensities.txt with _times.txt
        stem = int_path.name[: -len("_intensities.txt")]
        times_path = int_path.with_name(f"{stem}_times.txt")
        if not times_path.is_file():
            print(f"[WARN] Missing times file for {int_path.name}, skipping.")
            continue

        # Parse the video start time from the filename
        time_str, start_sec = parse_timestamp_from_filename(stem)
        if start_sec == 0.0:
            print(f"[WARN] No timestamp in filename for {int_path.name}, skipping.")
            continue

        times = np.loadtxt(times_path)
        intensities = np.loadtxt(int_path)

        if times.shape != intensities.shape:
            print(
                f"[WARN] Shape mismatch for {int_path.name} "
                f"(times={times.shape}, intensities={intensities.shape}), skipping."
            )
            continue

        # Shift relative times to absolute times
        all_times.append(times + start_sec)
        all_intensities.append(intensities)
        timestamp_pairs.append((time_str, start_sec))

    if not all_times:
        return (
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            timestamp_pairs,
        )

    times_merged = np.concatenate(all_times)
    intensities_merged = np.concatenate(all_intensities)

    # Sort ascending by time (stable sort keeps duplicates in order)
    order = np.argsort(times_merged, kind="stable")
    return times_merged[order], intensities_merged[order], timestamp_pairs

alpha_band: Final[tuple[float, float]] = (8.0, 13.0)
delta_band: Final[tuple[float, float]] = (0.5, 4.0)

def alpha_delta_powers(raw: mne.io.BaseRaw, duration_sec: float, peak_center_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, mne.io.Raw]:
    window_sec = 4.0
    step_sec = 1.0

    raw_fragment = raw.copy().crop(
        tmin=peak_center_sec - duration_sec / 2,
        tmax=peak_center_sec + duration_sec / 2,
        include_tmax=False,
    )
    raw_fragment.drop_channels(['ECG  ECG'])

    data = raw_fragment.get_data()  # (n_channels, n_samples)
    sfreq = raw_fragment.info["sfreq"]

    # Compute average band power in sliding windows
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    times = []
    alpha_power = []
    delta_power = []

    for start in range(0, data.shape[1] - window_samples + 1, step_samples):
        segment = data[:, start : start + window_samples]

        # Welch PSD for each channel
        freqs, psd = welch(
            segment,
            fs=sfreq,
            nperseg=window_samples,
            axis=-1,
        )

        # Integrate PSD over each frequency band, then average over channels
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        delta_mask = (freqs >= delta_band[0]) & (freqs <= delta_band[1])

        alpha_power.append(np.trapz(psd[:, alpha_mask], x=freqs[alpha_mask], axis=-1).mean())
        delta_power.append(np.trapz(psd[:, delta_mask], x=freqs[delta_mask], axis=-1).mean())

        times.append((start + window_samples / 2) / sfreq)

    times = np.asarray(times)
    alpha_power = np.asarray(alpha_power)
    delta_power = np.asarray(delta_power)
    return times, alpha_power, delta_power, raw_fragment