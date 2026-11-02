from dataclasses import dataclass
from typing import Callable


import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import welch
from scipy.special import expit

from src.utils_spectral_power import alpha_band, delta_band


@dataclass
class JRParams:
    # Synaptic gains (mV)
    A: float = 3.25
    B: float = 22.0

    # Inverse synaptic time constants (1/s)
    a: float = 100.0
    b: float = 50.0

    # Connectivity constants
    C1: float = 135.0
    C2: float = 108.0
    C3: float = 33.75
    C4: float = 33.75

    # Sigmoid parameters
    e0: float = 2.5  # Maximum sigmoid firing rate = 2*e0
    v0: float = 6.0  # mV
    r: float = 0.56  # 1/mV


def sigmoid(v: float, p: JRParams) -> float:
    """Convert membrane potential (mV) to firing rate (1/s)."""
    return 2.0 * p.e0 * expit(p.r * (v - p.v0))


def jr_rhs(
    t: float, y: np.ndarray, p: JRParams, external_input: Callable[[float], float]
) -> np.ndarray:
    """
    State:
        y[0:3] = y0, y1, y2 (mV)
        y[3:6] = dy0, dy1, dy2 (mV/s)

    external_input(t) returns the external firing rate (1/s).
    """
    y0, y1, y2, dy0, dy1, dy2 = y

    # Second derivatives of the three PSPs
    ddy0 = p.A * p.a * sigmoid(y1 - y2, p) - 2 * p.a * dy0 - p.a**2 * y0

    ddy1 = (
        p.A * p.a * (external_input(t) + p.C2 * sigmoid(p.C1 * y0, p))
        - 2 * p.a * dy1
        - p.a**2 * y1
    )

    ddy2 = p.B * p.b * p.C4 * sigmoid(p.C3 * y0, p) - 2 * p.b * dy2 - p.b**2 * y2

    return np.array([dy0, dy1, dy2, ddy0, ddy1, ddy2])


def simulate_jr(
    duration=10.0,
    fs=1000.0,
    external_rate=120.0,
    params=None,
):
    """Simulate a single Jansen-Rit cortical column."""
    p = params or JRParams()

    t_eval = np.arange(0.0, duration, 1.0 / fs)
    y_init = np.zeros(6)

    solution = solve_ivp(
        jr_rhs,
        t_span=(0.0, duration),
        y0=y_init,
        t_eval=t_eval,
        args=(p, lambda t: external_rate),
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        max_step=1.0 / fs,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    # Net excitatory minus inhibitory PSP
    eeg = solution.y[1] - solution.y[2]

    return solution.t, eeg, solution.y


# TODO unite with utils_spectral_power? but there - average over chans
def alpha_delta_spectral_power(
    times: np.ndarray,
    signal: np.ndarray,
    sfreq: float,
    window_sec=4.0,
    step_sec=1.0,
):
    window_samples = int(round(window_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))

    if len(signal) < window_samples:
        raise ValueError("Signal is shorter than one analysis window.")

    power_times = []
    alpha_power = []
    delta_power = []

    for start in range(0, len(signal) - window_samples + 1, step_samples):
        segment = signal[start : start + window_samples]

        freqs, psd = welch(
            segment,
            fs=sfreq,
            nperseg=window_samples,
        )

        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        delta_mask = (freqs >= delta_band[0]) & (freqs <= delta_band[1])
        if np.count_nonzero(alpha_mask) < 2:
            raise ValueError("Insufficient frequency bins in the alpha band.")

        alpha_power.append(np.trapz(psd[alpha_mask], x=freqs[alpha_mask]))
        delta_power.append(np.trapz(psd[delta_mask], x=freqs[delta_mask]))

        # Center of the current window, in seconds.
        center_sample = start + window_samples / 2
        power_times.append(times[0] + center_sample / sfreq)

    return np.asarray(power_times), np.asarray(alpha_power), np.asarray(delta_power)
