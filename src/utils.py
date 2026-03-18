from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist


def nw_smoothing(x: Sequence[int], y: Sequence[np.float32], sigma: float=1) -> list[np.float32]:
    x_np = np.array(x, dtype=np.float32).reshape(-1, 1)
    y_np = np.array(y, dtype=np.float32)

    dist_matrix = cdist(x_np, x_np, metric='euclidean')
    
    # Apply Gaussian kernel: K(u) = exp(-u^2/(2 sigma^2))
    weights = np.exp(-(dist_matrix ** 2) / (2 * sigma ** 2))
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_normalized = weights / weights_sum
    
    smoothed = np.sum(weights_normalized * y_np.reshape(1, -1), axis=1)
    return smoothed.tolist()