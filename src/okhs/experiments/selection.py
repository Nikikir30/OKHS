"""Standalone row selector matching the benchmark's choice."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class CURSelection:
    """Select highest norm rows after column-wise MinMax scaling."""

    def __init__(self, params: dict):
        """Store the requested requested rank."""
        self.rank = int(params["rank"])
        self.row_indices = None

    def fit_transform(self, matrix: np.ndarray):
        """Find row indices used as occupation-kernel basis trajectories."""
        scaled = MinMaxScaler().fit_transform(np.nan_to_num(np.asarray(matrix)))
        row_norms = np.linalg.norm(scaled, axis=1)
        self.row_indices = np.sort(np.argsort(row_norms)[-min(self.rank, len(row_norms)):])
        return scaled[self.row_indices]
