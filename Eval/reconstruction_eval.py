"""Feature-wise reconstruction metrics."""

import numpy as np


def compute_featurewise_pearson(input_matrix, reconstructed_matrix):
    """Median Pearson correlation across features over cells/spots."""
    x = np.asarray(input_matrix, dtype=float)
    y = np.asarray(reconstructed_matrix, dtype=float)
    values = [np.corrcoef(x[:, j], y[:, j])[0, 1] for j in range(x.shape[1])
              if np.isfinite(x[:, j]).all() and np.isfinite(y[:, j]).all()
              and np.std(x[:, j]) > 0 and np.std(y[:, j]) > 0]
    return float(np.median(values)) if values else np.nan


def compute_featurewise_nrmse(input_matrix, reconstructed_matrix):
    """Median feature-wise RMSE normalized by the input feature SD."""
    x = np.asarray(input_matrix, dtype=float)
    y = np.asarray(reconstructed_matrix, dtype=float)
    values = [np.sqrt(np.mean((x[:, j] - y[:, j]) ** 2)) / np.std(x[:, j])
              for j in range(x.shape[1])
              if np.isfinite(x[:, j]).all() and np.isfinite(y[:, j]).all()
              and np.std(x[:, j]) > 1e-12]
    return float(np.median(values)) if values else np.nan
