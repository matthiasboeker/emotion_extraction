from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd


class NotReversible(Exception):
    """The sum of the elementary matrices are not equal to trajectory matrix"""


def get_trajectory_matrix(
    time_series: np.ndarray, lag: int, window_size: int
) -> np.ndarray:
    M = lag
    K = window_size - M + 1
    if isinstance(time_series, pd.Series):
        time_series = time_series.to_numpy()
    traj_matrix = np.column_stack([time_series[i: i + K] for i in range(0, M)])
    return traj_matrix


def get_lag_cov_matrix(trajectory_matrix: np.array):
    return (
        np.matmul(trajectory_matrix, trajectory_matrix.T) / trajectory_matrix.shape[1]
    )


def diagonal_averaging(matrix: np.ndarray) -> np.ndarray:
    """Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series."""
    reversed_matrix = matrix[::-1]
    return np.array(
        [
            reversed_matrix.diagonal(i).mean()
            for i in range(-matrix.shape[0] + 1, matrix.shape[1])
        ]
    )


def reconstruct(matrices: List[np.ndarray]) -> Union[np.ndarray, int]:
    return sum(diagonal_averaging(matrix) for matrix in matrices)


def apply_svd(lag_cov_matrix: np.ndarray) -> Dict[str, np.ndarray]:
    u, s, v = np.linalg.svd(lag_cov_matrix)
    return {"U": u, "Sigma": s, "V": v}


def calculate_elementary_matrices(
    svd: Dict[str, np.ndarray], lag_cov_matrix: np.ndarray
) -> np.ndarray:
    v_transpose = svd["V"].T
    rank = np.linalg.matrix_rank(lag_cov_matrix)
    elementary_matrices = np.array(
        [
            svd["Sigma"][i] * np.outer(svd["U"][:, i], v_transpose[:, i])
            for i in range(0, rank)
        ]
    )
    if not np.allclose(lag_cov_matrix, elementary_matrices.sum(axis=0), atol=1e-10):
        raise NotReversible
    return elementary_matrices


def sum_elementary_matrices(svd, lag_cov_matrix, q):
    elementary_matrices = calculate_elementary_matrices(svd, lag_cov_matrix)
    if not np.allclose(lag_cov_matrix, elementary_matrices.sum(axis=0), atol=1e-10):
        raise NotReversible
    return sum(elementary_matrices[i, :, :] for i in range(0, q))


class SSA:
    def __init__(self, factorised_matrix):
        self.factorised_matrix = factorised_matrix

    @classmethod
    def fit(cls, trajectory_matrix: np.ndarray, q):
        lag_cov_matrix = get_lag_cov_matrix(trajectory_matrix)
        svd_results = apply_svd(lag_cov_matrix)
        return cls(sum_elementary_matrices(svd_results, lag_cov_matrix, q))

    @classmethod
    def transform_fit(cls, time_series: np.ndarray, lag: int, window_size: int, q):
        trajectory_matrix = get_trajectory_matrix(time_series, lag, window_size)
        return cls.fit(trajectory_matrix, q)
