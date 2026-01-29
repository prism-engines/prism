"""
DMD (Dynamic Mode Decomposition) Engine.

Computes dynamic mode decomposition for linear dynamics analysis.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute DMD of signal.

    Args:
        y: Signal values

    Returns:
        dict with dmd_dominant_freq, dmd_growth_rate, dmd_is_stable, dmd_n_modes
    """
    n = len(y)
    if n < 50:
        return {
            'dmd_dominant_freq': np.nan,
            'dmd_growth_rate': np.nan,
            'dmd_is_stable': True,
            'dmd_n_modes': 0
        }

    n_delays = min(10, n // 5)
    X = np.zeros((n_delays, n - n_delays))
    for i in range(n_delays):
        X[i, :] = y[i:n - n_delays + i]

    X1, X2 = X[:, :-1], X[:, 1:]

    try:
        U, S, Vh = np.linalg.svd(X1, full_matrices=False)
        r = min(len(S), 5)
        U, S, Vh = U[:, :r], S[:r], Vh[:r, :]

        A_tilde = U.T @ X2 @ Vh.T @ np.diag(1/S)
        eigenvalues, _ = np.linalg.eig(A_tilde)

        freqs = np.angle(eigenvalues) / (2 * np.pi)
        growth_rates = np.log(np.abs(eigenvalues) + 1e-10)
        dominant_idx = np.argmax(np.abs(eigenvalues))

        return {
            'dmd_dominant_freq': float(np.abs(freqs[dominant_idx])),
            'dmd_growth_rate': float(growth_rates[dominant_idx]),
            'dmd_is_stable': bool(np.all(np.abs(eigenvalues) <= 1.01)),
            'dmd_n_modes': int(r)
        }
    except Exception:
        return {
            'dmd_dominant_freq': np.nan,
            'dmd_growth_rate': np.nan,
            'dmd_is_stable': True,
            'dmd_n_modes': 0
        }
