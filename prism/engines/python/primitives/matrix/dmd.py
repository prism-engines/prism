"""
Dynamic Mode Decomposition Primitive (60)

DMD for analyzing spatiotemporal dynamics.
"""

import numpy as np
from typing import Tuple, Optional


def dynamic_mode_decomposition(
    signals: np.ndarray,
    dt: float = 1.0,
    rank: int = None,
    exact: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Dynamic Mode Decomposition.

    Parameters
    ----------
    signals : np.ndarray
        Data matrix (n_samples x n_signals) - time series in columns
    dt : float
        Time step between samples
    rank : int, optional
        Truncation rank (default: min(n_samples-1, n_signals))
    exact : bool
        If True, compute exact DMD modes

    Returns
    -------
    tuple
        (modes, eigenvalues, dynamics, amplitudes)
        modes: DMD modes (n_signals x rank) - spatial patterns
        eigenvalues: DMD eigenvalues (rank,) - complex growth/decay + frequency
        dynamics: Time dynamics (rank x n_samples) - temporal evolution
        amplitudes: Mode amplitudes (rank,) - initial mode contributions

    Notes
    -----
    DMD approximates: x_{k+1} ≈ A @ x_k
    where A is the best-fit linear operator.

    Eigenvalue interpretation:
    - |λ| > 1: growing mode
    - |λ| < 1: decaying mode
    - |λ| = 1: neutral mode
    - angle(λ): oscillation frequency (radians per time step)

    Continuous-time eigenvalue: ω = log(λ) / dt
    - real(ω): growth rate
    - imag(ω): frequency (rad/s)
    """
    signals = np.asarray(signals)

    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)

    n_samples, n_signals = signals.shape

    if n_samples < 3:
        return (
            np.full((n_signals, 1), np.nan),
            np.array([np.nan + 0j]),
            np.full((1, n_samples), np.nan),
            np.array([np.nan])
        )

    # Build time-shifted matrices
    # X = [x_0, x_1, ..., x_{n-2}]
    # X' = [x_1, x_2, ..., x_{n-1}]
    X = signals[:-1, :].T  # (n_signals x n_samples-1)
    Xprime = signals[1:, :].T  # (n_signals x n_samples-1)

    # SVD of X
    if rank is None:
        rank = min(n_samples - 1, n_signals)

    try:
        U, S, Vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return (
            np.full((n_signals, rank), np.nan),
            np.full(rank, np.nan + 0j),
            np.full((rank, n_samples), np.nan),
            np.full(rank, np.nan)
        )

    # Truncate to rank
    r = min(rank, len(S))
    Ur = U[:, :r]
    Sr = S[:r]
    Vr = Vh[:r, :]

    # Build reduced operator: Atilde = Ur.T @ Xprime @ Vr @ inv(diag(Sr))
    Sr_inv = 1.0 / (Sr + 1e-10)
    Atilde = Ur.T @ Xprime @ Vr.T @ np.diag(Sr_inv)

    # Eigendecomposition of Atilde
    eigenvalues, W = np.linalg.eig(Atilde)

    # DMD modes
    if exact:
        # Exact DMD: Phi = Xprime @ Vr @ diag(Sr_inv) @ W
        modes = Xprime @ Vr.T @ np.diag(Sr_inv) @ W
    else:
        # Projected DMD: Phi = Ur @ W
        modes = Ur @ W

    # Initial amplitudes (least squares fit to first snapshot)
    x0 = signals[0, :]
    try:
        amplitudes = np.linalg.lstsq(modes, x0, rcond=None)[0]
    except np.linalg.LinAlgError:
        amplitudes = np.zeros(r, dtype=complex)

    # Time dynamics
    time_steps = np.arange(n_samples)
    dynamics = np.zeros((r, n_samples), dtype=complex)
    for i, (lamb, amp) in enumerate(zip(eigenvalues, amplitudes)):
        dynamics[i, :] = amp * (lamb ** time_steps)

    return modes, eigenvalues, dynamics, amplitudes


def dmd_frequencies(
    eigenvalues: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Extract continuous-time frequencies from DMD eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        DMD eigenvalues (complex)
    dt : float
        Time step

    Returns
    -------
    np.ndarray
        Frequencies in Hz

    Notes
    -----
    ω = log(λ) / dt
    frequency = imag(ω) / (2π)
    """
    # Continuous-time eigenvalue
    omega = np.log(eigenvalues + 1e-10) / dt

    # Frequency in Hz
    frequencies = np.imag(omega) / (2 * np.pi)

    return np.abs(frequencies)


def dmd_growth_rates(
    eigenvalues: np.ndarray,
    dt: float = 1.0
) -> np.ndarray:
    """
    Extract growth rates from DMD eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        DMD eigenvalues (complex)
    dt : float
        Time step

    Returns
    -------
    np.ndarray
        Growth rates (positive = growing, negative = decaying)

    Notes
    -----
    ω = log(λ) / dt
    growth_rate = real(ω)
    """
    omega = np.log(eigenvalues + 1e-10) / dt
    return np.real(omega)
