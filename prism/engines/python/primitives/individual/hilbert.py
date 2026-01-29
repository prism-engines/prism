"""
Hilbert Transform Primitives (24-27)

Analytic signal, envelope, instantaneous frequency/amplitude.
"""

import numpy as np
from scipy import signal as scipy_signal


def hilbert_transform(signal: np.ndarray) -> np.ndarray:
    """
    Compute Hilbert transform (analytic signal).

    Parameters
    ----------
    signal : np.ndarray
        Input signal (real)

    Returns
    -------
    np.ndarray
        Complex analytic signal

    Notes
    -----
    z(t) = x(t) + i*H[x](t)
    where H is the Hilbert transform.
    """
    signal = np.asarray(signal)
    return scipy_signal.hilbert(signal)


def envelope(signal: np.ndarray) -> np.ndarray:
    """
    Compute signal envelope (amplitude modulation).

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Envelope (instantaneous amplitude)

    Notes
    -----
    A(t) = |z(t)| = sqrt(x² + H[x]²)
    Useful for detecting amplitude modulation.
    """
    analytic = hilbert_transform(signal)
    return np.abs(analytic)


def instantaneous_amplitude(signal: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous amplitude.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Instantaneous amplitude (same as envelope)
    """
    return envelope(signal)


def instantaneous_frequency(
    signal: np.ndarray,
    fs: float = 1.0
) -> np.ndarray:
    """
    Compute instantaneous frequency.

    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : float
        Sampling frequency

    Returns
    -------
    np.ndarray
        Instantaneous frequency

    Notes
    -----
    f(t) = (1/2π) * d(phase)/dt
    where phase = angle(z(t))
    """
    analytic = hilbert_transform(signal)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase, 1/fs) / (2 * np.pi)
    return inst_freq


def instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous phase.

    Parameters
    ----------
    signal : np.ndarray
        Input signal

    Returns
    -------
    np.ndarray
        Instantaneous phase (unwrapped)
    """
    analytic = hilbert_transform(signal)
    return np.unwrap(np.angle(analytic))
