"""
Pairwise Spectral Primitives (40-43)

Coherence, cross-spectral density, phase spectrum.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional


def coherence(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: float = 1.0,
    nperseg: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude-squared coherence.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    fs : float
        Sampling frequency
    nperseg : int, optional
        Segment length

    Returns
    -------
    tuple
        (frequencies, coherence)

    Notes
    -----
    C_{xy}(f) = |P_{xy}(f)|² / (P_{xx}(f) * P_{yy}(f))
    Measures linear correlation as function of frequency.
    Values in [0, 1].
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    if nperseg is None:
        nperseg = min(256, n)

    freqs, Cxy = scipy_signal.coherence(sig_a, sig_b, fs=fs, nperseg=nperseg)
    return freqs, Cxy


def cross_spectral_density(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: float = 1.0,
    nperseg: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-spectral density.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    fs : float
        Sampling frequency
    nperseg : int, optional
        Segment length

    Returns
    -------
    tuple
        (frequencies, cross_spectral_density)

    Notes
    -----
    P_{xy}(f) = FFT(x) * conj(FFT(y))
    Complex-valued: magnitude shows coupling, phase shows lead/lag.
    """
    sig_a = np.asarray(sig_a).flatten()
    sig_b = np.asarray(sig_b).flatten()

    n = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:n], sig_b[:n]

    if nperseg is None:
        nperseg = min(256, n)

    freqs, Pxy = scipy_signal.csd(sig_a, sig_b, fs=fs, nperseg=nperseg)
    return freqs, Pxy


def phase_spectrum(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fs: float = 1.0,
    nperseg: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute phase spectrum (phase lag vs frequency).

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    fs : float
        Sampling frequency
    nperseg : int, optional
        Segment length

    Returns
    -------
    tuple
        (frequencies, phase_in_radians)

    Notes
    -----
    φ(f) = angle(P_{xy}(f))
    Positive: sig_a leads sig_b
    Negative: sig_b leads sig_a
    """
    freqs, Pxy = cross_spectral_density(sig_a, sig_b, fs, nperseg)
    phase = np.angle(Pxy)
    return freqs, phase


def wavelet_coherence(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    wavelet: str = 'morl',
    scales: np.ndarray = None,
    fs: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute wavelet coherence.

    Parameters
    ----------
    sig_a, sig_b : np.ndarray
        Input signals
    wavelet : str
        Wavelet name (e.g., 'morl', 'cmor1.5-1.0')
    scales : np.ndarray, optional
        Scales to analyze
    fs : float
        Sampling frequency

    Returns
    -------
    tuple
        (scales, time, coherence_matrix)

    Notes
    -----
    Time-frequency coherence analysis.
    Shows how coupling varies with frequency AND time.
    """
    try:
        import pywt

        sig_a = np.asarray(sig_a).flatten()
        sig_b = np.asarray(sig_b).flatten()

        n = min(len(sig_a), len(sig_b))
        sig_a, sig_b = sig_a[:n], sig_b[:n]

        if scales is None:
            scales = np.arange(1, min(128, n // 4))

        # Continuous wavelet transform
        coef_a, freqs = pywt.cwt(sig_a, scales, wavelet, sampling_period=1/fs)
        coef_b, _ = pywt.cwt(sig_b, scales, wavelet, sampling_period=1/fs)

        # Cross-wavelet spectrum
        W_ab = coef_a * np.conj(coef_b)

        # Individual power spectra
        S_a = np.abs(coef_a) ** 2
        S_b = np.abs(coef_b) ** 2

        # Smoothing (simple moving average)
        def smooth(arr, window=5):
            kernel = np.ones(window) / window
            return np.array([np.convolve(row, kernel, mode='same') for row in arr])

        S_a_smooth = smooth(S_a)
        S_b_smooth = smooth(S_b)
        W_ab_smooth = smooth(np.abs(W_ab))

        # Coherence
        coherence = W_ab_smooth ** 2 / (S_a_smooth * S_b_smooth + 1e-10)

        time = np.arange(n) / fs
        return scales, time, coherence

    except ImportError:
        # Fallback: simple windowed coherence
        sig_a = np.asarray(sig_a).flatten()
        sig_b = np.asarray(sig_b).flatten()

        n = min(len(sig_a), len(sig_b))
        window_size = min(64, n // 4)
        n_windows = n // window_size

        coherence_matrix = np.zeros((8, n_windows))  # 8 frequency bands

        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            freqs, coh = coherence(sig_a[start:end], sig_b[start:end], fs)

            # Average into bands
            band_size = len(coh) // 8
            for j in range(8):
                coherence_matrix[j, i] = np.mean(coh[j*band_size:(j+1)*band_size])

        scales = np.arange(1, 9)
        time = np.arange(n_windows) * window_size / fs
        return scales, time, coherence_matrix
