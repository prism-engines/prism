"""
Spectral Engine.

Computes spectral properties via FFT.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute spectral properties of signal.

    Args:
        y: Signal values

    Returns:
        dict with spectral_slope, dominant_freq, spectral_entropy,
        spectral_centroid, spectral_bandwidth
    """
    n = len(y)
    if n < 64:
        return {
            'spectral_slope': np.nan,
            'dominant_freq': np.nan,
            'spectral_entropy': np.nan,
            'spectral_centroid': np.nan,
            'spectral_bandwidth': np.nan
        }

    fft = np.fft.rfft(y - np.mean(y))
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n)

    result = {
        'spectral_slope': np.nan,
        'dominant_freq': np.nan,
        'spectral_entropy': np.nan,
        'spectral_centroid': np.nan,
        'spectral_bandwidth': np.nan
    }

    if len(psd) > 1:
        result['dominant_freq'] = float(freqs[np.argmax(psd[1:]) + 1])

    psd_norm = psd / (np.sum(psd) + 1e-10)
    result['spectral_entropy'] = float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)) / np.log(len(psd)))

    # Spectral centroid
    result['spectral_centroid'] = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))

    # Spectral bandwidth
    centroid = result['spectral_centroid']
    result['spectral_bandwidth'] = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / (np.sum(psd) + 1e-10)))

    # Spectral slope
    mask = freqs > 0
    if np.sum(mask) > 3:
        slope, _ = np.polyfit(np.log10(freqs[mask] + 1e-10), np.log10(psd[mask] + 1e-10), 1)
        result['spectral_slope'] = float(slope)

    return result
