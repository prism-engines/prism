"""
Spectral Engine.

Computes spectral properties via FFT.
"""

import numpy as np
from typing import Dict


def compute(y: np.ndarray, sample_rate: float = 1.0) -> Dict[str, float]:
    """
    Compute spectral properties of signal.

    Args:
        y: Signal values
        sample_rate: Sampling rate in Hz (default: 1.0)

    Returns:
        dict with spectral_slope, dominant_freq, spectral_entropy,
        spectral_centroid, spectral_bandwidth
    """
    result = {
        'spectral_slope': np.nan,
        'dominant_freq': np.nan,
        'spectral_entropy': np.nan,
        'spectral_centroid': np.nan,
        'spectral_bandwidth': np.nan
    }

    # Handle NaN values
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 64:
        return result

    # Check for constant signal
    if np.std(y) < 1e-10:
        result['spectral_entropy'] = 0.0
        result['spectral_centroid'] = 0.0
        result['spectral_bandwidth'] = 0.0
        result['spectral_slope'] = 0.0
        result['dominant_freq'] = 0.0
        return result

    try:
        fft = np.fft.rfft(y - np.mean(y))
        psd = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)

        # Check for zero power (constant signal after detrending)
        psd_sum = np.sum(psd)
        if psd_sum < 1e-10:
            return result

        # Dominant frequency (skip DC component)
        if len(psd) > 1:
            result['dominant_freq'] = float(freqs[np.argmax(psd[1:]) + 1])

        # Spectral entropy (normalized)
        psd_norm = psd / psd_sum
        # Avoid log(0) by filtering out zeros
        nonzero_mask = psd_norm > 1e-10
        if np.sum(nonzero_mask) > 0:
            entropy = -np.sum(psd_norm[nonzero_mask] * np.log(psd_norm[nonzero_mask]))
            max_entropy = np.log(len(psd))
            result['spectral_entropy'] = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        # Spectral centroid (center of mass of spectrum)
        centroid = np.sum(freqs * psd) / psd_sum
        result['spectral_centroid'] = float(centroid)

        # Spectral bandwidth (spread around centroid)
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / psd_sum)
        result['spectral_bandwidth'] = float(bandwidth)

        # Spectral slope (log-log fit)
        mask = freqs > 0
        if np.sum(mask) > 3:
            log_freqs = np.log10(freqs[mask])
            log_psd = np.log10(psd[mask] + 1e-10)
            # Only fit if we have variation
            if np.std(log_psd) > 1e-10:
                slope, _ = np.polyfit(log_freqs, log_psd, 1)
                result['spectral_slope'] = float(slope)

    except Exception:
        pass

    return result
