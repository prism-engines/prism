"""
Hilbert Transform Amplitude Envelope
====================================

Uses the Hilbert transform to compute the instantaneous
amplitude envelope of a signal.

The analytic signal is:
    z(t) = x(t) + i·H[x(t)]

Where H[x] is the Hilbert transform.

Amplitude: A(t) = |z(t)|
Phase: φ(t) = arg(z(t))
"""

import numpy as np
from scipy.signal import hilbert
from typing import Dict


def compute(series: np.ndarray) -> Dict[str, float]:
    """
    Compute amplitude envelope via Hilbert transform.

    Args:
        series: 1D numpy array of observations

    Returns:
        dict with:
            - amplitude_mean: Mean amplitude
            - amplitude_std: Amplitude variability
            - amplitude_max: Maximum amplitude
            - amplitude_min: Minimum amplitude
            - phase_std: Phase variability
            - inst_freq_mean: Mean instantaneous frequency
    """
    if len(series) < 10:
        return {
            'amplitude_mean': 0.0,
            'amplitude_std': 0.0,
            'amplitude_max': 0.0,
            'amplitude_min': 0.0,
            'phase_std': 0.0,
            'inst_freq_mean': 0.0
        }

    # Detrend
    detrended = series - np.mean(series)

    # Hilbert transform
    analytic = hilbert(detrended)
    amplitude = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    # Instantaneous frequency (derivative of phase)
    inst_freq = np.diff(phase) / (2 * np.pi)

    return {
        'amplitude_mean': float(np.mean(amplitude)),
        'amplitude_std': float(np.std(amplitude)),
        'amplitude_max': float(np.max(amplitude)),
        'amplitude_min': float(np.min(amplitude)),
        'phase_std': float(np.std(phase)),
        'inst_freq_mean': float(np.mean(np.abs(inst_freq)))
    }
