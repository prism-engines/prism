"""
Rolling Envelope Engine.

Envelope RMS over sliding window.
"""

import numpy as np
from scipy.signal import hilbert


def compute(y: np.ndarray, window: int = 100) -> dict:
    """
    Compute rolling envelope RMS.

    Args:
        y: Signal values
        window: Window size

    Returns:
        dict with 'rolling_envelope_rms' array
    """
    n = len(y)
    result = np.full(n, np.nan)

    if n < window:
        return {'rolling_envelope_rms': result}

    try:
        # Compute full envelope first (more efficient)
        analytic = hilbert(y)
        envelope = np.abs(analytic)

        # Rolling RMS of envelope
        for i in range(window, n):
            chunk = envelope[i-window:i]
            result[i] = np.sqrt(np.mean(chunk ** 2))

    except Exception:
        pass

    return {'rolling_envelope_rms': result}
