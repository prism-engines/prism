"""
Rolling Envelope Engine.

Envelope RMS over sliding window.
All parameters from manifest via params dict.
"""

import numpy as np
from typing import Dict, Any
from scipy.signal import hilbert


def compute(y: np.ndarray, params: Dict[str, Any] = None) -> dict:
    """
    Compute rolling envelope RMS.

    Args:
        y: Signal values
        params: Parameters from manifest:
            - window: Window size
            - stride: Step size between windows

    Returns:
        dict with 'rolling_envelope_rms' array
    """
    params = params or {}
    window = params.get('window', 100)
    # Cheap engine (O(n)): stride=1 is acceptable default
    stride = params.get('stride', 1)

    n = len(y)
    result = np.full(n, np.nan)

    if n < window:
        return {'rolling_envelope_rms': result}

    try:
        # Compute full envelope first (more efficient)
        analytic = hilbert(y)
        envelope = np.abs(analytic)

        # Rolling RMS of envelope
        for i in range(0, n - window + 1, stride):
            chunk = envelope[i:i + window]
            result[i + window - 1] = np.sqrt(np.mean(chunk ** 2))

    except Exception:
        pass

    return {'rolling_envelope_rms': result}
