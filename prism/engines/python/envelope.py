"""
Envelope Engine.

Computes the Hilbert envelope of a signal.
"""

import numpy as np
from scipy.signal import hilbert
from scipy import stats


def compute(y: np.ndarray) -> dict:
    """
    Compute Hilbert envelope metrics of signal.

    Args:
        y: Signal values

    Returns:
        dict with envelope_rms, envelope_peak, envelope_kurtosis
    """
    if len(y) < 10:
        return {
            'envelope_rms': np.nan,
            'envelope_peak': np.nan,
            'envelope_kurtosis': np.nan
        }

    try:
        analytic = hilbert(y)
        envelope = np.abs(analytic)

        return {
            'envelope_rms': float(np.sqrt(np.mean(envelope ** 2))),
            'envelope_peak': float(np.max(envelope)),
            'envelope_kurtosis': float(stats.kurtosis(envelope, fisher=True))
        }
    except Exception:
        return {
            'envelope_rms': np.nan,
            'envelope_peak': np.nan,
            'envelope_kurtosis': np.nan
        }
