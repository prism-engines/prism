"""
Frequency Bands Engine.

Computes energy in specified frequency bands.
Replaces hardcoded BPFO/BPFI - bands are parameters.
"""

import numpy as np
from scipy.signal import welch
from typing import Dict


def compute(y: np.ndarray, sample_rate: float = 1.0, bands: dict = None) -> Dict[str, float]:
    """
    Compute energy in frequency bands.

    Args:
        y: Signal values
        sample_rate: Sampling rate in Hz
        bands: Dict mapping band names to [low, high] frequency ranges
               e.g., {'low': [0, 100], 'mid': [100, 500], 'high': [500, 2000]}
               If None, uses default bands based on Nyquist

    Returns:
        dict with 'band_{name}' keys for each band
    """
    result = {}

    # Handle NaN values
    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 64:
        if bands:
            for name in bands:
                result[f'band_{name}'] = np.nan
                result[f'band_{name}_rel'] = np.nan
        result['total_power'] = np.nan
        return result

    try:
        nyquist = sample_rate / 2

        # Compute PSD
        nperseg = min(256, n)
        freqs, psd = welch(y, fs=sample_rate, nperseg=nperseg)

        # Default bands if not specified
        if bands is None:
            bands = {
                'low': [0, nyquist * 0.1],
                'mid': [nyquist * 0.1, nyquist * 0.5],
                'high': [nyquist * 0.5, nyquist]
            }

        total_power = np.sum(psd)

        for name, (low, high) in bands.items():
            # Nyquist check: skip bands entirely above Nyquist
            if low >= nyquist:
                result[f'band_{name}'] = 0.0
                result[f'band_{name}_rel'] = 0.0
                continue

            # Clamp high frequency to Nyquist
            high = min(high, nyquist)

            # Skip invalid bands
            if low >= high:
                result[f'band_{name}'] = 0.0
                result[f'band_{name}_rel'] = 0.0
                continue

            mask = (freqs >= low) & (freqs <= high)
            band_power = float(np.sum(psd[mask]))
            result[f'band_{name}'] = band_power
            result[f'band_{name}_rel'] = float(band_power / (total_power + 1e-10))

        result['total_power'] = float(total_power)
        result['nyquist'] = float(nyquist)

    except Exception:
        if bands:
            for name in bands:
                result[f'band_{name}'] = np.nan
                result[f'band_{name}_rel'] = np.nan
        result['total_power'] = np.nan

    return result
