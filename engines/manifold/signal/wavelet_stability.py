"""
Wavelet Stability Engine.

Uses continuous wavelet transform to decompose signal into
time-frequency representation. Extracts energy distribution
across frequency bands and how it changes over time.

Key insight: Eigendecomp captures INTER-signal structure.
Wavelet captures INTRA-signal structure (frequency content).
They're orthogonal views — one can't replace the other.
"""

import numpy as np
import pywt


def compute(y: np.ndarray, fs: float = 1.0, n_scales: int = 16) -> dict:
    """
    Compute wavelet-derived stability metrics.

    Args:
        y: Signal values
        fs: Sampling frequency
        n_scales: Number of wavelet scales to analyze

    Returns:
        dict with 10 wavelet stability metrics
    """
    result = {
        'wavelet_energy_low': np.nan,      # Energy in lowest 1/4 of scales
        'wavelet_energy_mid': np.nan,      # Energy in middle 1/2 of scales
        'wavelet_energy_high': np.nan,     # Energy in highest 1/4 of scales
        'wavelet_energy_ratio': np.nan,    # low/high energy ratio
        'wavelet_entropy': np.nan,         # Shannon entropy of scale energies
        'wavelet_concentration': np.nan,   # Max scale energy / total
        'wavelet_dominant_scale': np.nan,  # Which scale carries most energy
        'wavelet_energy_drift': np.nan,    # Is energy migrating across scales?
        'wavelet_temporal_std': np.nan,    # How much does energy vary over time?
        'wavelet_intermittency': np.nan,   # Kurtosis of wavelet coefficients (bursts)
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]
    n = len(y)

    if n < 8:
        return result

    if np.std(y) < 1e-10:
        result['wavelet_entropy'] = 0.0
        result['wavelet_concentration'] = 1.0
        return result

    try:
        # Scales logarithmically spaced to cover frequency range
        max_scale = max(8, n // 4)
        actual_n_scales = min(n_scales, n // 2)
        if actual_n_scales < 2:
            return result

        scales = np.geomspace(2, max_scale, num=actual_n_scales)

        # CWT with complex Morlet wavelet (pywt)
        coefficients, _ = pywt.cwt(y, scales, 'cmor1.5-1.0')

        # Power = |coefficients|^2
        power = np.abs(coefficients) ** 2

        # Energy per scale (sum over time)
        scale_energy = np.sum(power, axis=1)
        total_energy = np.sum(scale_energy)

        if total_energy < 1e-10:
            return result

        # Normalize
        scale_energy_norm = scale_energy / total_energy

        # Band energies
        n_scales_actual = len(scales)
        q1 = max(1, n_scales_actual // 4)
        q3 = min(n_scales_actual - 1, 3 * n_scales_actual // 4)

        result['wavelet_energy_low'] = float(np.sum(scale_energy_norm[:q1]))
        result['wavelet_energy_mid'] = float(np.sum(scale_energy_norm[q1:q3]))
        result['wavelet_energy_high'] = float(np.sum(scale_energy_norm[q3:]))

        high_e = result['wavelet_energy_high']
        if high_e > 1e-10:
            result['wavelet_energy_ratio'] = float(result['wavelet_energy_low'] / high_e)

        # Wavelet entropy (like eigenvalue entropy but for frequency scales)
        p = scale_energy_norm[scale_energy_norm > 1e-10]
        result['wavelet_entropy'] = float(-np.sum(p * np.log(p)))

        # Concentration (like energy_concentration for eigenvalues)
        result['wavelet_concentration'] = float(np.max(scale_energy_norm))

        # Dominant scale
        result['wavelet_dominant_scale'] = float(scales[np.argmax(scale_energy)])

        # Energy drift: split time axis in half, compare scale distributions
        mid_t = power.shape[1] // 2
        if mid_t > 0:
            first_half = np.sum(power[:, :mid_t], axis=1)
            second_half = np.sum(power[:, mid_t:], axis=1)
            first_total = np.sum(first_half)
            second_total = np.sum(second_half)
            if first_total > 1e-10 and second_total > 1e-10:
                first_norm = first_half / first_total
                second_norm = second_half / second_total
                drift = np.sum(np.abs(second_norm - first_norm))
                result['wavelet_energy_drift'] = float(drift)

        # Temporal variability: how much does total power fluctuate over time?
        temporal_power = np.sum(power, axis=0)  # Sum across scales per time
        temporal_mean = np.mean(temporal_power)
        if temporal_mean > 1e-10:
            result['wavelet_temporal_std'] = float(np.std(temporal_power) / temporal_mean)

        # Intermittency: kurtosis of wavelet coefficients
        # High kurtosis = bursty energy (transients, impacts)
        flat_coeffs = np.abs(coefficients).flatten()
        if len(flat_coeffs) > 10:
            m = np.mean(flat_coeffs)
            s = np.std(flat_coeffs)
            if s > 1e-10:
                result['wavelet_intermittency'] = float(
                    np.mean(((flat_coeffs - m) / s) ** 4) - 3.0
                )

    except Exception:
        pass

    return result
