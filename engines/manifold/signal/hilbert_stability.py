"""
Hilbert Stability Engine.

Imports from primitives/individual/hilbert.py (canonical).
Computes instantaneous frequency stability — a replacement for FTLE
that works at full temporal resolution with zero minimum window.

Key insight: FTLE measures trajectory divergence (needs embedding).
Hilbert measures frequency stability (needs only the signal).
Both answer "how stable is this signal?" but Hilbert works everywhere.
"""

import numpy as np
from engines.primitives.individual.hilbert import (
    instantaneous_frequency,
    instantaneous_amplitude,
    instantaneous_phase,
)
from engines.primitives.individual.statistics import kurtosis, skewness


def compute(y: np.ndarray, fs: float = 1.0) -> dict:
    """
    Compute Hilbert-derived stability metrics.

    Args:
        y: Signal values
        fs: Sampling frequency (default 1.0)

    Returns:
        dict with 11 stability metrics
    """
    result = {
        'inst_freq_mean': np.nan,
        'inst_freq_std': np.nan,
        'inst_freq_stability': np.nan,     # 1/std — higher = more stable
        'inst_freq_kurtosis': np.nan,      # Heavy tails = intermittent instability
        'inst_freq_skewness': np.nan,      # Asymmetric frequency drift
        'inst_freq_range': np.nan,         # Max - min frequency
        'inst_freq_drift': np.nan,         # Slope of inst_freq over time
        'inst_amp_cv': np.nan,             # Coefficient of variation of amplitude
        'inst_amp_trend': np.nan,          # Is amplitude growing or decaying?
        'phase_coherence': np.nan,         # How linear is the phase? (1=perfect tone)
        'am_fm_ratio': np.nan,             # Amplitude modulation vs frequency modulation
    }

    y = np.asarray(y).flatten()
    y = y[~np.isnan(y)]

    if len(y) < 4:
        return result

    # Constant signal
    if np.std(y) < 1e-10:
        result['inst_freq_std'] = 0.0
        result['inst_freq_stability'] = float('inf')
        result['phase_coherence'] = 1.0
        return result

    try:
        # Instantaneous frequency
        inst_freq = instantaneous_frequency(y, fs=fs)

        # Remove edge artifacts (first and last 2 samples)
        if len(inst_freq) > 8:
            inst_freq_clean = inst_freq[2:-2]
        else:
            inst_freq_clean = inst_freq

        # Remove extreme outliers (> 5 MAD from median)
        median_freq = np.median(inst_freq_clean)
        mad = np.median(np.abs(inst_freq_clean - median_freq))
        if mad > 0:
            mask = np.abs(inst_freq_clean - median_freq) < 5 * mad * 1.4826
            inst_freq_clean = inst_freq_clean[mask]

        if len(inst_freq_clean) < 3:
            return result

        # Frequency statistics
        result['inst_freq_mean'] = float(np.mean(inst_freq_clean))
        freq_std = float(np.std(inst_freq_clean))
        result['inst_freq_std'] = freq_std
        result['inst_freq_stability'] = float(1.0 / (freq_std + 1e-10))
        result['inst_freq_kurtosis'] = float(kurtosis(inst_freq_clean, fisher=True))
        result['inst_freq_skewness'] = float(skewness(inst_freq_clean))
        result['inst_freq_range'] = float(np.max(inst_freq_clean) - np.min(inst_freq_clean))

        # Frequency drift (linear trend)
        t = np.arange(len(inst_freq_clean))
        if len(t) > 2:
            slope = np.polyfit(t, inst_freq_clean, 1)[0]
            result['inst_freq_drift'] = float(slope)

        # Amplitude metrics
        inst_amp = instantaneous_amplitude(y)
        amp_mean = np.mean(inst_amp)
        if amp_mean > 1e-10:
            result['inst_amp_cv'] = float(np.std(inst_amp) / amp_mean)

        # Amplitude trend
        if len(inst_amp) > 4:
            t_amp = np.arange(len(inst_amp))
            amp_slope = np.polyfit(t_amp, inst_amp, 1)[0]
            result['inst_amp_trend'] = float(amp_slope)

        # Phase coherence: how linear is the phase?
        # Perfect sinusoid = perfectly linear phase = coherence 1.0
        phase = instantaneous_phase(y)
        if len(phase) > 4:
            t_phase = np.arange(len(phase))
            coeffs = np.polyfit(t_phase, phase, 1)
            residual = phase - coeffs[0] * t_phase - coeffs[1]
            phase_var = np.var(residual)
            total_var = np.var(phase)
            if total_var > 1e-10:
                result['phase_coherence'] = float(max(0.0, 1.0 - phase_var / total_var))

        # AM/FM ratio: relative strength of amplitude vs frequency modulation
        amp_variation = np.std(inst_amp) / (np.mean(inst_amp) + 1e-10)
        freq_variation = freq_std / (abs(result['inst_freq_mean']) + 1e-10)
        if freq_variation > 1e-10:
            result['am_fm_ratio'] = float(amp_variation / freq_variation)

    except Exception:
        pass

    return result
