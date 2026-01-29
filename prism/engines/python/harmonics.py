"""
Harmonics Engine.

Computes harmonic analysis (fundamental + harmonics + THD).
Replaces motor_signature, gear_mesh, rotor_dynamics outputs.
"""

import numpy as np


def compute(y: np.ndarray, sample_rate: float = 1.0, fundamental: float = None) -> dict:
    """
    Compute harmonic analysis of signal.

    Args:
        y: Signal values
        sample_rate: Sampling rate in Hz
        fundamental: Known fundamental frequency. If None, auto-detected.

    Returns:
        dict with fundamental_freq, fundamental_amplitude, harmonic_2x, harmonic_3x, thd
    """
    result = {
        'fundamental_freq': np.nan,
        'fundamental_amplitude': np.nan,
        'harmonic_2x': np.nan,
        'harmonic_3x': np.nan,
        'thd': np.nan
    }

    if len(y) < 64:
        return result

    try:
        # Compute FFT
        n = len(y)
        fft = np.fft.rfft(y - np.mean(y))
        freqs = np.fft.rfftfreq(n, d=1/sample_rate)
        mag = np.abs(fft)

        # Find fundamental
        if fundamental is None:
            # Auto-detect: largest peak (skip DC)
            fund_idx = np.argmax(mag[1:]) + 1
            fundamental = freqs[fund_idx]
        else:
            # Find closest frequency bin
            fund_idx = np.argmin(np.abs(freqs - fundamental))

        f1_amp = mag[fund_idx]

        # Get harmonic amplitudes
        def amplitude_at_freq(target_freq):
            idx = np.argmin(np.abs(freqs - target_freq))
            return mag[idx] if idx < len(mag) else 0.0

        f2_amp = amplitude_at_freq(fundamental * 2)
        f3_amp = amplitude_at_freq(fundamental * 3)

        # THD: sqrt(sum of harmonics^2) / fundamental
        harmonics_sq = f2_amp**2 + f3_amp**2
        # Add more harmonics for better THD estimate
        for h in range(4, 11):
            harmonics_sq += amplitude_at_freq(fundamental * h) ** 2

        thd = np.sqrt(harmonics_sq) / (f1_amp + 1e-10) * 100  # As percentage

        result = {
            'fundamental_freq': float(fundamental),
            'fundamental_amplitude': float(f1_amp),
            'harmonic_2x': float(f2_amp),
            'harmonic_3x': float(f3_amp),
            'thd': float(thd)
        }

    except Exception:
        pass

    return result
