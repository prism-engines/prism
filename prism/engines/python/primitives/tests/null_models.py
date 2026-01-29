"""
Null Model Test Primitives (106-107)

Surrogate data tests and random matrix null models.
"""

import numpy as np
from typing import Tuple, Callable


def surrogate_test(
    signal: np.ndarray,
    metric_func: Callable,
    n_surrogates: int = 100,
    surrogate_type: str = 'phase_randomized'
) -> Tuple[float, float, float]:
    """
    Surrogate data test for nonlinearity/determinism.

    Parameters
    ----------
    signal : np.ndarray
        Original time series
    metric_func : callable
        Function that computes metric from signal
    n_surrogates : int
        Number of surrogate realizations
    surrogate_type : str
        'phase_randomized': Preserves spectrum, destroys nonlinearity
        'shuffle': Destroys all temporal structure
        'block_shuffle': Preserves short-term structure

    Returns
    -------
    observed : float
        Metric value for original signal
    p_value : float
        p-value (fraction of surrogates with more extreme value)
    z_score : float
        Z-score of observed relative to surrogate distribution

    Notes
    -----
    Surrogate data test (Theiler et al., 1992):

    1. Compute metric for original signal
    2. Generate surrogate signals that preserve SOME properties
    3. Compute metric for each surrogate
    4. If original is significantly different from surrogates,
       the metric captures something the surrogates destroyed

    Phase-randomized surrogates:
    - Preserve: Power spectrum (frequency content)
    - Destroy: Phase relationships (nonlinear structure)

    If Lyapunov is significant vs phase-randomized surrogates,
    the chaos is REAL, not just a spectral artifact.

    Physical interpretation:
    "Is this metric capturing real structure, or could it arise from
    a linear stochastic process with the same spectrum?"
    """
    signal = np.asarray(signal).flatten()

    # Observed value
    observed = metric_func(signal)

    # Generate surrogates and compute metric
    surrogate_values = np.zeros(n_surrogates)

    for i in range(n_surrogates):
        if surrogate_type == 'phase_randomized':
            surr = _phase_randomize(signal)
        elif surrogate_type == 'shuffle':
            surr = np.random.permutation(signal)
        elif surrogate_type == 'block_shuffle':
            surr = _block_shuffle(signal, block_size=max(1, len(signal) // 10))
        else:
            raise ValueError(f"Unknown surrogate type: {surrogate_type}")

        try:
            surrogate_values[i] = metric_func(surr)
        except:
            surrogate_values[i] = np.nan

    # Remove NaN surrogates
    surrogate_values = surrogate_values[~np.isnan(surrogate_values)]

    if len(surrogate_values) == 0:
        return float(observed), np.nan, np.nan

    # Compute p-value and z-score
    p_value = np.mean(np.abs(surrogate_values) >= np.abs(observed))

    surr_mean = np.mean(surrogate_values)
    surr_std = np.std(surrogate_values)

    if surr_std > 0:
        z = (observed - surr_mean) / surr_std
    else:
        z = np.inf if observed != surr_mean else 0

    return float(observed), float(p_value), float(z)


def _phase_randomize(signal: np.ndarray) -> np.ndarray:
    """Generate phase-randomized surrogate preserving power spectrum."""
    n = len(signal)

    # FFT
    fft_vals = np.fft.fft(signal)

    # Randomize phases
    phases = np.random.uniform(0, 2 * np.pi, n // 2 + 1)

    # Construct new FFT (preserve Hermitian symmetry for real output)
    new_fft = np.zeros(n, dtype=complex)
    new_fft[0] = fft_vals[0]  # DC component (no phase)

    for i in range(1, n // 2):
        magnitude = np.abs(fft_vals[i])
        new_fft[i] = magnitude * np.exp(1j * phases[i])
        new_fft[n - i] = magnitude * np.exp(-1j * phases[i])

    if n % 2 == 0:
        new_fft[n // 2] = np.abs(fft_vals[n // 2])  # Nyquist (real)

    # Inverse FFT
    surrogate = np.real(np.fft.ifft(new_fft))

    return surrogate


def _block_shuffle(signal: np.ndarray, block_size: int) -> np.ndarray:
    """Shuffle signal in blocks (preserves short-term structure)."""
    n = len(signal)
    n_blocks = n // block_size

    if n_blocks == 0:
        return np.random.permutation(signal)

    blocks = [signal[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    remainder = signal[n_blocks*block_size:]

    np.random.shuffle(blocks)

    if len(remainder) > 0:
        return np.concatenate(blocks + [remainder])
    return np.concatenate(blocks)


def marchenko_pastur_test(
    eigenvalues: np.ndarray,
    n_samples: int,
    n_features: int
) -> Tuple[np.ndarray, float]:
    """
    Test eigenvalues against Marchenko-Pastur distribution (random matrix null).

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues from covariance matrix (sorted descending)
    n_samples : int
        Number of samples used to compute covariance
    n_features : int
        Number of features (signals)

    Returns
    -------
    significant_mask : np.ndarray
        Boolean mask: True for eigenvalues above MP upper bound
    mp_upper_bound : float
        Upper edge of Marchenko-Pastur distribution

    Notes
    -----
    Marchenko-Pastur (MP) distribution describes eigenvalues of
    random covariance matrices.

    For n_samples observations of n_features random variables:
    - λ_max = σ² × (1 + √(n_features/n_samples))²
    - λ_min = σ² × (1 - √(n_features/n_samples))²

    Eigenvalues ABOVE λ_max are "signal" (significant structure).
    Eigenvalues WITHIN [λ_min, λ_max] could be noise.

    Physical interpretation:
    "Which modes are real structure vs random fluctuation?"

    This is ESSENTIAL for:
    - Determining how many significant modes
    - Avoiding overfitting to noise eigenvalues
    - Robust estimation of effective dimension
    """
    eigenvalues = np.asarray(eigenvalues).flatten()

    # Ratio of features to samples
    q = n_features / n_samples

    if q > 1:
        # More features than samples - swap
        q = 1 / q

    # Estimate noise variance (from smallest eigenvalues)
    # Or assume unit variance for normalized data
    sigma2 = 1.0  # Assume normalized data

    # Marchenko-Pastur bounds
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(q)) ** 2

    # Eigenvalues above upper bound are significant
    significant_mask = eigenvalues > lambda_plus

    return significant_mask, float(lambda_plus)


def significance_summary(
    value: float,
    baseline_mean: float,
    baseline_std: float,
    sample_size: int = None
) -> dict:
    """
    Comprehensive significance assessment for a single value.

    Parameters
    ----------
    value : float
        Observed value
    baseline_mean : float
        Expected/baseline mean
    baseline_std : float
        Baseline standard deviation
    sample_size : int, optional
        Sample size for effect size calculation

    Returns
    -------
    dict
        Dictionary with z-score, p-value, effect size, and interpretation

    Notes
    -----
    Provides actionable significance levels:
    - not significant (p > 0.10): ignore
    - marginal (0.05 < p < 0.10): monitor
    - significant (0.01 < p < 0.05): investigate
    - highly significant (p < 0.01): act
    """
    from scipy import stats

    if baseline_std <= 0:
        return {
            'value': value,
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'deviation': np.nan,
            'z_score': np.nan,
            'p_value': np.nan,
            'effect_size': np.nan,
            'significance': 'undefined',
            'action': 'investigate',
        }

    z = (value - baseline_mean) / baseline_std
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    # Effect size (Cohen's d approximation)
    effect_size = abs(z)

    # Interpretation
    if p > 0.10:
        significance = 'not significant'
        action = 'ignore'
    elif p > 0.05:
        significance = 'marginal'
        action = 'monitor'
    elif p > 0.01:
        significance = 'significant'
        action = 'investigate'
    else:
        significance = 'highly significant'
        action = 'act'

    return {
        'value': value,
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'deviation': value - baseline_mean,
        'z_score': float(z),
        'p_value': float(p),
        'effect_size': float(effect_size),
        'significance': significance,
        'action': action,
    }
