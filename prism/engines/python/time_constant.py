"""
Time Constant Engine.

Fits exponential decay/rise to estimate thermal time constant.
y(t) = y_final + (y_initial - y_final) * exp(-t/tau)
"""

import numpy as np
from scipy.optimize import curve_fit


def compute(y: np.ndarray, I: np.ndarray = None) -> dict:
    """
    Estimate time constant from exponential fit.

    Args:
        y: Signal values (temperature, etc.)
        I: Time/index values

    Returns:
        dict with time_constant, equilibrium_value, fit_r2
    """
    result = {
        'time_constant': np.nan,
        'equilibrium_value': np.nan,
        'fit_r2': np.nan
    }

    if len(y) < 10:
        return result

    if I is None:
        I = np.arange(len(y), dtype=float)

    try:
        # Normalize time to start at 0
        t = I - I[0]

        # Initial guesses
        y0 = y[0]
        y_final = y[-1]
        tau_guess = (t[-1] - t[0]) / 3

        def exp_func(t, y_inf, y_0, tau):
            return y_inf + (y_0 - y_inf) * np.exp(-t / tau)

        popt, _ = curve_fit(
            exp_func, t, y,
            p0=[y_final, y0, tau_guess],
            bounds=([y.min() - abs(y.max()), y.min() - abs(y.max()), 1e-6],
                    [y.max() + abs(y.max()), y.max() + abs(y.max()), t[-1] * 10]),
            maxfev=1000
        )

        y_inf, y_0, tau = popt

        # Compute R^2
        y_pred = exp_func(t, y_inf, y_0, tau)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        result = {
            'time_constant': float(tau),
            'equilibrium_value': float(y_inf),
            'fit_r2': float(r2)
        }

    except Exception:
        pass

    return result
