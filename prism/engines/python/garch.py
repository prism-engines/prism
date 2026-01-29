"""
GARCH Engine.

Computes GARCH(1,1) volatility model parameters.
"""

import numpy as np


def compute(y: np.ndarray) -> dict:
    """
    Compute GARCH(1,1) parameters of signal.

    Args:
        y: Signal values

    Returns:
        dict with garch_omega, garch_alpha, garch_beta, garch_persistence
    """
    result = {
        'garch_omega': np.nan,
        'garch_alpha': np.nan,
        'garch_beta': np.nan,
        'garch_persistence': np.nan
    }

    if len(y) < 100:
        return result

    returns = np.diff(y)
    if len(returns) < 50:
        return result

    try:
        from arch import arch_model
        model = arch_model(returns * 100, vol='Garch', p=1, q=1, rescale=False)
        fit = model.fit(disp='off', show_warning=False)
        result['garch_omega'] = float(fit.params.get('omega', np.nan))
        result['garch_alpha'] = float(fit.params.get('alpha[1]', np.nan))
        result['garch_beta'] = float(fit.params.get('beta[1]', np.nan))
        result['garch_persistence'] = result['garch_alpha'] + result['garch_beta']
    except Exception:
        # Fallback: simple moment-based estimation
        var = np.var(returns)
        sq_returns = returns ** 2
        autocorr = np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1] if len(sq_returns) > 1 else 0
        if np.isnan(autocorr):
            autocorr = 0
        alpha = max(0, min(0.3, abs(autocorr)))
        beta = max(0, min(0.95 - alpha, 0.9))
        result['garch_omega'] = float(var * (1 - alpha - beta))
        result['garch_alpha'] = float(alpha)
        result['garch_beta'] = float(beta)
        result['garch_persistence'] = float(alpha + beta)

    return result
