"""
Cointegration Engine.

Computes Engle-Granger cointegration test between signal pairs.

Two series are cointegrated if they share a common stochastic trend -
their linear combination is stationary even if each series is not.
"""

import numpy as np
from typing import Dict


def compute(y_a: np.ndarray, y_b: np.ndarray) -> Dict[str, float]:
    """
    Compute cointegration between two signals.

    Uses Engle-Granger two-step method:
    1. Regress y_a on y_b
    2. Test residuals for stationarity (ADF test)

    Args:
        y_a: First signal
        y_b: Second signal

    Returns:
        dict with:
            - 'coint_stat': ADF test statistic on residuals
            - 'coint_pvalue': P-value
            - 'is_cointegrated': True if p < 0.05
            - 'coint_beta': Cointegrating coefficient
    """
    result = {
        'coint_stat': np.nan,
        'coint_pvalue': np.nan,
        'is_cointegrated': False,
        'coint_beta': np.nan,
    }

    n = min(len(y_a), len(y_b))
    if n < 50:
        return result

    y_a = np.asarray(y_a[:n]).flatten()
    y_b = np.asarray(y_b[:n]).flatten()

    # Remove NaN
    mask = ~(np.isnan(y_a) | np.isnan(y_b))
    y_a, y_b = y_a[mask], y_b[mask]
    n = len(y_a)

    if n < 50:
        return result

    try:
        # Try statsmodels first (most reliable)
        from statsmodels.tsa.stattools import coint
        stat, pvalue, crit = coint(y_a, y_b)

        # Get beta from regression
        X = np.column_stack([np.ones(n), y_b])
        beta = np.linalg.lstsq(X, y_a, rcond=None)[0]

        result['coint_stat'] = float(stat)
        result['coint_pvalue'] = float(pvalue)
        result['is_cointegrated'] = pvalue < 0.05
        result['coint_beta'] = float(beta[1])

        return result

    except ImportError:
        pass

    # Fallback: manual Engle-Granger
    try:
        # Step 1: Regression
        X = np.column_stack([np.ones(n), y_b])
        beta = np.linalg.lstsq(X, y_a, rcond=None)[0]
        residuals = y_a - X @ beta

        # Step 2: ADF test on residuals
        # Determine lag order (Schwert rule)
        n_lags = int(np.floor(4 * (n / 100) ** 0.25))
        n_lags = max(1, min(n_lags, n // 10))

        # Build ADF regression: Δε_t = γ*ε_{t-1} + Σ δ_i*Δε_{t-i} + u_t
        diff_resid = np.diff(residuals)

        # Skip if not enough data
        if len(diff_resid) <= n_lags + 5:
            return result

        # Construct regressors
        y_adf = diff_resid[n_lags:]
        X_list = [residuals[n_lags:-1]]  # Lagged level

        for i in range(1, n_lags + 1):
            X_list.append(diff_resid[n_lags - i:-i])

        X_adf = np.column_stack(X_list)

        # OLS
        coefs, residual_adf, rank, s = np.linalg.lstsq(X_adf, y_adf, rcond=None)
        gamma = coefs[0]

        # Standard error of gamma
        resid = y_adf - X_adf @ coefs
        mse = np.sum(resid ** 2) / (len(y_adf) - len(coefs))

        try:
            var_coef = mse * np.linalg.inv(X_adf.T @ X_adf)
            se_gamma = np.sqrt(var_coef[0, 0])
        except np.linalg.LinAlgError:
            se_gamma = np.std(resid) / (np.std(residuals[n_lags:-1]) * np.sqrt(len(y_adf)))

        t_stat = gamma / (se_gamma + 1e-10)

        result['coint_stat'] = float(t_stat)
        result['coint_beta'] = float(beta[1])

        # Engle-Granger critical values (approximate)
        # These are MORE NEGATIVE than standard ADF critical values
        # because we're testing residuals from an estimated relationship
        #
        # For n=50-100, two variables:
        # 1%: -4.07, 5%: -3.37, 10%: -3.07 (approximate)

        if t_stat < -4.07:
            result['coint_pvalue'] = 0.01
            result['is_cointegrated'] = True
        elif t_stat < -3.37:
            result['coint_pvalue'] = 0.05
            result['is_cointegrated'] = True
        elif t_stat < -3.07:
            result['coint_pvalue'] = 0.10
            result['is_cointegrated'] = False  # Marginal
        else:
            # Rough interpolation for larger p-values
            result['coint_pvalue'] = min(0.5, 0.1 * np.exp(0.5 * (t_stat + 3.07)))
            result['is_cointegrated'] = False

    except Exception:
        pass

    return result
