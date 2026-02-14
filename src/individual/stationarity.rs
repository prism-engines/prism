use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

/// Augmented Dickey-Fuller test (simple implementation).
/// Line-by-line port of manifold.primitives.tests.stationarity_tests._adf_simple
///
/// Returns (adf_statistic, p_value, used_lag).
/// Critical values are constant and handled by the Python bridge.
#[pyfunction]
#[pyo3(signature = (signal, max_lag=None, regression="c"))]
pub fn adf_test(
    signal: PyReadonlyArray1<f64>,
    max_lag: Option<usize>,
    regression: &str,
) -> PyResult<(f64, f64, usize)> {
    let raw = signal.as_slice()?;

    // Strip NaN (matching signal[~np.isnan(signal)])
    let y: Vec<f64> = raw.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = y.len();

    if n < 10 {
        return Ok((f64::NAN, f64::NAN, 0));
    }

    // Default max_lag: int(floor(12 * (n/100)^(1/4))), capped at n//3
    let ml = max_lag.unwrap_or_else(|| {
        let lag = (12.0 * (n as f64 / 100.0).powf(0.25)).floor() as usize;
        lag.min(n / 3)
    });

    // First difference: diff = y[1:] - y[:-1]
    let diff: Vec<f64> = (1..n).map(|i| y[i] - y[i - 1]).collect();
    // Lagged level: level_lag = y[:-1]
    let level_lag: Vec<f64> = y[..n - 1].to_vec();

    // Select lag using AIC
    let mut best_aic = f64::INFINITY;
    let mut best_lag = 1usize;

    for lag in 1..=ml {
        let ny = diff.len().saturating_sub(lag);
        if ny == 0 {
            continue;
        }
        let yy: Vec<f64> = diff[lag..].to_vec();
        let ny = yy.len();

        // Build X matrix: [ones, level_lag[lag:n-1], lagged_diffs..., (optional trend)]
        let mut ncols = 2 + lag; // constant + level + lag diffs
        if regression == "ct" {
            ncols += 1; // trend column
        }

        if ny < ncols + 2 {
            continue;
        }

        // Build X as flat row-major: ny rows, ncols columns
        let mut x = vec![0.0f64; ny * ncols];
        for i in 0..ny {
            x[i * ncols] = 1.0; // constant
            x[i * ncols + 1] = level_lag[lag + i]; // level_lag[lag:n-1]
            for li in 1..=lag {
                x[i * ncols + 1 + li] = diff[lag - li + i]; // diff[lag-li:n-1-li]
            }
            if regression == "ct" {
                x[i * ncols + ncols - 1] = i as f64; // trend
            }
        }

        // OLS: beta = (X^T X)^{-1} X^T y
        match ols_fit(&x, ny, ncols, &yy) {
            Some((beta, ssr)) => {
                let aic = ny as f64 * (ssr / ny as f64 + 1e-10).ln() + 2.0 * ncols as f64;
                if aic < best_aic {
                    best_aic = aic;
                    best_lag = lag;
                }
            }
            None => continue,
        }
    }

    // Final regression with best lag
    let lag = best_lag;
    let yy: Vec<f64> = diff[lag..].to_vec();
    let ny = yy.len();

    let mut ncols = 2 + lag;
    if regression == "ct" {
        ncols += 1;
    }

    if ny < ncols + 2 {
        return Ok((f64::NAN, f64::NAN, 0));
    }

    let mut x = vec![0.0f64; ny * ncols];
    for i in 0..ny {
        x[i * ncols] = 1.0;
        x[i * ncols + 1] = level_lag[lag + i];
        for li in 1..=lag {
            x[i * ncols + 1 + li] = diff[lag - li + i];
        }
        if regression == "ct" {
            x[i * ncols + ncols - 1] = i as f64;
        }
    }

    // Compute beta, residuals, and standard errors
    // X^T X
    let mut xtx = vec![0.0f64; ncols * ncols];
    for i in 0..ny {
        for j in 0..ncols {
            for k in 0..ncols {
                xtx[j * ncols + k] += x[i * ncols + j] * x[i * ncols + k];
            }
        }
    }

    // X^T y
    let mut xty = vec![0.0f64; ncols];
    for i in 0..ny {
        for j in 0..ncols {
            xty[j] += x[i * ncols + j] * yy[i];
        }
    }

    // Invert X^T X
    let xtx_inv = match mat_invert(&xtx, ncols) {
        Some(inv) => inv,
        None => return Ok((f64::NAN, f64::NAN, 0)),
    };

    // beta = (X^T X)^{-1} X^T y
    let mut beta = vec![0.0f64; ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            beta[i] += xtx_inv[i * ncols + j] * xty[j];
        }
    }

    // Residuals: r = y - X @ beta
    let mut ssr = 0.0f64;
    for i in 0..ny {
        let mut pred = 0.0;
        for j in 0..ncols {
            pred += x[i * ncols + j] * beta[j];
        }
        ssr += (yy[i] - pred).powi(2);
    }

    let sigma2 = ssr / (ny - ncols) as f64;

    // Standard error of beta[1] (level coefficient)
    let se = (sigma2 * xtx_inv[1 * ncols + 1]).sqrt();

    if se <= 0.0 || se.is_nan() {
        return Ok((f64::NAN, f64::NAN, 0));
    }

    let adf_stat = beta[1] / se;

    // Approximate p-value using normal CDF (matching scipy.stats.norm.cdf)
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = 2.0 * normal.cdf(adf_stat); // One-sided, doubled

    Ok((adf_stat, p_value, lag))
}

/// OLS fit: returns (beta, sum_squared_residuals) or None if singular.
fn ols_fit(x: &[f64], nrows: usize, ncols: usize, y: &[f64]) -> Option<(Vec<f64>, f64)> {
    // X^T X
    let mut xtx = vec![0.0f64; ncols * ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            for k in 0..ncols {
                xtx[j * ncols + k] += x[i * ncols + j] * x[i * ncols + k];
            }
        }
    }

    // X^T y
    let mut xty = vec![0.0f64; ncols];
    for i in 0..nrows {
        for j in 0..ncols {
            xty[j] += x[i * ncols + j] * y[i];
        }
    }

    // Invert X^T X
    let xtx_inv = mat_invert(&xtx, ncols)?;

    // beta = (X^T X)^{-1} X^T y
    let mut beta = vec![0.0f64; ncols];
    for i in 0..ncols {
        for j in 0..ncols {
            beta[i] += xtx_inv[i * ncols + j] * xty[j];
        }
    }

    // SSR
    let mut ssr = 0.0f64;
    for i in 0..nrows {
        let mut pred = 0.0;
        for j in 0..ncols {
            pred += x[i * ncols + j] * beta[j];
        }
        ssr += (y[i] - pred).powi(2);
    }

    Some((beta, ssr))
}

/// Invert a square matrix using Gauss-Jordan elimination.
/// Matrix stored as flat row-major array of n*n elements.
fn mat_invert(mat: &[f64], n: usize) -> Option<Vec<f64>> {
    // Build augmented matrix [mat | I]
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = mat[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_val = aug[col * 2 * n + col].abs();
        let mut max_row = col;
        for row in col + 1..n {
            let v = aug[row * 2 * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return None;
        }

        // Swap rows
        if max_row != col {
            for j in 0..2 * n {
                let a = col * 2 * n + j;
                let b = max_row * 2 * n + j;
                aug.swap(a, b);
            }
        }

        // Scale pivot row
        let pivot = aug[col * 2 * n + col];
        for j in 0..2 * n {
            aug[col * 2 * n + j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[row * 2 * n + col];
                for j in 0..2 * n {
                    aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    Some(inv)
}
