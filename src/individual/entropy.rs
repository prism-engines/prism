use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Sample entropy.
/// Matches: manifold.primitives.individual.entropy.sample_entropy
#[pyfunction]
#[pyo3(signature = (signal, m=2, r=None))]
pub fn sample_entropy(signal: PyReadonlyArray1<f64>, m: usize, r: Option<f64>) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();

    if n < m + 2 {
        return Ok(f64::NAN);
    }

    // Default tolerance: 0.2 * std
    let tolerance = match r {
        Some(val) => val,
        None => {
            let mean: f64 = y.iter().sum::<f64>() / n as f64;
            let var: f64 = y.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            0.2 * var.sqrt()
        }
    };

    if tolerance <= 0.0 {
        return Ok(f64::NAN);
    }

    let count_matches = |template_len: usize| -> usize {
        let mut count = 0usize;
        for i in 0..n - template_len {
            for j in (i + 1)..n - template_len {
                let mut matched = true;
                for k in 0..template_len {
                    if (y[i + k] - y[j + k]).abs() > tolerance {
                        matched = false;
                        break;
                    }
                }
                if matched {
                    count += 1;
                }
            }
        }
        count
    };

    let a = count_matches(m + 1) as f64;
    let b = count_matches(m) as f64;

    if b == 0.0 {
        return Ok(f64::NAN);
    }

    Ok(-(a / b).ln())
}

/// Permutation entropy.
/// Matches: manifold.primitives.individual.entropy.permutation_entropy
#[pyfunction]
#[pyo3(signature = (signal, order=3, delay=1, normalize=true))]
pub fn permutation_entropy(
    signal: PyReadonlyArray1<f64>,
    order: usize,
    delay: usize,
    normalize: bool,
) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();

    if n < order * delay {
        return Ok(f64::NAN);
    }

    let n_patterns = n - (order - 1) * delay;
    let mut counts: HashMap<Vec<usize>, usize> = HashMap::new();

    for i in 0..n_patterns {
        // Extract pattern indices
        let mut indices: Vec<usize> = (0..order).collect();
        // Sort by value at each delay position
        indices.sort_by(|&a, &b| {
            y[i + a * delay]
                .partial_cmp(&y[i + b * delay])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        *counts.entry(indices).or_insert(0) += 1;
    }

    // Shannon entropy of permutation distribution (log2, matching Python)
    let total = n_patterns as f64;
    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    if normalize {
        let max_entropy = (factorial(order) as f64).log2();
        if max_entropy > 0.0 {
            entropy /= max_entropy;
        }
    }

    Ok(entropy)
}

fn factorial(n: usize) -> usize {
    (1..=n).product()
}

/// Approximate entropy.
/// Matches: manifold.primitives.individual.entropy.approximate_entropy
#[pyfunction]
#[pyo3(signature = (signal, m=2, r=None))]
pub fn approximate_entropy(
    signal: PyReadonlyArray1<f64>,
    m: usize,
    r: Option<f64>,
) -> PyResult<f64> {
    let y = signal.as_slice()?;
    let n = y.len();

    if n < m + 2 {
        return Ok(f64::NAN);
    }

    let tolerance = match r {
        Some(val) => val,
        None => {
            let mean: f64 = y.iter().sum::<f64>() / n as f64;
            let var: f64 = y.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            0.2 * var.sqrt()
        }
    };

    if tolerance <= 0.0 {
        return Ok(f64::NAN);
    }

    let phi = |template_len: usize| -> f64 {
        let nn = n - template_len + 1;
        let mut counts = vec![0usize; nn];

        for i in 0..nn {
            for j in 0..nn {
                let mut matched = true;
                for k in 0..template_len {
                    if (y[i + k] - y[j + k]).abs() > tolerance {
                        matched = false;
                        break;
                    }
                }
                if matched {
                    counts[i] += 1;
                }
            }
        }

        counts
            .iter()
            .map(|&c| (c as f64 / nn as f64).ln())
            .sum::<f64>()
            / nn as f64
    };

    Ok(phi(m) - phi(m + 1))
}
