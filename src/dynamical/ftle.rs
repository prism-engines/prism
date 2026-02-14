use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// FTLE via local linearization of neighbor trajectories.
/// Line-by-line port of manifold.primitives.dynamical.ftle.ftle_local_linearization
///
/// Returns (ftle_values, confidence) arrays of length n_points
/// (last time_horizon entries are NaN/0.0).
#[pyfunction]
#[pyo3(signature = (trajectory, time_horizon=10, n_neighbors=10, epsilon=None))]
pub fn ftle_local_linearization<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<f64>,
    time_horizon: usize,
    n_neighbors: usize,
    epsilon: Option<f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let traj = trajectory.as_array();
    let (n_points, dim) = (traj.nrows(), traj.ncols());
    let n_valid = n_points.saturating_sub(time_horizon);

    // (matching: if n_valid < 10: return np.full(n_points, np.nan), np.full(n_points, 0.0))
    if n_valid < 10 || dim == 0 {
        return Ok((
            PyArray1::from_vec(py, vec![f64::NAN; n_points]),
            PyArray1::from_vec(py, vec![0.0; n_points]),
        ));
    }

    // Auto epsilon: 20th percentile of sampled pairwise distances
    let eps = epsilon.unwrap_or_else(|| {
        let sample_size = 100.min(n_points);
        // Use deterministic sampling (first sample_size points) for reproducibility
        let mut dists: Vec<f64> = Vec::new();
        for i in 0..sample_size {
            for j in 0..sample_size {
                if i != j {
                    let d: f64 = (0..dim)
                        .map(|d| (traj[[i, d]] - traj[[j, d]]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    dists.push(d);
                }
            }
        }
        if dists.is_empty() {
            return 1.0;
        }
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((0.20) * (dists.len() - 1) as f64) as usize;
        dists[idx]
    });

    let mut ftle = vec![f64::NAN; n_points];
    let mut confidence = vec![0.0f64; n_points];

    for i in 0..n_valid {
        // Find neighbors within epsilon ball
        let mut valid_neighbors: Vec<usize> = Vec::new();
        for j in 0..n_points {
            if j != i && j + time_horizon < n_points {
                let d: f64 = (0..dim)
                    .map(|dd| (traj[[i, dd]] - traj[[j, dd]]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if d < eps {
                    valid_neighbors.push(j);
                }
            }
        }

        // Fall back to k-nearest if not enough in epsilon ball
        if valid_neighbors.len() < n_neighbors {
            let mut dists: Vec<(usize, f64)> = (0..n_points)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f64 = (0..dim)
                        .map(|dd| (traj[[i, dd]] - traj[[j, dd]]).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    (j, d)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            valid_neighbors = dists
                .iter()
                .take(n_neighbors)
                .filter(|&&(j, _)| j + time_horizon < n_points)
                .map(|&(j, _)| j)
                .collect();
        }

        // (matching: if len(valid_neighbors) < dim + 1: continue)
        if valid_neighbors.len() < dim + 1 {
            continue;
        }

        // Build delta_x0 and delta_xT matrices
        let nn = valid_neighbors.len();
        // delta_x0[k][d] = trajectory[neighbor_k] - trajectory[i]
        let mut delta_x0 = vec![vec![0.0f64; dim]; nn];
        let mut delta_xt = vec![vec![0.0f64; dim]; nn];

        for (k, &j) in valid_neighbors.iter().enumerate() {
            for d in 0..dim {
                delta_x0[k][d] = traj[[j, d]] - traj[[i, d]];
                delta_xt[k][d] = traj[[j + time_horizon, d]] - traj[[i + time_horizon, d]];
            }
        }

        // Least squares: delta_xT = delta_x0 @ Phi_T
        // Solve via normal equations: (delta_x0^T @ delta_x0) @ Phi_T = delta_x0^T @ delta_xT
        let phi_t = match lstsq(&delta_x0, &delta_xt, dim) {
            Some(p) => p,
            None => continue,
        };

        // Phi = Phi_T^T (dim x dim)
        let mut phi = vec![vec![0.0f64; dim]; dim];
        for r in 0..dim {
            for c in 0..dim {
                phi[r][c] = phi_t[c][r];
            }
        }

        // Compute FTLE from largest singular value of Phi
        let sigma = sigma_max(&phi, dim);

        if sigma > 0.0 {
            ftle[i] = sigma.ln() / time_horizon as f64;
        }

        // Confidence based on fit quality (R²)
        // residuals = delta_xT - delta_x0 @ Phi_T
        let mut ss_res = 0.0f64;
        let mut ss_tot = 0.0f64;
        for k in 0..nn {
            for d in 0..dim {
                let mut pred = 0.0;
                for d2 in 0..dim {
                    pred += delta_x0[k][d2] * phi_t[d2][d];
                }
                ss_res += (delta_xt[k][d] - pred).powi(2);
                ss_tot += delta_xt[k][d].powi(2);
            }
        }

        if ss_res > 0.0 && ss_tot > 0.0 {
            let r2 = 1.0 - ss_res / ss_tot;
            confidence[i] = r2.clamp(0.0, 1.0);
        } else {
            confidence[i] = 0.5;
        }
    }

    Ok((
        PyArray1::from_vec(py, ftle),
        PyArray1::from_vec(py, confidence),
    ))
}

/// Solve least squares: A @ X = B via normal equations.
/// A is (m x n), B is (m x p), returns X (n x p).
fn lstsq(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let m = a.len();
    let p = b[0].len();

    // A^T A (n x n)
    let mut ata = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    // Invert A^T A
    let ata_inv = mat_invert(&ata, n)?;

    // A^T B (n x p)
    let mut atb = vec![vec![0.0f64; p]; n];
    for i in 0..n {
        for j in 0..p {
            for k in 0..m {
                atb[i][j] += a[k][i] * b[k][j];
            }
        }
    }

    // X = (A^T A)^{-1} @ A^T B
    let mut x = vec![vec![0.0f64; p]; n];
    for i in 0..n {
        for j in 0..p {
            for k in 0..n {
                x[i][j] += ata_inv[i][k] * atb[k][j];
            }
        }
    }

    Some(x)
}

/// Largest singular value of a square matrix via power iteration on M^T M.
fn sigma_max(mat: &[Vec<f64>], n: usize) -> f64 {
    // C = mat^T @ mat
    let mut c = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i][j] += mat[k][i] * mat[k][j];
            }
        }
    }

    // Power iteration to find largest eigenvalue of C
    let mut v = vec![1.0 / (n as f64).sqrt(); n];

    for _ in 0..200 {
        // w = C @ v
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += c[i][j] * v[j];
            }
        }

        // Normalize
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return 0.0;
        }
        v = w.iter().map(|x| x / norm).collect();
    }

    // Final eigenvalue estimate: v^T C v
    let mut cv = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            cv[i] += c[i][j] * v[j];
        }
    }
    let lambda: f64 = cv.iter().zip(v.iter()).map(|(a, b)| a * b).sum();

    if lambda > 0.0 {
        lambda.sqrt()
    } else {
        0.0
    }
}

/// Invert a square matrix using Gauss-Jordan elimination.
fn mat_invert(mat: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    // Build augmented matrix [mat | I]
    let mut aug = vec![vec![0.0f64; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = mat[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in col + 1..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return None;
        }

        if max_row != col {
            aug.swap(col, max_row);
        }

        let pivot = aug[col][col];
        for j in 0..2 * n {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2 * n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    let mut inv = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }

    Some(inv)
}

/// FTLE via direct perturbation of delay-embedded signal.
/// Returns (ftle_values, jacobian_norms).
/// Matches: manifold.primitives.dynamical.ftle.ftle_direct_perturbation
#[pyfunction]
#[pyo3(signature = (signal, dimension=3, delay=1, time_horizon=10, perturbation=1e-6, n_perturbations=10))]
pub fn ftle_direct_perturbation<'py>(
    py: Python<'py>,
    signal: PyReadonlyArray1<f64>,
    dimension: usize,
    delay: usize,
    time_horizon: usize,
    perturbation: f64,
    n_perturbations: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let y = signal.as_slice()?;
    let n = y.len();

    let n_points = n.saturating_sub((dimension - 1) * delay);
    if n_points < time_horizon + 2 {
        return Ok((
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    // Build embedding
    let embedded: Vec<Vec<f64>> = (0..n_points)
        .map(|i| (0..dimension).map(|d| y[i + d * delay]).collect())
        .collect();

    let n_valid = n_points - time_horizon;
    let mut ftle_vals = vec![0.0f64; n_valid];
    let mut jac_norms = vec![0.0f64; n_valid];

    for i in 0..n_valid {
        // Find nearest neighbor with temporal separation
        let mut best_dist = f64::INFINITY;
        let mut best_j = 0;
        for j in 0..n_valid {
            if (i as isize - j as isize).unsigned_abs() < dimension * delay {
                continue;
            }
            let dist: f64 = embedded[i]
                .iter()
                .zip(embedded[j].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < best_dist && dist > 1e-15 {
                best_dist = dist;
                best_j = j;
            }
        }

        if best_dist < f64::INFINITY {
            let d0 = best_dist;
            let dt: f64 = embedded[i + time_horizon]
                .iter()
                .zip(embedded[best_j + time_horizon].iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if d0 > 1e-15 && dt > 0.0 {
                ftle_vals[i] = (dt / d0).ln() / time_horizon as f64;
                jac_norms[i] = dt / d0;
            }
        }
    }

    Ok((
        PyArray1::from_vec(py, ftle_vals),
        PyArray1::from_vec(py, jac_norms),
    ))
}

/// Compute Cauchy-Green deformation tensor.
/// Returns (ftle_field, eigenvalues, eigenvectors_flat).
/// Matches: manifold.primitives.dynamical.ftle.compute_cauchy_green_tensor
#[pyfunction]
#[pyo3(signature = (trajectory, time_horizon, n_neighbors=10))]
pub fn compute_cauchy_green_tensor<'py>(
    py: Python<'py>,
    trajectory: PyReadonlyArray2<f64>,
    time_horizon: usize,
    n_neighbors: usize,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let traj = trajectory.as_array();
    let (n_points, dim) = (traj.nrows(), traj.ncols());

    if n_points < time_horizon + 2 {
        return Ok((
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
            PyArray1::from_vec(py, vec![]),
        ));
    }

    let n_valid = n_points - time_horizon;
    let mut ftle_field = vec![0.0f64; n_valid];
    let mut eigenvalues = vec![0.0f64; n_valid];
    let mut eigenvectors = vec![0.0f64; n_valid * dim];

    for i in 0..n_valid {
        let mut dists: Vec<(usize, f64)> = (0..n_valid)
            .filter(|&j| j != i)
            .map(|j| {
                let d: f64 = (0..dim)
                    .map(|d| (traj[[i, d]] - traj[[j, d]]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (j, d)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut max_stretch = 0.0f64;
        let mut best_dir = vec![0.0f64; dim];
        for &(j, d0) in dists.iter().take(n_neighbors) {
            if d0 < 1e-15 {
                continue;
            }
            let dt: f64 = (0..dim)
                .map(|d| (traj[[i + time_horizon, d]] - traj[[j + time_horizon, d]]).powi(2))
                .sum::<f64>()
                .sqrt();
            let stretch = dt / d0;
            if stretch > max_stretch {
                max_stretch = stretch;
                for d in 0..dim {
                    best_dir[d] = traj[[j, d]] - traj[[i, d]];
                }
            }
        }

        let lambda = max_stretch * max_stretch;
        eigenvalues[i] = lambda;
        if lambda > 0.0 {
            ftle_field[i] = lambda.ln() / (2.0 * time_horizon as f64);
        }

        let norm: f64 = best_dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        for d in 0..dim {
            eigenvectors[i * dim + d] = if norm > 1e-15 {
                best_dir[d] / norm
            } else {
                0.0
            };
        }
    }

    Ok((
        PyArray1::from_vec(py, ftle_field),
        PyArray1::from_vec(py, eigenvalues),
        PyArray1::from_vec(py, eigenvectors),
    ))
}

/// Detect LCS ridges from FTLE field.
/// Returns binary mask of ridge points.
/// Matches: manifold.primitives.dynamical.ftle.detect_lcs_ridges
#[pyfunction]
#[pyo3(signature = (ftle_field, trajectory, threshold_percentile=90.0))]
pub fn detect_lcs_ridges<'py>(
    py: Python<'py>,
    ftle_field: PyReadonlyArray1<f64>,
    trajectory: PyReadonlyArray2<f64>,
    threshold_percentile: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ftle = ftle_field.as_slice()?;
    let n = ftle.len();

    if n == 0 {
        return Ok(PyArray1::from_vec(py, vec![]));
    }

    let mut sorted = ftle.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((threshold_percentile / 100.0) * (n - 1) as f64).round() as usize;
    let threshold = sorted[idx.min(n - 1)];

    let mut ridges = vec![0.0f64; n];
    for i in 1..n - 1 {
        if ftle[i] > threshold && ftle[i] > ftle[i - 1] && ftle[i] > ftle[i + 1] {
            ridges[i] = 1.0;
        }
    }

    Ok(PyArray1::from_vec(py, ridges))
}
