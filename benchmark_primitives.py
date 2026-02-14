"""
Benchmark all Manifold primitive functions on real FD004 signals.

Usage:
    cd ~/manifold
    ./venv/bin/python benchmark_primitives.py
"""

import time
import numpy as np
import polars as pl
from typing import Callable, Any, List, Tuple

# ── Load FD004 signals ──────────────────────────────────────────────────────

OBS_PATH = "/Users/jasonrudder/domains/cmapss/FD_004/train/observations.parquet"

SIGNALS = [
    ("engine_1", "T24"),
    ("engine_1", "Nf"),
    ("engine_100", "Ps30"),
    ("engine_200", "BPR"),
    ("engine_50", "phi"),
]


def load_signals() -> dict:
    """Load target signals from FD004 observations.parquet."""
    df = pl.read_parquet(OBS_PATH)
    out = {}
    for cohort, sig_id in SIGNALS:
        sub = df.filter(
            (pl.col("cohort") == cohort) & (pl.col("signal_id") == sig_id)
        ).sort("I")
        if len(sub) > 0:
            vals = sub["value"].to_numpy().astype(np.float64)
            label = f"{cohort}/{sig_id}"
            out[label] = vals
        else:
            print(f"  SKIP {cohort}/{sig_id}: not found in observations")
    return out


# ── Timing harness ──────────────────────────────────────────────────────────

N_REPEATS = 10


def time_fn(fn: Callable, *args, **kwargs) -> float:
    """Time a function N_REPEATS times, return average ms."""
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        try:
            fn(*args, **kwargs)
        except Exception:
            return float("nan")
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.mean(times)


# ── Import primitives ──────────────────────────────────────────────────────

def import_primitives():
    """Import all primitive functions. Returns list of (name, category, fn, needs_pair, needs_matrix, needs_cloud)."""
    primitives = []

    # --- individual ---
    try:
        from manifold.primitives.individual.fractal import hurst_exponent
        primitives.append(("hurst_exponent", "individual", hurst_exponent, False, False, False))
    except ImportError as e:
        print(f"  SKIP hurst_exponent: {e}")

    try:
        from manifold.primitives.individual.entropy import permutation_entropy
        primitives.append(("permutation_entropy", "individual", permutation_entropy, False, False, False))
    except ImportError as e:
        print(f"  SKIP permutation_entropy: {e}")

    try:
        from manifold.primitives.individual.entropy import sample_entropy
        primitives.append(("sample_entropy", "individual", sample_entropy, False, False, False))
    except ImportError as e:
        print(f"  SKIP sample_entropy: {e}")

    try:
        from manifold.primitives.individual.spectral import spectral_entropy
        primitives.append(("spectral_entropy", "individual", spectral_entropy, False, False, False))
    except ImportError as e:
        print(f"  SKIP spectral_entropy: {e}")

    try:
        from manifold.primitives.individual.spectral import psd
        primitives.append(("psd", "individual", psd, False, False, False))
    except ImportError as e:
        print(f"  SKIP psd: {e}")

    try:
        from manifold.primitives.individual.spectral import spectral_centroid
        primitives.append(("spectral_centroid", "individual", spectral_centroid, False, False, False))
    except ImportError as e:
        print(f"  SKIP spectral_centroid: {e}")

    try:
        from manifold.primitives.individual.statistics import kurtosis
        primitives.append(("kurtosis", "individual", kurtosis, False, False, False))
    except ImportError as e:
        print(f"  SKIP kurtosis: {e}")

    try:
        from manifold.primitives.individual.statistics import skewness
        primitives.append(("skewness", "individual", skewness, False, False, False))
    except ImportError as e:
        print(f"  SKIP skewness: {e}")

    try:
        from manifold.primitives.individual.statistics import crest_factor
        primitives.append(("crest_factor", "individual", crest_factor, False, False, False))
    except ImportError as e:
        print(f"  SKIP crest_factor: {e}")

    try:
        from manifold.primitives.tests.stationarity_tests import adf_test
        primitives.append(("adf_test", "individual", adf_test, False, False, False))
    except ImportError as e:
        print(f"  SKIP adf_test: {e}")

    # --- dynamical ---
    try:
        from manifold.primitives.dynamical.lyapunov import lyapunov_rosenstein
        primitives.append(("lyapunov_rosenstein", "dynamical", lyapunov_rosenstein, False, False, False))
    except ImportError as e:
        print(f"  SKIP lyapunov_rosenstein: {e}")

    try:
        from manifold.primitives.dynamical.ftle import ftle_local_linearization
        # ftle needs a trajectory (2D array), handled specially below
        primitives.append(("ftle_local_linearization", "dynamical", ftle_local_linearization, False, False, False))
    except ImportError as e:
        print(f"  SKIP ftle_local_linearization: {e}")

    # --- embedding ---
    try:
        from manifold.primitives.embedding.delay import time_delay_embedding
        primitives.append(("time_delay_embedding", "embedding", time_delay_embedding, False, False, False))
    except ImportError as e:
        print(f"  SKIP time_delay_embedding: {e}")

    try:
        from manifold.primitives.embedding.delay import estimate_tau_ami
        primitives.append(("estimate_tau_ami", "embedding", estimate_tau_ami, False, False, False))
    except ImportError as e:
        print(f"  SKIP estimate_tau_ami: {e}")

    try:
        from manifold.primitives.embedding.delay import estimate_embedding_dim_cao
        primitives.append(("estimate_embedding_dim_cao", "embedding", estimate_embedding_dim_cao, False, False, False))
    except ImportError as e:
        print(f"  SKIP estimate_embedding_dim_cao: {e}")

    # --- pairwise (need two signals) ---
    try:
        from manifold.primitives.pairwise.causality import granger_causality
        primitives.append(("granger_causality", "pairwise", granger_causality, True, False, False))
    except ImportError as e:
        print(f"  SKIP granger_causality: {e}")

    try:
        from manifold.primitives.pairwise.information import transfer_entropy
        primitives.append(("transfer_entropy", "pairwise", transfer_entropy, True, False, False))
    except ImportError as e:
        print(f"  SKIP transfer_entropy: {e}")

    try:
        from manifold.primitives.pairwise.distance import dynamic_time_warping
        primitives.append(("dynamic_time_warping", "pairwise", dynamic_time_warping, True, False, False))
    except ImportError as e:
        print(f"  SKIP dynamic_time_warping: {e}")

    try:
        from manifold.primitives.pairwise.information import mutual_information
        primitives.append(("mutual_information", "pairwise", mutual_information, True, False, False))
    except ImportError as e:
        print(f"  SKIP mutual_information: {e}")

    try:
        from manifold.primitives.pairwise.correlation import correlation as pearson_r
        primitives.append(("pearson_r", "pairwise", pearson_r, True, False, False))
    except ImportError as e:
        print(f"  SKIP pearson_r: {e}")

    try:
        from manifold.primitives.individual.similarity import spearman_correlation
        primitives.append(("spearman_correlation", "pairwise", spearman_correlation, True, False, False))
    except ImportError as e:
        print(f"  SKIP spearman_correlation: {e}")

    # --- matrix (need matrix input) ---
    try:
        from manifold.primitives.matrix.decomposition import eigendecomposition
        primitives.append(("eigendecomposition", "matrix", eigendecomposition, False, True, False))
    except ImportError as e:
        print(f"  SKIP eigendecomposition: {e}")

    try:
        from manifold.primitives.matrix.covariance import covariance_matrix
        primitives.append(("covariance_matrix", "matrix", covariance_matrix, False, True, False))
    except ImportError as e:
        print(f"  SKIP covariance_matrix: {e}")

    # --- topology (need point cloud) ---
    try:
        from manifold.primitives.topology.persistence import persistence_diagram
        primitives.append(("persistence_diagram", "topology", persistence_diagram, False, False, True))
    except ImportError as e:
        print(f"  SKIP persistence_diagram: {e}")

    try:
        from manifold.primitives.topology.persistence import betti_numbers
        primitives.append(("betti_numbers", "topology", betti_numbers, False, False, True))
    except ImportError as e:
        print(f"  SKIP betti_numbers: {e}")

    try:
        from manifold.primitives.topology.persistence import persistence_entropy
        primitives.append(("persistence_entropy", "topology", persistence_entropy, False, False, True))
    except ImportError as e:
        print(f"  SKIP persistence_entropy: {e}")

    return primitives


# ── Run benchmarks ──────────────────────────────────────────────────────────

def build_matrix(signals: dict) -> np.ndarray:
    """Build a matrix from signal windows for matrix primitives."""
    arrays = []
    for vals in signals.values():
        arrays.append(vals[:170])  # trim to shortest common length
    return np.column_stack(arrays)


def build_trajectory(signal: np.ndarray, dim: int = 3, tau: int = 1) -> np.ndarray:
    """Build delay-embedding trajectory for FTLE."""
    n = len(signal) - (dim - 1) * tau
    traj = np.empty((n, dim))
    for d in range(dim):
        traj[:, d] = signal[d * tau: d * tau + n]
    return traj


def run_benchmarks():
    print("=" * 72)
    print("Manifold Primitives Benchmark — FD004 Data")
    print("=" * 72)
    print()

    # Load signals
    print("Loading FD004 signals...")
    signals = load_signals()
    signal_names = list(signals.keys())
    signal_arrays = list(signals.values())

    for name, arr in signals.items():
        print(f"  {name}: {len(arr)} samples")
    print()

    # Import primitives
    print("Importing primitives...")
    primitives = import_primitives()
    print(f"  Loaded {len(primitives)} primitives")
    print()

    # Build derived inputs
    matrix = build_matrix(signals)
    print(f"Matrix shape: {matrix.shape}")

    # Build point clouds for topology (delay embedding of each signal)
    clouds = {}
    for name, arr in signals.items():
        clouds[name] = build_trajectory(arr, dim=3, tau=1)

    # Pre-compute persistence diagrams for betti_numbers and persistence_entropy
    try:
        from manifold.primitives.topology.persistence import persistence_diagram as pd_fn
        diagram_cache = {}
        for name, cloud in clouds.items():
            try:
                diagram_cache[name] = pd_fn(cloud[:100])  # limit to 100 points for speed
            except Exception:
                pass
    except ImportError:
        diagram_cache = {}

    print()

    # Create a second signal for pairwise benchmarks (use Nf paired with each)
    pair_b = signal_arrays[1] if len(signal_arrays) > 1 else signal_arrays[0]

    # Run benchmarks
    results: List[Tuple[str, str, str, float]] = []

    print(f"{'Primitive':<28} {'Signal':<20} {'Avg (ms)':>10} {'Calls/sec':>10}")
    print("\u2500" * 72)

    for prim_name, category, fn, needs_pair, needs_matrix, needs_cloud in primitives:
        if needs_matrix:
            # Matrix primitives: run once on the matrix
            if prim_name == "eigendecomposition":
                # eigendecomposition needs a square matrix — use covariance
                from manifold.primitives.matrix.covariance import covariance_matrix as cov_fn
                cov = cov_fn(matrix)
                avg_ms = time_fn(fn, cov)
            elif prim_name == "covariance_matrix":
                avg_ms = time_fn(fn, matrix)
            else:
                avg_ms = time_fn(fn, matrix)

            sig_label = f"matrix({matrix.shape[0]}x{matrix.shape[1]})"
            calls_sec = 1000.0 / avg_ms if avg_ms > 0 and not np.isnan(avg_ms) else float("nan")
            print(f"{prim_name:<28} {sig_label:<20} {avg_ms:>10.2f} {calls_sec:>10.1f}")
            results.append((prim_name, category, sig_label, avg_ms))

        elif needs_cloud:
            # Topology primitives: run on point clouds
            for sig_name in signal_names:
                cloud = clouds.get(sig_name)
                if cloud is None:
                    continue

                if prim_name == "persistence_diagram":
                    avg_ms = time_fn(fn, cloud[:100])
                elif prim_name in ("betti_numbers", "persistence_entropy"):
                    diagrams = diagram_cache.get(sig_name)
                    if diagrams is None:
                        avg_ms = float("nan")
                    else:
                        avg_ms = time_fn(fn, diagrams)
                else:
                    avg_ms = time_fn(fn, cloud)

                calls_sec = 1000.0 / avg_ms if avg_ms > 0 and not np.isnan(avg_ms) else float("nan")
                short_name = sig_name.split("/")[1]
                print(f"{prim_name:<28} {short_name:<20} {avg_ms:>10.2f} {calls_sec:>10.1f}")
                results.append((prim_name, category, short_name, avg_ms))

        elif needs_pair:
            # Pairwise: pair each signal with engine_1/Nf
            for sig_name, arr in signals.items():
                if sig_name == "engine_1/Nf":
                    continue  # skip self-pair
                min_len = min(len(arr), len(pair_b))
                a_trimmed = arr[:min_len]
                b_trimmed = pair_b[:min_len]
                avg_ms = time_fn(fn, a_trimmed, b_trimmed)
                calls_sec = 1000.0 / avg_ms if avg_ms > 0 and not np.isnan(avg_ms) else float("nan")
                short_name = sig_name.split("/")[1]
                print(f"{prim_name:<28} {short_name:<20} {avg_ms:>10.2f} {calls_sec:>10.1f}")
                results.append((prim_name, category, short_name, avg_ms))

        else:
            # Single-signal primitives
            for sig_name, arr in signals.items():
                if prim_name == "ftle_local_linearization":
                    traj = build_trajectory(arr, dim=3, tau=1)
                    avg_ms = time_fn(fn, traj)
                else:
                    avg_ms = time_fn(fn, arr)

                calls_sec = 1000.0 / avg_ms if avg_ms > 0 and not np.isnan(avg_ms) else float("nan")
                short_name = sig_name.split("/")[1]
                print(f"{prim_name:<28} {short_name:<20} {avg_ms:>10.2f} {calls_sec:>10.1f}")
                results.append((prim_name, category, short_name, avg_ms))

    # ── Summary: Top 10 slowest primitives ──────────────────────────────
    print()
    print("=" * 72)
    print("TOP 10 SLOWEST PRIMITIVES (by average ms across all signals)")
    print("=" * 72)

    # Aggregate by primitive name: mean of all signal benchmarks
    from collections import defaultdict
    agg = defaultdict(list)
    for prim_name, category, sig, ms in results:
        if not np.isnan(ms):
            agg[(prim_name, category)].append(ms)

    summary = []
    for (prim_name, category), ms_list in agg.items():
        avg = np.mean(ms_list)
        summary.append((prim_name, category, avg, len(ms_list)))

    summary.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Rank':<6} {'Primitive':<28} {'Category':<12} {'Avg (ms)':>10} {'Signals':>8}")
    print("\u2500" * 68)
    for i, (name, cat, avg, n_sigs) in enumerate(summary[:10], 1):
        print(f"{i:<6} {name:<28} {cat:<12} {avg:>10.2f} {n_sigs:>8}")

    # ── Estimated pipeline impact ───────────────────────────────────────
    # Rough estimate: each primitive is called once per (cohort, signal, window)
    # FD004 has ~250 engines × 21 signals × ~1 window = ~5250 calls per primitive
    EST_CALLS = 5250

    print()
    print("=" * 72)
    print(f"ESTIMATED PIPELINE IMPACT (assuming ~{EST_CALLS:,} calls per primitive)")
    print("=" * 72)
    print(f"\n{'Rank':<6} {'Primitive':<28} {'Avg (ms)':>10} {'Est Total (s)':>14} {'Priority':>10}")
    print("\u2500" * 72)

    for i, (name, cat, avg, n_sigs) in enumerate(summary[:10], 1):
        total_s = (avg * EST_CALLS) / 1000.0
        priority = "HIGH" if total_s > 10 else "MED" if total_s > 1 else "LOW"
        print(f"{i:<6} {name:<28} {avg:>10.2f} {total_s:>14.1f} {priority:>10}")


if __name__ == "__main__":
    run_benchmarks()
