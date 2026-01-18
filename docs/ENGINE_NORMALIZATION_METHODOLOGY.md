# Engine Normalization Methodology

## The Problem

PRISM engines produce outputs on wildly different scales:

| Engine | Typical Range | Scale |
|--------|---------------|-------|
| Hurst | 0.0 - 1.0 | Bounded |
| Entropy | 0.0 - 5.0+ | Unbounded |
| Spectral | Arbitrary | Power units |
| GARCH | 0.0 - ∞ | Volatility |
| RQA | 0.0 - 1.0 | Bounded |
| Lyapunov | -∞ to +∞ | Exponent |
| Wavelet | 0.0 - ∞ | Energy |

**Without normalization:**
- Delta computation biased toward high-variance engines
- An entropy change of 0.1 appears 10x more important than a Hurst change of 0.1
- ML feature importance reflects scale, not signal
- Geometry distances skewed toward arbitrary units
- "Hurst is noise" might just mean "Hurst has small numbers"

## The Solution

Normalize each engine to comparable scales using robust statistics computed from the full historical dataset.

### Normalization Methods

1. **Z-Score**: `(x - mean) / std`
   - Standard normalization
   - Sensitive to outliers

2. **Robust (Recommended)**: `(x - median) / IQR`
   - Uses median and interquartile range
   - Resistant to outliers
   - Best for data with fat tails

3. **Min-Max**: `(x - min) / (max - min)` → [0, 1]
   - Scales to bounded range
   - Sensitive to extreme values

4. **Rank**: Percentile rank → [0, 1]
   - Non-parametric
   - Loses magnitude information

### Configuration

Each engine has specific configuration:

```python
ENGINE_NORM_CONFIG = {
    'hurst': {
        'method': NormMethod.ROBUST,
        'clip_std': 4.0,
        'theoretical_range': (0.0, 1.0),
        'notes': 'Bounded [0,1], use robust to handle edge cases'
    },
    'entropy': {
        'method': NormMethod.ROBUST,
        'clip_std': 4.0,
        'theoretical_range': (0.0, None),  # Unbounded above
        'notes': 'Unbounded, varies by data complexity'
    },
    # ... all engines use ROBUST with 4.0 std clipping
}
```

## Implementation

### Step 1: Compute Global Statistics (Run Once)

Compute statistics across the FULL historical dataset:

```python
stats = compute_engine_statistics(observations)
```

Statistics computed per engine/metric:
- `mean`, `std` (for z-score)
- `median`, `iqr`, `q25`, `q75` (for robust)
- `min_val`, `max_val` (for min-max)
- `p01`, `p99` (for robust min-max)
- `n_observations` (for validation)

### Step 2: Apply Normalization

For each engine output:

```python
normalized = (raw_value - median) / IQR
normalized = clip(normalized, -clip_std/1.35, clip_std/1.35)  # ~[-3, 3]
```

### Step 3: Validate

Check that normalized outputs have:
- Mean near 0
- Std near 1
- Range within [-4, 4]

## Output Schema

### Long Format (Current)
```
signal_id | obs_date | engine | metric_name | metric_value | metric_value_norm
```

### Wide Format (Alternative)
```
signal_id | obs_date | hurst_raw | hurst_norm | entropy_raw | entropy_norm | ...
```

## File Outputs

```
data/vector/
├── normalization_engine_stats.parquet    # Global statistics
├── normalization_vectors_normalized.parquet  # Normalized vectors
└── normalization_validation.parquet      # Validation results
```

## Usage

```bash
# Full normalization pipeline
python -m prism.entry_points.engine_normalization

# Statistics only
python -m prism.entry_points.engine_normalization --stats-only

# Validate existing normalization
python -m prism.entry_points.engine_normalization --validate
```

## Before vs After

### Before Normalization
| Engine | Value | "Importance" |
|--------|-------|--------------|
| Entropy | 4.2 | HIGH (big number) |
| Hurst | 0.4 | LOW (small number) |
| GARCH | 150.0 | EXTREME |

### After Normalization
| Engine | Value | Interpretation |
|--------|-------|----------------|
| Entropy | 0.8 | 0.8 IQRs above median |
| Hurst | 0.4 | 0.4 IQRs above median |
| GARCH | 1.2 | 1.2 IQRs above median |

Now all engines are on comparable scales, and geometry/delta computations are fair.

## Integration Points

### Signal Vector (signal_vector.py)
After raw engine computation, apply normalization before saving.

### Domain Geometry (domain_geometry_alt.py)
Use `metric_value_norm` column for all computations.

### Delta Computation
Delta = normalized[t] - normalized[t-1]
Now a delta of 0.5 means the same thing regardless of engine.

## Why This Matters

The fundamental weakness discovered: **Without normalization at the engine output level, ALL downstream analysis is biased toward engines with larger absolute values.**

This explains why some engines appeared to "not work" - they were simply measuring in smaller units. The geometry doesn't care about units; it only sees the numbers.

Normalization is the foundation. Everything downstream inherits it.
