# PRISM

**Geometry leads — signals follow.**

PRISM computes signal geometry for industrial diagnostics and research discovery. It extracts the mathematical structure that predicts system behavior.

```bash
pip install prism-signal
```

---

## What It Does

PRISM transforms raw observations into geometric features that reveal system health:

```
observations.parquet     Raw sensor data
       ↓
signal_vector.parquet    Per-signal metrics (36 features)
       ↓
state_vector.parquet     System centroid (WHERE)
       ↓
state_geometry.parquet   Eigenstructure (SHAPE)
       ↓
effective_dim            The number that predicts failure
```

**The key insight:** Systems lose dimensionality before failure. Signals that once varied independently begin to move together. This "dimensional collapse" appears in the eigenvalue spectrum before physical symptoms emerge.

`effective_dim` shows 63% importance in predicting remaining useful life (RUL).

---

## Quick Start

```bash
# Run the pipeline
python -m prism.entry_points.stage_01_signal_vector manifest.yaml
python -m prism.entry_points.stage_02_state_vector signal_vector.parquet typology.parquet
python -m prism.entry_points.stage_03_state_geometry signal_vector.parquet state_vector.parquet
```

Or use Python directly:

```python
from prism.entry_points.stage_01_signal_vector import run

df = run(
    observations_path="observations.parquet",
    output_path="signal_vector.parquet",
    manifest=manifest,
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ENTRY POINTS                           │
│              Orchestration: parquet → parquet               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        ENGINES                              │
│         Orchestrate primitives, return dicts                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      PRIMITIVES                             │
│              Pure math: numpy → float/array                 │
└─────────────────────────────────────────────────────────────┘
```

PRISM computes. It does not interpret. For classification and visualization, see [ORTHON](https://github.com/prism-engines/orthon).

---

## Features Computed

**Per-signal (stage_01):**
- Shape: kurtosis, skewness, crest factor
- Memory: ACF decay, Hurst exponent
- Spectral: entropy, centroid, dominant frequency, bandwidth
- Complexity: permutation entropy, sample entropy
- Trend: slope, R², CUSUM

**Cross-signal (stage_02, stage_03):**
- State vector: centroid position in feature space
- Eigenvalues: variance distribution across principal components
- Effective dimension: participation ratio of eigenvalues
- Condition number: ratio of largest to smallest eigenvalue

---

## Input Format

PRISM expects `observations.parquet` with columns:

| Column | Type | Description |
|--------|------|-------------|
| `signal_id` | string | Unique signal identifier |
| `I` | int | Sequential index (0, 1, 2, ...) |
| `value` | float | Measurement value |
| `cohort` | string | (optional) Grouping for multi-unit analysis |

Plus a `manifest.yaml` specifying window size, stride, and engines per signal.

---

## Requirements

- Python 3.10+
- numpy, polars, scipy
- joblib (parallel processing)

---

## Citation

If PRISM helps your research, please cite:

```
@software{prism2026,
  title = {PRISM: Signal Geometry for Industrial Diagnostics},
  author = {Rudder, Jason and Rudder, Avery},
  year = {2026},
  url = {https://github.com/prism-engines/prism}
}
```

---

## License

MIT

---

## Acknowledgments

The dimensional collapse insight that makes this work came from Avery Rudder, who identified that `effective_dim` dominates RUL prediction. Systems lose coherence before they lose function.

*geometry leads — ørthon*