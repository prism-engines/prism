"""
Tests for non-uniform signal_0 spacing fixes.

Validates that all patched stages produce correct results on both
uniform (time axis) and non-uniform (reaxis) spacing.

Core principle: on uniform spacing (dt=1), patched code MUST produce
identical results to the original code. On non-uniform spacing, it
must produce physically correct derivatives.
"""

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────
# Helpers: derivative computation (mirrors patched logic)
# ─────────────────────────────────────────────────────────────────────

def velocity_uniform(x):
    """Original: np.diff(x) — assumes dt=1."""
    return np.diff(x, axis=0)

def velocity_nonuniform(x, s0):
    """Patched: np.diff(x) / np.diff(s0)."""
    dx = np.diff(x, axis=0)
    dt = np.diff(s0)
    dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
    if x.ndim == 1:
        return dx / dt
    return dx / dt[:, np.newaxis]

def accel_nonuniform(v, dt):
    """Patched: np.diff(v) / dt_mid."""
    dv = np.diff(v, axis=0)
    dt_mid = (dt[:-1] + dt[1:]) / 2.0
    dt_mid = np.where(np.abs(dt_mid) < 1e-12, 1e-12, dt_mid)
    if v.ndim == 1:
        return dv / dt_mid
    return dv / dt_mid[:, np.newaxis]


# ─────────────────────────────────────────────────────────────────────
# Tests: Backward compatibility (uniform spacing)
# ─────────────────────────────────────────────────────────────────────

class TestUniformSpacingIdentical:
    """On uniform spacing, patched code must match original exactly."""

    def test_velocity_1d_uniform(self):
        x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        s0 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # uniform dt=1

        v_old = velocity_uniform(x)
        v_new = velocity_nonuniform(x, s0)

        np.testing.assert_allclose(v_old, v_new, atol=1e-12)

    def test_velocity_2d_uniform(self):
        x = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 9.0], [10.0, 14.0]])
        s0 = np.arange(4, dtype=float)

        v_old = velocity_uniform(x)
        v_new = velocity_nonuniform(x, s0)

        np.testing.assert_allclose(v_old, v_new, atol=1e-12)

    def test_gradient_1d_uniform(self):
        """np.gradient(y, coords) == np.gradient(y) when coords are 0,1,2,..."""
        y = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        coords = np.arange(5, dtype=float)

        g_old = np.gradient(y)
        g_new = np.gradient(y, coords)

        np.testing.assert_allclose(g_old, g_new, atol=1e-12)


# ─────────────────────────────────────────────────────────────────────
# Tests: Non-uniform spacing correctness
# ─────────────────────────────────────────────────────────────────────

class TestNonUniformCorrectness:
    """On non-uniform spacing, derivatives must be physically correct."""

    def test_linear_function_exact(self):
        """f(x) = 2x → f'(x) = 2 regardless of spacing."""
        s0 = np.array([0.0, 1.0, 3.0, 7.0, 15.0])  # very non-uniform
        x = 2.0 * s0  # linear

        v = velocity_nonuniform(x, s0)

        # Every velocity should be exactly 2.0
        np.testing.assert_allclose(v, 2.0, atol=1e-10)

    def test_linear_function_old_code_wrong(self):
        """Same linear function, old code gives wrong answer on non-uniform."""
        s0 = np.array([0.0, 1.0, 3.0, 7.0, 15.0])
        x = 2.0 * s0

        v_old = velocity_uniform(x)

        # Old code gives [2, 4, 8, 16] — wrong! Should be [2, 2, 2, 2]
        assert not np.allclose(v_old, 2.0), \
            "Old code should give wrong answer on non-uniform spacing"
        # Specifically: np.diff([0, 2, 6, 14, 30]) = [2, 4, 8, 16]
        np.testing.assert_allclose(v_old, [2.0, 4.0, 8.0, 16.0])

    def test_quadratic_function(self):
        """f(x) = x² → f'(x) = 2x. Finite differences approximate this."""
        s0 = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        x = s0 ** 2

        v = velocity_nonuniform(x, s0)

        # Forward difference: f'(xi) ≈ (f(xi+1) - f(xi)) / (xi+1 - xi)
        # At s0=0: (1-0)/(1-0) = 1.0, true f'(0) = 0
        # At s0=1: (9-1)/(3-1) = 4.0, true f'(1) = 2
        # At s0=3: (36-9)/(6-3) = 9.0, true f'(3) = 6
        # Forward differences of quadratic have known error, but should
        # be closer to truth than old code
        expected_forward = np.array([1.0, 4.0, 9.0, 16.0])
        np.testing.assert_allclose(v, expected_forward, atol=1e-10)

    def test_acceleration_linear_velocity(self):
        """If velocity is constant, acceleration should be zero."""
        s0 = np.array([0.0, 1.0, 3.0, 7.0, 15.0])
        x = 5.0 * s0  # constant velocity = 5

        v = velocity_nonuniform(x, s0)
        dt = np.diff(s0)
        dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
        a = accel_nonuniform(v, dt)

        np.testing.assert_allclose(a, 0.0, atol=1e-10)

    def test_tiebreaker_spacing(self):
        """Simulates the fractional tiebreaker fix: 390.0, 390.17, 390.33, 392.0"""
        s0 = np.array([390.0, 390.167, 390.333, 392.0, 394.0])
        x = np.array([10.0, 10.5, 11.0, 15.0, 20.0])

        v = velocity_nonuniform(x, s0)
        dt = np.diff(s0)

        # First three: closely spaced → large velocity (fast change per unit s0)
        # dt[0] = 0.167, dx[0] = 0.5 → v = 2.994
        # Last: dt = 2.0, dx = 5.0 → v = 2.5
        assert v[0] > 2.5  # steep per-unit change
        assert abs(v[3] - 2.5) < 0.01  # expected

    def test_dt_floor_prevents_inf(self):
        """Zero-width dt should be floored, not produce inf."""
        s0 = np.array([1.0, 1.0, 2.0, 3.0])  # duplicate s0 value
        x = np.array([5.0, 6.0, 7.0, 8.0])

        v = velocity_nonuniform(x, s0)

        # Should not contain inf or nan
        assert np.all(np.isfinite(v))


# ─────────────────────────────────────────────────────────────────────
# Tests: np.gradient with coordinates (Stage 07, 23)
# ─────────────────────────────────────────────────────────────────────

class TestGradientWithCoords:
    """np.gradient(y, coords) correctness for Stages 07 and 23."""

    def test_linear_exact(self):
        """Linear function: gradient should be constant."""
        coords = np.array([0.0, 1.0, 4.0, 10.0, 20.0])
        y = 3.0 * coords + 1.0

        g = np.gradient(y, coords)
        np.testing.assert_allclose(g, 3.0, atol=1e-10)

    def test_ftle_gradient_nonuniform(self):
        """Simulates FTLE gradient computation in ridge_proximity."""
        # Suppose FTLE increases linearly with signal_0
        ftle_i = np.array([0.0, 1.0, 5.0, 6.0, 10.0])
        ftle_vals = np.array([0.01, 0.02, 0.06, 0.07, 0.11])
        # Slope is 0.01 per unit signal_0

        grad = np.gradient(ftle_vals, ftle_i)
        np.testing.assert_allclose(grad, 0.01, atol=0.002)

    def test_second_derivative_quadratic(self):
        """f(x) = x² → f''(x) = 2. np.gradient applied twice."""
        coords = np.array([0.0, 1.0, 3.0, 6.0, 10.0, 15.0])
        y = coords ** 2

        dy = np.gradient(y, coords)
        d2y = np.gradient(dy, coords)

        # Second derivative of x² = 2 everywhere (approximately)
        # np.gradient applied twice on non-uniform grids has larger error
        # near boundaries. Deep interior points should be close to 2.0.
        np.testing.assert_allclose(d2y[2:-2], 2.0, atol=0.1)


# ─────────────────────────────────────────────────────────────────────
# Tests: Thermodynamics velocity (Stage 09a)
# ─────────────────────────────────────────────────────────────────────

class TestThermodynamicsVelocity:
    """Stage 09a: effective_dim velocity for temperature computation."""

    def test_constant_eff_dim(self):
        """Constant effective_dim → zero velocity → zero temperature."""
        eff_dims = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
        i_vals = np.array([0.0, 1.0, 3.0, 7.0, 15.0])

        dt = np.diff(i_vals)
        dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
        velocities = np.diff(eff_dims) / dt

        assert np.allclose(velocities, 0.0)

    def test_linear_eff_dim(self):
        """Linear effective_dim → constant velocity."""
        i_vals = np.array([0.0, 1.0, 3.0, 7.0, 15.0])
        eff_dims = 2.0 + 0.1 * i_vals  # linear: slope = 0.1

        dt = np.diff(i_vals)
        dt = np.where(np.abs(dt) < 1e-12, 1e-12, dt)
        velocities = np.diff(eff_dims) / dt

        np.testing.assert_allclose(velocities, 0.1, atol=1e-10)

    def test_old_code_wrong_nonuniform(self):
        """Old np.diff(eff_dims) gives wrong velocity on non-uniform I."""
        i_vals = np.array([0.0, 1.0, 3.0, 7.0, 15.0])
        eff_dims = 2.0 + 0.1 * i_vals

        old_vel = np.diff(eff_dims)  # [0.1, 0.2, 0.4, 0.8] — WRONG
        new_vel = np.diff(eff_dims) / np.diff(i_vals)  # [0.1, 0.1, 0.1, 0.1] — correct

        assert not np.allclose(old_vel, 0.1), "Old code should be wrong"
        np.testing.assert_allclose(new_vel, 0.1, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
