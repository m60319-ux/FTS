"""
Low-pass filter for 6-DOF force/torque wrenches.

Runs at the control-loop rate (1 kHz) on the compensated wrench, BEFORE the
dead-zone, so that sensor quantisation noise (±0.1 N / ±0.1 Nm LSB) cannot
chatter across the dead-zone edge and the 50 Hz staircase of the gravity-
compensation term is smoothed out.

Filter types
------------
"butter2"  2nd-order Butterworth (bilinear transform, prewarped).  −12 dB/oct
           attenuates the 15–100 Hz sensor/tool-resonance band that closes the
           m·a feedback loop, at ~22 ms group delay for the 10 Hz default —
           imperceptible in hand guiding.  Recommended for flange mode.
"ema"      One-pole exponential moving average (−6 dB/oct).
"none"     Pass-through.

The loop period is fixed (CONTROL_PERIOD), so coefficients are designed once
at construction; there is no per-tick redesign.  The state is primed to the
steady-state of the first input after a reset, so there is no startup
transient (important: an un-primed filter would ramp the gravity term from
zero and read as a phantom operator push).
"""

import math
from typing import Sequence

import numpy as np


FILTER_TYPES = ("butter2", "ema", "none")


class WrenchLowPass:
    """6-channel low-pass filter, vectorised over the wrench.

    Parameters
    ----------
    filter_type : str
        One of "butter2", "ema", "none".
    cutoff_hz : float
        −3 dB cutoff frequency.  Must satisfy 0 < cutoff_hz < 0.45/dt
        (comfortably below Nyquist).  Ignored for "none".
    dt : float
        Sample period in seconds (the control-loop period).
    """

    def __init__(self, filter_type: str, cutoff_hz: float, dt: float) -> None:
        if filter_type not in FILTER_TYPES:
            raise ValueError(
                f"filter_type must be one of {FILTER_TYPES}, got '{filter_type}'"
            )
        self.filter_type = filter_type
        self.cutoff_hz = float(cutoff_hz)
        self.dt = float(dt)

        if filter_type != "none":
            if dt <= 0.0:
                raise ValueError(f"dt must be positive, got {dt}")
            if not 0.0 < self.cutoff_hz < 0.45 / dt:
                raise ValueError(
                    f"cutoff_hz must be in (0, {0.45 / dt:.1f}) for dt={dt}, "
                    f"got {self.cutoff_hz}"
                )

        if filter_type == "butter2":
            # Bilinear transform with frequency prewarping.
            K = math.tan(math.pi * self.cutoff_hz * self.dt)
            norm = 1.0 / (1.0 + math.sqrt(2.0) * K + K * K)
            self._b0 = K * K * norm
            self._b1 = 2.0 * self._b0
            self._b2 = self._b0
            self._a1 = 2.0 * (K * K - 1.0) * norm
            self._a2 = (1.0 - math.sqrt(2.0) * K + K * K) * norm
        elif filter_type == "ema":
            self._alpha = 1.0 - math.exp(-2.0 * math.pi * self.cutoff_hz * self.dt)

        # Filter state — None means "prime on next sample".
        self._z1 = None
        self._z2 = None
        self._y = None

    def filter(self, w: Sequence[float]) -> np.ndarray:
        """Filter one wrench sample (6,); returns the filtered wrench."""
        w = np.asarray(w, dtype=float)

        if self.filter_type == "none":
            return w

        if self.filter_type == "ema":
            if self._y is None:
                self._y = w.copy()
            else:
                self._y = self._y + self._alpha * (w - self._y)
            return self._y.copy()

        # butter2 — direct form II transposed, vectorised over the 6 channels
        if self._z1 is None:
            # Prime to the steady state of a constant input w (DC gain is 1,
            # so y == w): z1 = (1−b0)·w, z2 = (b2−a2)·w.
            self._z1 = (1.0 - self._b0) * w
            self._z2 = (self._b2 - self._a2) * w
        y = self._b0 * w + self._z1
        self._z1 = self._b1 * w - self._a1 * y + self._z2
        self._z2 = self._b2 * w - self._a2 * y
        return y

    def reset(self) -> None:
        """Clear state; the next sample re-primes the filter."""
        self._z1 = None
        self._z2 = None
        self._y = None


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    dt = 0.001

    for ftype in ("butter2", "ema"):
        f = WrenchLowPass(ftype, cutoff_hz=10.0, dt=dt)

        # 1. Priming: first output equals first input exactly.
        first = f.filter(np.array([3.0, -1.0, 2.0, 0.1, -0.2, 0.3]))
        assert np.allclose(first, [3.0, -1.0, 2.0, 0.1, -0.2, 0.3]), ftype

        # 2. DC gain = 1: constant input stays constant.
        for _ in range(2000):
            y = f.filter(np.array([3.0, -1.0, 2.0, 0.1, -0.2, 0.3]))
        assert np.allclose(y, [3.0, -1.0, 2.0, 0.1, -0.2, 0.3], atol=1e-9), ftype

        # 3. Step response converges to the step value.
        f.reset()
        f.filter(np.zeros(6))
        for _ in range(1000):
            y = f.filter(np.ones(6))
        assert np.allclose(y, 1.0, atol=1e-6), ftype

        # 4. Gain at the cutoff frequency ≈ −3 dB (0.707).
        f.reset()
        n = 20000
        t = np.arange(n) * dt
        x = np.sin(2.0 * math.pi * 10.0 * t)
        out = np.empty(n)
        for i in range(n):
            out[i] = f.filter(np.full(6, x[i]))[0]
        # Steady-state amplitude over the last half of the run
        amp = out[n // 2:].max()
        assert abs(amp - 1.0 / math.sqrt(2.0)) < 0.02, (ftype, amp)

        print(f"{ftype:8s}: priming OK, DC gain OK, step OK, |H(fc)| = {amp:.3f}")

    # 5. "none" passes through.
    f = WrenchLowPass("none", cutoff_hz=0.0, dt=dt)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(f.filter(x), x)
    print("none    : pass-through OK")

    # 6. Validation errors.
    for bad in ({"filter_type": "bogus", "cutoff_hz": 10.0, "dt": dt},
                {"filter_type": "butter2", "cutoff_hz": 0.0, "dt": dt},
                {"filter_type": "butter2", "cutoff_hz": 500.0, "dt": dt}):
        try:
            WrenchLowPass(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad}")
    print("validation OK\nself-test OK")
