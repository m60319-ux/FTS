"""
Delta-gravity payload compensation for a flange-mounted FTS.

With the sensor bolted to the flange and the tool on its measurement side,
the reader's blind startup tare (robot stationary, hands off) absorbs

    tare mean = sensor_bias + gravity_wrench(R_tare)

so recovering the operator wrench needs only the CHANGE of the gravity
wrench since the tare:

    w_operator = w_reader − [gravity_wrench(R_now) − gravity_wrench(R_tare)]

The intrinsic sensor bias — and its temperature drift between sessions —
cancels without ever being identified.  Only the payload mass and centre of
mass (from payload_calibration.py) are needed.

Everything here is expressed in the SENSOR frame, upstream of the existing
sensor→TCP→base transform chain in FTS_Free_Drive.py, which stays untouched.

Profile switching (load released/picked up mid-session): the tare reference
is FROZEN with the gravity wrench of the profile active at tare time; a
switch only swaps the (mass, com) used for the current-orientation term, so
the delta formulation stays exact:

    w_raw(t)  = bias + gravity_new(R(t)) + operator
    tare mean = bias + gravity_old(R_tare)
    ⇒ operator = w_reader − [gravity_new(R(t)) − gravity_old(R_tare)]

Split of work: set_tcp_orientation() does one 3×3 matmul + a cross product
and runs at the controller's 50 Hz pose refresh; compensate() is a 6-element
subtraction and runs every 1 ms tick.
"""

from typing import Sequence, Tuple

import numpy as np

from frame_transformer import rotation_matrix_zyx
from payload_model import PayloadParams, gravity_dir_sensor, gravity_wrench


class GravityCompensator:
    """Subtract the orientation-dependent payload gravity delta.

    Parameters
    ----------
    params : PayloadParams
        Active payload profile (mass, com).  Bias fields are ignored —
        the session tare handles bias.
    R_sensor_tcp : np.ndarray, shape (3, 3)
        Rotation mapping sensor-frame vectors into the TCP frame (the same
        matrix FTS_Free_Drive builds from the sensor_offset config).
    g_base : sequence of 3 floats
        Gravity vector in the BASE frame (m/s²).  (0, 0, −9.81) for a
        floor-mounted robot; set accordingly for wall/ceiling mounts.
        Its magnitude is used as the gravity constant.
    """

    def __init__(self, params: PayloadParams, R_sensor_tcp: np.ndarray,
                 g_base: Sequence[float] = (0.0, 0.0, -9.80665)) -> None:
        self._params = params
        self.R_sensor_tcp = np.asarray(R_sensor_tcp, dtype=float)
        self.g_base = np.asarray(g_base, dtype=float)
        self._g_mag = float(np.linalg.norm(self.g_base))
        if self._g_mag < 1e-9:
            raise ValueError("g_base must be a non-zero vector")

        self._g_dir = None          # (3,) unit gravity dir, sensor frame
        self._R_base_sensor = None  # cached for inertial_wrench()
        self._grav = None           # (6,) gravity wrench at current orientation
        self._tare = None           # (6,) gravity wrench frozen at tare

    # ── Orientation refresh (50 Hz) ──────────────────────────────────

    def set_tcp_orientation(self, roll: float, pitch: float, yaw: float) -> None:
        """Update the gravity wrench for a new TCP orientation (radians)."""
        R_tcp_base = np.array(rotation_matrix_zyx(roll, pitch, yaw))
        self._R_base_sensor = R_tcp_base @ self.R_sensor_tcp
        self._g_dir = gravity_dir_sensor(self._R_base_sensor, self.g_base)
        self._recompute_grav()

    def _recompute_grav(self) -> None:
        self._grav = gravity_wrench(
            self._params.mass_kg, self._params.com, self._g_dir, self._g_mag
        )

    # ── Tare ─────────────────────────────────────────────────────────

    def capture_tare(self) -> None:
        """Freeze the current gravity wrench as the tare reference.

        Must be called once, at the orientation the reader's blind tare ran
        at, with the profile that was on the sensor during the tare.
        """
        if self._grav is None:
            raise RuntimeError("capture_tare() before set_tcp_orientation()")
        self._tare = self._grav.copy()

    @property
    def tare_captured(self) -> bool:
        return self._tare is not None

    # ── Profile switch (load released / picked up) ───────────────────

    def set_profile(self, params: PayloadParams) -> None:
        """Swap the active payload profile; the tare reference stays frozen."""
        self._params = params
        if self._g_dir is not None:
            self._recompute_grav()

    @property
    def params(self) -> PayloadParams:
        return self._params

    # ── Per-tick compensation (1 kHz) ────────────────────────────────

    def compensate(self, raw: Sequence[float]) -> np.ndarray:
        """Return the reader wrench minus the gravity delta since the tare."""
        if self._tare is None:
            raise RuntimeError("compensate() before capture_tare()")
        return np.asarray(raw, dtype=float) - (self._grav - self._tare)

    # ── Optional inertial feedforward ────────────────────────────────

    def inertial_wrench(self, a_lin_base: Sequence[float]) -> np.ndarray:
        """Predicted inertial wrench for a linear acceleration (base frame).

        The sensor reads F_op − m·a; adding this back before the low-pass
        filter recovers F_op.  Experimental — the robot lags the commanded
        acceleration, and a wrong phase can worsen chatter.  Keep the
        config flag off unless the slew limiter alone makes a heavy tool
        feel sluggish to start.
        """
        if self._R_base_sensor is None:
            raise RuntimeError("inertial_wrench() before set_tcp_orientation()")
        a_s = self._R_base_sensor.T @ np.asarray(a_lin_base, dtype=float)
        f = self._params.mass_kg * a_s
        m = np.cross(self._params.com, f)
        return np.concatenate((f, m))

    def reset(self) -> None:
        """Drop the tare and orientation state (controller shutdown)."""
        self._g_dir = None
        self._R_base_sensor = None
        self._grav = None
        self._tare = None


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import math

    from payload_model import GRAVITY, predicted_wrench

    rng = np.random.RandomState(5)

    tool = PayloadParams(
        mass_kg=2.4, com=np.array([0.003, -0.012, 0.065]),
        bias_f=np.array([1.2, -0.4, 3.1]), bias_m=np.array([0.05, -0.02, 0.01]),
        gravity=GRAVITY,
    )
    tool_load = tool._replace(mass_kg=4.1, com=np.array([0.001, -0.005, 0.110]))

    # Random sensor mounting and a set of random TCP orientations
    R_sensor_tcp = np.array(rotation_matrix_zyx(0.3, -0.2, 1.1))

    def raw_reading(params, rpy, operator=np.zeros(6)):
        """Simulated raw sensor value: bias + gravity + operator."""
        R_tcp_base = np.array(rotation_matrix_zyx(*rpy))
        g_dir = gravity_dir_sensor(R_tcp_base @ R_sensor_tcp)
        return predicted_wrench(params, g_dir) + operator

    comp = GravityCompensator(tool, R_sensor_tcp)

    # Tare at a random orientation, hands off, tool profile
    rpy_tare = rng.uniform(-math.pi, math.pi, size=3)
    tare_mean = raw_reading(tool, rpy_tare)
    comp.set_tcp_orientation(*rpy_tare)
    comp.capture_tare()

    # 1. Exact model ⇒ compensated residual is zero at any orientation,
    #    and an applied operator wrench is recovered exactly.
    for _ in range(50):
        rpy = rng.uniform(-math.pi, math.pi, size=3)
        op = rng.uniform(-10.0, 10.0, size=6)
        reader = raw_reading(tool, rpy, op) - tare_mean   # blind tare applied
        comp.set_tcp_orientation(*rpy)
        rec = comp.compensate(reader)
        assert np.allclose(rec, op, atol=1e-9), (rpy, rec - op)
    print("exact model: operator wrench recovered at 50 random orientations")

    # 2. Profile switch mid-session: load appears on the tool; the frozen
    #    tare (captured with `tool`) plus the swapped profile stays exact.
    comp.set_profile(tool_load)
    for _ in range(50):
        rpy = rng.uniform(-math.pi, math.pi, size=3)
        op = rng.uniform(-10.0, 10.0, size=6)
        reader = raw_reading(tool_load, rpy, op) - tare_mean
        comp.set_tcp_orientation(*rpy)
        rec = comp.compensate(reader)
        assert np.allclose(rec, op, atol=1e-9), (rpy, rec - op)
    print("profile switch: frozen tare + swapped (m, com) stays exact")

    # 3. Mass error δm bounds the residual by 2·g·δm (force channels).
    dm = 0.05
    comp2 = GravityCompensator(tool._replace(mass_kg=tool.mass_kg + dm),
                               R_sensor_tcp)
    comp2.set_tcp_orientation(*rpy_tare)
    comp2.capture_tare()
    worst = 0.0
    for _ in range(200):
        rpy = rng.uniform(-math.pi, math.pi, size=3)
        reader = raw_reading(tool, rpy) - tare_mean
        comp2.set_tcp_orientation(*rpy)
        worst = max(worst, float(np.linalg.norm(comp2.compensate(reader)[:3])))
    bound = 2.0 * GRAVITY * dm
    assert worst <= bound + 1e-9, (worst, bound)
    print(f"mass error: worst residual {worst:.3f} N <= 2*g*dm = {bound:.3f} N")

    # 4. Guards.
    fresh = GravityCompensator(tool, R_sensor_tcp)
    for call in (fresh.capture_tare, lambda: fresh.compensate(np.zeros(6)),
                 lambda: fresh.inertial_wrench([0.1, 0.0, 0.0])):
        try:
            call()
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError before orientation/tare")
    print("guards OK\nself-test OK")
