"""
Convert force/torque sensor readings to jog velocities [vx, vy, vz, ωroll, ωpitch, ωyaw].

Proportional mapping with dead-zone filtering and saturation clamping to ensure
the robot never exceeds configurable speed limits.
"""

import math
import threading
from typing import Optional, Tuple

# Type alias for the 6-DOF velocity tuple
Velocity6D = Tuple[float, float, float, float, float, float]


class ForceToVelocity:
    """Map force/torque readings to clamped jog velocities.

    Parameters
    ----------
    max_linear_vel : float
        Hard cap on each translational axis (m/s).  Default 1.0.
    max_angular_vel : float
        Hard cap on each rotational axis (rad/s).  Default 1.0.
    dead_zone_force : float
        Forces below this magnitude produce zero velocity (N).  Default 0.5.
    dead_zone_torque : float
        Torques below this magnitude produce zero velocity (Nm). Default 0.1.
    gain_force : float
        Linear gain  (m/s per N).   With default 0.02 → 50 N = 1.0 m/s.
    gain_torque : float
        Angular gain (rad/s per Nm). With default 0.1  → 10 Nm = 1.0 rad/s.
    clamp_norm : bool
        If True, after per-axis clamping also clamp the Euclidean norm of
        the linear part to *max_linear_vel* and angular part to
        *max_angular_vel*.  Default False (per-axis only).
    smoothing_alpha : float | None
        If set to a value in (0, 1], apply an exponential moving average
        (low-pass filter) to the output velocity.  A smaller value means
        heavier smoothing.  None disables smoothing.

        The value is interpreted as the blend factor for one update of
        length *smoothing_ref_period*.  When :meth:`update` is called with
        an explicit *dt* the factor is rescaled so the filter's time
        constant — and therefore its feel — is independent of the control
        loop rate.
    smoothing_ref_period : float
        Update period (seconds) at which *smoothing_alpha* is the literal
        blend factor.  Default 0.11, the period this app's control loop
        used before it moved to 1 kHz, so existing tuning keeps its meaning.
    instant_stop : bool
        If True (default), whenever the dead-zone has zeroed every raw axis
        the output velocity snaps to exactly zero instead of decaying
        through the filter.  The operator has released the handle; the robot
        must stop now, not asymptotically.  Smoothing then shapes only the
        rise, never the fall.
    """

    def __init__(
        self,
        max_linear_vel: float = 1.0,
        max_angular_vel: float = 1.0,
        dead_zone_force: float = 0.5,
        dead_zone_torque: float = 0.1,
        gain_force: float = 0.02,
        gain_torque: float = 0.1,
        clamp_norm: bool = False,
        smoothing_alpha: Optional[float] = None,
        smoothing_ref_period: float = 0.11,
        instant_stop: bool = True,
    ) -> None:
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.dead_zone_force = dead_zone_force
        self.dead_zone_torque = dead_zone_torque
        self.gain_force = gain_force
        self.gain_torque = gain_torque
        self.clamp_norm = clamp_norm
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_ref_period = smoothing_ref_period
        self.instant_stop = instant_stop

        # Time constant of the EMA, derived so that one update of length
        # smoothing_ref_period blends by exactly smoothing_alpha.
        # alpha >= 1 means "no smoothing", which has no finite time constant.
        self._tau: Optional[float] = None
        if (
            smoothing_alpha is not None
            and 0.0 < smoothing_alpha < 1.0
            and smoothing_ref_period > 0.0
        ):
            self._tau = -smoothing_ref_period / math.log(1.0 - smoothing_alpha)

        self._lock = threading.Lock()
        self._velocity: Velocity6D = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        fx: float,
        fy: float,
        fz: float,
        mx: float,
        my: float,
        mz: float,
        dt: Optional[float] = None,
    ) -> None:
        """Accept new force/torque reading and compute velocity.

        This method signature matches :pydata:`FTCallback` so it can be
        passed directly as the *callback* argument to
        :class:`FTSSensorReader`.

        *dt* is the elapsed time since the previous update (seconds).  Pass
        it from a fixed-rate control loop so the smoothing time constant is
        honoured regardless of the loop period.  When omitted,
        *smoothing_alpha* is applied verbatim as a per-call blend factor.
        """
        vx = self._apply_axis(fx, self.dead_zone_force, self.gain_force, self.max_linear_vel)
        vy = self._apply_axis(fy, self.dead_zone_force, self.gain_force, self.max_linear_vel)
        vz = self._apply_axis(fz, self.dead_zone_force, self.gain_force, self.max_linear_vel)
        vroll = self._apply_axis(mx, self.dead_zone_torque, self.gain_torque, self.max_angular_vel)
        vpitch = self._apply_axis(my, self.dead_zone_torque, self.gain_torque, self.max_angular_vel)
        vyaw = self._apply_axis(mz, self.dead_zone_torque, self.gain_torque, self.max_angular_vel)

        if self.clamp_norm:
            vx, vy, vz = self._clamp_vector(vx, vy, vz, self.max_linear_vel)
            vroll, vpitch, vyaw = self._clamp_vector(vroll, vpitch, vyaw, self.max_angular_vel)

        new_vel: Velocity6D = (vx, vy, vz, vroll, vpitch, vyaw)

        # Every axis inside its dead zone: the handle has been released.
        released = not any(new_vel)

        with self._lock:
            if released and self.instant_stop:
                self._velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                return

            a = self._blend_factor(dt)
            if a < 1.0:
                prev = self._velocity
                new_vel = tuple(  # type: ignore[assignment]
                    a * n + (1.0 - a) * p for n, p in zip(new_vel, prev)
                )
            self._velocity = new_vel  # type: ignore[assignment]

    def get_velocity(self) -> Velocity6D:
        """Return the latest computed velocity (thread-safe)."""
        with self._lock:
            return self._velocity

    def reset(self) -> None:
        """Reset stored velocity to zero."""
        with self._lock:
            self._velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blend_factor(self, dt: Optional[float]) -> float:
        """EMA blend factor for an update spanning *dt* seconds.

        Returns 1.0 (pass the new value straight through) whenever smoothing
        is disabled or degenerate.
        """
        if self.smoothing_alpha is None:
            return 1.0
        if dt is None:
            return min(1.0, max(0.0, self.smoothing_alpha))
        if self._tau is None:
            # alpha outside (0, 1) — nothing to interpolate.
            return 1.0
        if dt <= 0.0:
            return 0.0
        return 1.0 - math.exp(-dt / self._tau)

    @staticmethod
    def _apply_axis(value: float, dead_zone: float, gain: float, limit: float) -> float:
        """Dead-zone → scale → saturate for a single axis."""
        if abs(value) < dead_zone:
            return 0.0
        return max(-limit, min(limit, value * gain))

    @staticmethod
    def _clamp_vector(x: float, y: float, z: float, limit: float) -> Tuple[float, float, float]:
        """Scale a 3-vector so its Euclidean norm does not exceed *limit*."""
        norm = math.sqrt(x * x + y * y + z * z)
        if norm > limit and norm > 0.0:
            scale = limit / norm
            return (x * scale, y * scale, z * scale)
        return (x, y, z)


class VelocitySlewLimiter:
    """Per-axis acceleration clamp on a commanded 6-DOF velocity.

    With a tool mass on the sensor, commanded acceleration couples straight
    back into the force reading as an inertial term m·a.  Clamping the
    commanded acceleration caps that term at m·a_max regardless of gains or
    filtering; sizing m·a_max <= dead_zone_force/3 guarantees the inertial
    term alone can never push the reading across the dead-zone, so no
    self-sustained limit cycle is possible.

    Deceleration (a step that moves the velocity toward zero) is allowed at
    *decel_multiplier* times the acceleration rate, so release-to-stop stays
    fast while starts are ramped gently.  A sign change decelerates to zero
    first, then accelerates into the new direction with whatever remains of
    the tick.

    Parameters
    ----------
    max_accel_linear : float
        Acceleration limit for axes 0-2 (m/s²).
    max_accel_angular : float
        Acceleration limit for axes 3-5 (rad/s²).
    decel_multiplier : float
        Deceleration rate = acceleration limit × this factor (>= 1).
    """

    def __init__(
        self,
        max_accel_linear: float,
        max_accel_angular: float,
        decel_multiplier: float = 4.0,
    ) -> None:
        if max_accel_linear <= 0.0 or max_accel_angular <= 0.0:
            raise ValueError("acceleration limits must be positive")
        if decel_multiplier < 1.0:
            raise ValueError(f"decel_multiplier must be >= 1, got {decel_multiplier}")
        self.max_accel_linear = max_accel_linear
        self.max_accel_angular = max_accel_angular
        self.decel_multiplier = decel_multiplier
        self._prev = [0.0] * 6

    def limit(self, velocity, dt: float) -> list:
        """Rate-limit one velocity command; returns the ramped 6-list."""
        out = []
        for i, target in enumerate(velocity):
            accel = self.max_accel_linear if i < 3 else self.max_accel_angular
            out.append(self._step(self._prev[i], float(target), accel,
                                  accel * self.decel_multiplier, dt))
        self._prev = list(out)
        return out

    def reset(self) -> None:
        """Forget the previous command (controller shutdown/startup)."""
        self._prev = [0.0] * 6

    @staticmethod
    def _step(prev: float, target: float, accel: float, decel: float,
              dt: float) -> float:
        """Advance one axis from *prev* toward *target* over *dt* seconds."""
        if prev * target < 0.0:
            # Sign change: decelerate to zero first, then accelerate into
            # the new direction with the remaining fraction of the tick —
            # the new-direction ramp must not inherit the decel rate.
            t_zero = abs(prev) / decel
            if t_zero >= dt:
                return prev - math.copysign(decel * dt, prev)
            step = min(abs(target), accel * (dt - t_zero))
            return math.copysign(step, target)

        # Same side of zero (or starting/stopping at zero)
        rate = accel if abs(target) >= abs(prev) else decel
        delta = target - prev
        max_step = rate * dt
        if delta > max_step:
            return prev + max_step
        if delta < -max_step:
            return prev - max_step
        return target
