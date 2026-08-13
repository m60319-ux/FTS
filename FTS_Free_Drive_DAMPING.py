"""
FTS M4313SFA9A hand-guiding controller for Neura robots.

Adapted from fts_hand_guid_ctrl.py.

Lifecycle: __init__(robot, config_path) → init() → run() → finish()

Every tunable value is read from the JSON file at config_path; see
FTS_Free_DriveConfig.py for the schema and FTS_Free_Drive_config.json for the
defaults.

Two sensor mountings are supported, selected by the optional `payload`
config section:

HANDLE mode (legacy, payload section absent or enabled=false)
  The sensor hangs on a handle — no static mass on its measurement side, so
  the reader's blind startup tare removes everything static and readings are
  purely the operator wrench.

FLANGE mode (payload.enabled == true)
  The sensor is bolted to the flange with the tool on its measurement side.
  The blind tare then absorbs `bias + gravity(R_tare)`, and the loop
  additionally subtracts the delta-gravity term
  `gravity(R_now) − gravity(R_tare)` in the sensor frame
  (GravityCompensator), so the operator wrench is isolated at any
  orientation.  Payload mass/COM come from payload_calibration.py; two
  profiles (tool / tool+load) can be swapped mid-session via a digital
  input.  Commanded acceleration is slew-limited (VelocitySlewLimiter) so
  the tool's m·a inertial term cannot chatter across the dead-zone.

Pipeline:
  FTS sensor (sensor frame)
    → GravityCompensator.compensate() — flange mode: delta-gravity subtract
    → WrenchLowPass.filter()          — optional low-pass (both modes)
    → Sensor→TCP transform (rotation + lever-arm correction)
    → FrameTransformer.transform()   — rotate TCP→Base
    → ForceToVelocity.update()       — dead-zone, gain, clamp
    → ContactDamping                 — continuous force/reversal damping
    → Optionally zero rotation       — when orientation_control is False
    → VelocitySlewLimiter.limit()    — flange mode: acceleration clamp
    → speed_x() or speed_j()        — depending on control_space
    → Robot moves

The control loop runs at CONTROL_PERIOD (1 kHz) on a perf_counter deadline
and issues a velocity command on *every* tick — the commanded velocity when
the operator is pushing, zeros otherwise.  speed_j/speed_x have no controller
-side watchdog, so a commanded velocity persists until the next command; the
robot may only ever be one tick away from a fresh zero.

Stop mechanisms, all converging on _stop_event → _shutdown():
  • the digital input named by param.io_num reads high (polled every 50 ms)
  • that input (or the profile-switch input) cannot be read at all (fail-safe)
  • sensor data goes stale (> MAX_SENSOR_AGE) or the reader thread dies
  • flange mode: the robot moved during the tare window, the post-tare
    residual check failed, the TCP pose went stale, the compensated wrench
    is implausibly large, or a profile switch failed to verify
  • finish() is called by the caller
"""

import math
import sys
import time
import threading
import traceback
from pathlib import Path
from typing import Tuple

import numpy as np

from FTS_Free_DriveConfig import FTS_Free_DriveConfig, load_config

from M4313SFA9A_helper import FTSSensorReader, resolve_serial_port
from force_to_velocity import ForceToVelocity, VelocitySlewLimiter
from frame_transformer import (
    FrameTransformer,
    build_sensor_to_tcp_transform,
    resolve_sensor_offset,
    rotation_matrix_zyx,
)
from gravity_compensator import GravityCompensator
from payload_model import params_from_profile, sensor_correction_matrix
from robot_kinematics import RobotKinematics
from wrench_filter import WrenchLowPass

class FTS_Free_Drive:
    """FTS hand-guiding controller."""

    # ── Safety constants (not exposed in the config yet) ─────────────
    DAMPING = 0.035                # Damped pseudo-inverse λ
    MAX_JOINT_VEL = 1.0            # rad/s — proportional clamp
    JOINT_LIMIT_BUFFER_DEG = 10.0  # braking starts 10° from limit
    JOINT_LIMIT_HARD_STOP_DEG = 3.0  # velocity → 0 at 3° from limit

    # ── Timing constants (seconds) ───────────────────────────────────
    CONTROL_PERIOD = 0.001      # 1 kHz — matches servo_j's default cycle_time
    IO_POLL_INTERVAL = 0.05     # stop-input poll period
    MAX_SENSOR_AGE = 0.040      # ~40 missed frames at the sensor's 2 kHz #0.02
    POSE_REFRESH = 0.020        # 50 Hz  — get_tcp_pose()
    Q_REFRESH = 0.010           # 100 Hz — get_current_joint_angles()
    # ── FREE_DRIVE → HOLD → EXIT constants ─────────────────────────
    # ONE-DI AUTO-EXIT behavior:
    # first rising edge leaves FREE_DRIVE, captures the current TCP pose,
    # commands that SAME pose 10 times at 100 ms intervals, then proceeds
    # directly to EXIT.  No DI release or second press is required.
    POSITION_HOLD_REPEAT = 1            # same captured pose, exactly 10 commands
    POSITION_HOLD_INTERVAL = 0.100       # 100 ms between position-hold commands
    POSITION_HOLD_GAIN = 10.0            # ServoX proportional gain
    POSITION_HOLD_STATUS_INTERVAL = 1.0  # console status rate limit

    # ── Flange-mode constants (seconds / radians) ────────────────────
    POSE_MAX_AGE = 0.1          # 5× POSE_REFRESH — gravity model needs a live pose
    TARE_VERIFY_SECONDS = 0.5   # post-tare residual window, zeros commanded
    TARE_ORIENT_TOL_RAD = math.radians(0.5)  # motion tolerance during tare
    PROFILE_VERIFY_HOLD = 0.2   # residual must stay low this long after a switch
    PROFILE_VERIFY_TIMEOUT = 5.0  # give up on a profile switch after this
    DRIFT_EMA_TAU = 10.0        # slow EMA on the idle compensated wrench
    DRIFT_WARN_FRACTION = 0.8   # warn when the EMA reaches this × dead-zone
    DRIFT_WARN_INTERVAL = 30.0  # rate limit for the drift warning

    # ── Contact damping constants ─────────────────────────────────────
    # No contact damping state machine is used.  FREE_DRIVE remains active at
    # all times.  Large contact force continuously reduces the commanded
    # linear velocity, and an attempted instantaneous direction reversal is
    # damped much more strongly.  Because nothing is latched, the operator can
    # always pull the robot away from the contact surface.
    CONTACT_DAMPING_ENABLED = True
    CONTACT_DAMPING_START_N = 8.0        # no extra damping below this force
    CONTACT_DAMPING_FULL_N = 30.0        # full damping reached at this force
    CONTACT_DAMPING_MIN_SCALE = 0.15     # high-force speed keeps 20% of target
    CONTACT_DAMPING_REVERSE_SCALE = 0.1 # extra scale on sudden reversal
    CONTACT_DAMPING_REVERSE_MIN_SPEED = 0.003  # 3 mm/s previous motion needed

    # Console force display interval.  This does NOT slow the control loop;
    # sensor/control updates remain at the normal loop rate.
    FORCE_PRINT_INTERVAL = 0.100        # print force every 100 ms

    def __init__(self, robot, config_path, log_path=None):
        self.robot = robot
        self.config_path = config_path
        self._log_path = log_path
        self._tick_log = None
        self._stop_event = threading.Event()
        self._reader = None
        self._converter = None
        self._frame_tf = None
        self._kin = None
        self._joint_limits = None

        self._io_thread = None
        self._stop_reason = None
        self._servo_active = False
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        self._q = None              # cached joint angles, refreshed at Q_REFRESH

        # Last compensated force seen by the active control loop. Shutdown
        # also polls the live FTS, but this gives a useful first-tick fallback.
        self._last_force_norm = 0.0
        self._last_force_t = -math.inf

        # Loop-exit handshake: _shutdown() called from another thread (e.g.
        # finish()) must wait for the control loop's last command before
        # streaming its zeros, or a stray in-flight non-zero command could
        # land AFTER them and stand with no watchdog.  Set by default so
        # shutdown never blocks when the loop never ran.
        self._loop_exited = threading.Event()
        self._loop_exited.set()
        self._loop_thread = None

        # ── Flange-mode state ────────────────────────────────────────
        self.flange_mode = False
        self._compensator = None        # GravityCompensator
        self._wrench_filter = None      # WrenchLowPass (both modes, optional)
        self._slew = None               # VelocitySlewLimiter
        self._profiles = {}             # name → PayloadParams
        self._active_profile = None     # name of the profile on the compensator
        self._switch_request = None     # name requested by the switch input
        self._switch_io = -1

    # ─────────────────────────────────────────────────────────────────
    # init — called once; reads every parameter from the JSON config
    # ─────────────────────────────────────────────────────────────────

    def init(self) -> None:
        self.config: FTS_Free_DriveConfig = load_config(self.config_path)

        # ── Control config ───────────────────────────────────────────
        self.use_joint_space = bool(self.config.control.control_space)
        self.enable_orientation = bool(self.config.control.orientation_control)

        # ── Sensor-to-TCP offset ─────────────────────────────────────
        # Either convention: legacy sensor_offset, or flange-based
        # sensor_mount + tool_tcp (derived by resolve_sensor_offset).
        self.R_sensor_tcp, self.p_sensor_tcp = build_sensor_to_tcp_transform(
            *resolve_sensor_offset(self.config)
        )
        self.has_lever_arm = np.linalg.norm(self.p_sensor_tcp) > 1e-9
        # Moment transport needs the sensor→TCP offset in the TCP frame.
        # p_sensor_tcp is measured in the SENSOR frame (see the docstring
        # of build_sensor_to_tcp_transform), so rotate it once here.
        self.p_lever_tcp = self.R_sensor_tcp @ self.p_sensor_tcp

        # ── Force → velocity converter ───────────────────────────────
        max_linear_vel = float(self.config.force_to_velocity.max_linear_vel)
        max_angular_vel = float(self.config.force_to_velocity.max_angular_vel)
        dead_zone_force = float(self.config.force_to_velocity.dead_zone_force)
        dead_zone_torque = float(self.config.force_to_velocity.dead_zone_torque)
        gain_force = float(self.config.force_to_velocity.gain_force)
        gain_torque = float(self.config.force_to_velocity.gain_torque)

        # When orientation control is disabled, override torque params
        if not self.enable_orientation:
            max_angular_vel = 0.0
            dead_zone_torque = 999.0
            gain_torque = 0.0

        # Optional output-velocity smoothing (config force_to_velocity.
        # smoothing_tau).  alpha = 1 − e⁻¹ at ref_period = tau makes the
        # converter's EMA time-constant exactly tau; update() receives the
        # REAL tick dt, so the feel is loop-rate independent.  instant_stop
        # (default) still snaps to zero on release.
        smoothing_tau = float(getattr(self.config.force_to_velocity,
                                      "smoothing_tau", 0.0))
        self._converter = ForceToVelocity(
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            dead_zone_force=dead_zone_force,
            dead_zone_torque=dead_zone_torque,
            gain_force=gain_force,
            gain_torque=gain_torque,
            smoothing_alpha=(1.0 - math.exp(-1.0)) if smoothing_tau > 0.0
            else None,
            smoothing_ref_period=smoothing_tau if smoothing_tau > 0.0
            else 0.11,
        )

        # ── Frame transformer (TCP → Base rotation) ──────────────────
        self._frame_tf = FrameTransformer()

        # ── Payload compensation (flange mode) ───────────────────────
        pl = self.config.payload
        self.flange_mode = pl is not None and bool(pl.enabled)
        if self.flange_mode:
            g_base = np.array([
                float(pl.gravity_x), float(pl.gravity_y), float(pl.gravity_z)
            ])
            g_mag = float(np.linalg.norm(g_base))
            if abs(g_mag - 9.80665) > 0.05 * 9.80665:
                raise ValueError(
                    f"payload.gravity_* has magnitude {g_mag:.3f} m/s², "
                    "expected within 5% of 9.81"
                )

            # Only calibrated profiles are usable.
            self._profiles = {}
            for name, prof in (("tool", pl.profile_tool),
                               ("tool_load", pl.profile_tool_load)):
                if str(prof.identified_on) and float(prof.mass_kg) > 0.0:
                    self._profiles[name] = params_from_profile(prof, gravity=g_mag)

            self._switch_io = int(float(pl.switch_io))
            if self._switch_io >= 0:
                missing = [n for n in ("tool", "tool_load")
                           if n not in self._profiles]
                if missing:
                    raise ValueError(
                        "payload.switch_io is set but profile(s) not calibrated: "
                        f"{', '.join(missing)} — run payload_calibration.py"
                    )

            active = str(pl.active_profile)
            if active not in ("tool", "tool_load"):
                raise ValueError(
                    f"payload.active_profile must be 'tool' or 'tool_load', "
                    f"got '{active}'"
                )
            if active not in self._profiles:
                raise ValueError(
                    f"payload.active_profile '{active}' is not calibrated — "
                    "run payload_calibration.py"
                )

            for field in ("startup_max_residual_force", "startup_max_residual_torque",
                          "max_operator_force", "max_operator_torque",
                          "max_accel_linear", "max_accel_angular"):
                if float(getattr(pl, field)) <= 0.0:
                    raise ValueError(f"payload.{field} must be positive")

            self._startup_max_res_f = float(pl.startup_max_residual_force)
            self._startup_max_res_m = float(pl.startup_max_residual_torque)
            self._max_op_f = float(pl.max_operator_force)
            self._max_op_m = float(pl.max_operator_torque)
            self._inertial_ff = bool(pl.inertial_feedforward)
            policy = str(getattr(pl, "overforce_policy", "stop"))
            if policy not in ("stop", "saturate"):
                raise ValueError(
                    f"payload.overforce_policy must be 'stop' or 'saturate', "
                    f"got '{policy}'"
                )
            self._overforce_saturate = policy == "saturate"
            self._hard_stop_multiplier = float(
                getattr(pl, "hard_stop_multiplier", 5.0)
            )
            if self._hard_stop_multiplier < 1.0:
                raise ValueError("payload.hard_stop_multiplier must be >= 1.0")
            # -inf: perf_counter()'s epoch is arbitrary (≈0 at process start
            # on Windows), so 0.0 would suppress the first warning for a
            # whole rate-limit interval.
            self._last_overforce_warn = -math.inf

            self._active_profile = active
            self._switch_request = active
            self._compensator = GravityCompensator(
                self._profiles[active], self.R_sensor_tcp, g_base
            )
            self._slew = VelocitySlewLimiter(
                float(pl.max_accel_linear),
                float(pl.max_accel_angular),
                float(pl.decel_multiplier),
            )

            # Dead-zone floor advisory — the compensation residual plus the
            # commanded m·a inertial term must fit inside the dead-zone or
            # the robot can drift / limit-cycle.  See _deadzone_floors().
            worst_m = max(p.mass_kg for p in self._profiles.values())
            worst_com = max(float(np.linalg.norm(p.com))
                            for p in self._profiles.values())
            floor_f, floor_m = _deadzone_floors(
                worst_m, worst_com, float(pl.max_accel_linear),
                self.enable_orientation, g_mag,
            )
            if dead_zone_force < floor_f:
                self._info(
                    f"WARNING: dead_zone_force {dead_zone_force:.2f} N is below "
                    f"the recommended flange-mode floor {floor_f:.2f} N "
                    f"(m={worst_m:.2f} kg) — the robot may drift or chatter"
                )
            if self.enable_orientation and dead_zone_torque < floor_m:
                self._info(
                    f"WARNING: dead_zone_torque {dead_zone_torque:.2f} Nm is below "
                    f"the recommended flange-mode floor {floor_m:.2f} Nm"
                )

        # Dead-zones are needed again by the drift monitor in run().
        self._dead_zone_force = dead_zone_force
        self._dead_zone_torque = dead_zone_torque

        # ── Wrench low-pass filter (both modes, optional) ────────────
        wf = self.config.wrench_filter
        if wf is not None and str(wf.filter_type) != "none":
            self._wrench_filter = WrenchLowPass(
                str(wf.filter_type), float(wf.cutoff_hz), self.CONTROL_PERIOD
            )

        # ── Sensor crosstalk correction (both modes, optional) ───────
        # Same 6×6 K the calibration applied to its captures, so the
        # payload profile and the runtime readings describe the same
        # corrected sensor.  Linear, so applying it to the reader's
        # already-tared output is equivalent to correcting before the tare.
        self._sensor_K = None
        sc = self.config.sensor_correction
        if sc is not None:
            self._sensor_K = sensor_correction_matrix(
                sc.moment_to_force, sc.force_to_moment
            )

        # ── Robot type from robot object ─────────────────────────────
        # robot.robot_name returns e.g. "ds6-800"; DH registry expects "DS6-800"
        self._robot_type = self.robot.robot_name.upper()

        # ── Robot kinematics (for joint-space mode) ──────────────────
        self._buffer_rad = math.radians(self.JOINT_LIMIT_BUFFER_DEG)
        self._hard_stop_rad = math.radians(self.JOINT_LIMIT_HARD_STOP_DEG)
        if self.use_joint_space:
            self._kin = RobotKinematics(self._robot_type)
            self._joint_limits = self._kin.joint_limits

        # ── Sensor reader ────────────────────────────────────────────
        port = resolve_serial_port(self.config.param.serial_port)
        auto_zero_seconds = float(self.config.param.auto_zero_seconds)

        self._reader = FTSSensorReader(
            port=port,
            callback=lambda *args: None,  # we poll via get_latest()
            auto_zero_seconds=auto_zero_seconds,
        )

        # ── Stop input ───────────────────────────────────────────────
        # get_digital_input() takes an int pin number; tolerate a config that
        # carries it as a float or a quoted number.
        self.io_num = int(float(self.config.param.io_num))
        if self.io_num < 0:
            raise ValueError(
                f"param.io_num must be a non-negative pin number, got {self.io_num}"
            )
        if self.flange_mode and self._switch_io == self.io_num:
            raise ValueError(
                f"payload.switch_io ({self._switch_io}) collides with the stop "
                f"input param.io_num ({self.io_num})"
            )

        mode = "flange" if self.flange_mode else "handle"
        extra = ""
        if self.flange_mode:
            p = self._profiles[self._active_profile]
            extra = (f", profile: {self._active_profile} "
                     f"(m={p.mass_kg:.2f} kg, |com|={np.linalg.norm(p.com) * 1e3:.0f} mm)"
                     + (f", switch input: DI {self._switch_io}" if self._switch_io >= 0 else ""))
        if self._sensor_K is not None:
            extra += ", crosstalk correction: on"
        self._info(f"FTS App initialized — robot: {self._robot_type}, "
              f"mounting: {mode}, "
              f"joint-space: {self.use_joint_space}, "
              f"orientation: {self.enable_orientation}, "
              f"stop input: DI {self.io_num}{extra}")

    # ─────────────────────────────────────────────────────────────────
    # run — main control loop
    # ─────────────────────────────────────────────────────────────────

    def _apply_contact_damping(
        self, velocity, force_norm: float, prev_cmd_vel
    ):
        """Continuously damp contact-induced velocity without latching a mode.

        Force damping progressively reduces linear speed as contact force rises.
        Reversal damping adds a stronger reduction when the new FREE_DRIVE target
        points opposite to the previous commanded motion — the typical rebound
        condition after a rigid contact.  There is no STOP/BLOCKED state.

        Returns (damped_velocity, total_scale, reversing).
        """
        out = list(velocity)
        if not self.CONTACT_DAMPING_ENABLED:
            return out, 1.0, False

        f0 = float(self.CONTACT_DAMPING_START_N)
        f1 = float(self.CONTACT_DAMPING_FULL_N)
        min_scale = float(self.CONTACT_DAMPING_MIN_SCALE)

        if f1 <= f0:
            force_scale = min_scale if force_norm > f0 else 1.0
        elif force_norm <= f0:
            force_scale = 1.0
        elif force_norm >= f1:
            force_scale = min_scale
        else:
            ratio = (force_norm - f0) / (f1 - f0)
            force_scale = 1.0 - ratio * (1.0 - min_scale)

        linear = np.asarray(out[:3], dtype=float)
        prev = np.asarray(prev_cmd_vel[:3], dtype=float)
        linear_norm = float(np.linalg.norm(linear))
        prev_norm = float(np.linalg.norm(prev))

        reversing = (
            linear_norm > 1e-9
            and prev_norm >= self.CONTACT_DAMPING_REVERSE_MIN_SPEED
            and float(np.dot(linear, prev)) < 0.0
        )

        total_scale = force_scale
        if reversing:
            total_scale *= float(self.CONTACT_DAMPING_REVERSE_SCALE)

        linear *= total_scale
        out[0] = float(linear[0])
        out[1] = float(linear[1])
        out[2] = float(linear[2])
        return out, total_scale, reversing

    def run(self) -> None:
        # ── Tick logger (--log) ──────────────────────────────────────
        # Created before anything is armed so a bad path fails here, not
        # after collision detection is already off.
        if self._log_path:
            self._tick_log = _TickLogger(self._log_path)
            self._info(f"Tick logging to {self._tick_log.path}")

        # ── Robot initialization ─────────────────────────────────────
        # self.robot.set_override(1)
        # self.robot.reset_errors()
        # self.robot.power_on()
        # time.sleep(1)

        # Probe the stop input before anything can move.  The watcher thread
        # fails safe on a read error, so an unreadable pin must surface here
        # as a config error rather than as an instant, unexplained stop.
        try:
            already_high = bool(int(self.robot.get_digital_input(self.io_num)))
        except Exception as ex:
            raise RuntimeError(
                f"Cannot read stop input DI {self.io_num} "
                f"(param.io_num): {ex}"
            ) from ex

        # Refuse to arm against a stop input that is already asserted, rather
        # than starting the loop and racing the watcher thread for a tick.
        if already_high:
            self._info(f"Stop input DI {self.io_num} is already HIGH — not starting control")
            self._stop_reason = "io_high_at_startup"
            self._shutdown()
            return

        self.robot.disable_collision_detection()
        self.robot.activate_servo_interface("position")
        self._servo_active = True
        self._info("Servo interface activated")

        # ── Flange mode: freeze the pre-tare state ───────────────────
        # The blind tare that starts with the reader absorbs
        # bias + gravity(R_tare); both the orientation it ran at and the
        # profile that was physically on the sensor must be pinned down
        # BEFORE the first frame arrives.
        R_pre_tare = None
        if self.flange_mode:
            if self._switch_io >= 0:
                try:
                    high = bool(int(self.robot.get_digital_input(self._switch_io)))
                except Exception as ex:
                    raise RuntimeError(
                        f"Cannot read profile-switch input DI {self._switch_io} "
                        f"(payload.switch_io): {ex}"
                    ) from ex
                initial = "tool_load" if high else "tool"
                if initial != self._active_profile:
                    self._compensator.set_profile(self._profiles[initial])
                    self._active_profile = initial
                self._switch_request = initial
                self._info(
                    f"Profile from switch input DI {self._switch_io}: {initial}"
                )
            pose = self.robot.get_tcp_pose()
            R_pre_tare = np.array(rotation_matrix_zyx(pose[3], pose[4], pose[5]))

        # ── Start sensor reader ──────────────────────────────────────
        self._reader.start()
        self._info("Waiting for sensor zero calibration …")

        # ── Stop mechanism: the digital input from the config ────────
        self._stop_event.clear()
        self._io_thread = threading.Thread(
            target=self._io_stop_watcher,
            name="FTS_IOStopWatcher",
            daemon=True,
        )
        self._io_thread.start()

        # ── Control loop ─────────────────────────────────────────────
        zero_accel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        zero_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        next_t = time.perf_counter()
        loop_start = next_t
        # -inf forces the first tick to fetch pose/q: perf_counter()'s epoch
        # is arbitrary (near zero at process start on Windows), so 0.0 would
        # NOT be "long ago" and the loop could run its first POSE_REFRESH
        # window on the frame transformer's default identity orientation.
        pose_t = -math.inf
        q_t = -math.inf
        ticks = 0
        overruns = 0
        tcp_pose = None     # cached [x,y,z,r,p,y]; refreshed at POSE_REFRESH
        last_tick_t = None  # start of the previous command tick (perf_counter)

        # ── Flange-mode loop state ───────────────────────────────────
        # phase: "verify" (post-tare residual window, zeros commanded)
        #        → "active" (normal control)
        #        → "profile_verify" (after a switch, zeros until settled)
        phase = "verify" if self.flange_mode else "active"
        tare_done = False
        verify_deadline = 0.0
        verify_sum = np.zeros(6)     # sum of compensated wrench over the window
        verify_count = 0
        verify_max_f = 0.0           # peak norm, kept for the diagnostic message
        verify_max_m = 0.0
        switch_start = 0.0
        hold_start = None
        drift_ema = np.zeros(6)
        last_drift_warn = -math.inf   # -inf: see _last_overforce_warn note
        prev_cmd_vel = list(zero_vel)
        accel_cmd = np.zeros(3)     # last commanded linear accel (base frame)

        # ── Contact damping loop state ────────────────────────────────
        damping_scale = 1.0
        damping_reversing = False

        # Console force display timer
        last_force_print_t = -math.inf

        # A reader that dies (bad port, serial error) or a sensor that sends
        # no frames would otherwise leave the zero-wait below spinning
        # forever — with collision detection already disabled.
        zero_deadline = (time.perf_counter()
                         + float(self.config.param.auto_zero_seconds) + 5.0)

        self._loop_thread = threading.current_thread()
        self._loop_exited.clear()
        try:
            while not self._stop_event.is_set():
                # Wait for auto-zero to finish.  Interruptible, so a stop
                # during calibration is honoured immediately.
                if not self._reader.zero_done:
                    if not self._reader.is_running:
                        self._stop_reason = "reader_thread_died"
                        self._error(
                            "Sensor reader thread died during zero calibration"
                        )
                        break
                    if time.perf_counter() > zero_deadline:
                        self._stop_reason = "zero_calibration_timeout"
                        self._error(
                            "Sensor zero calibration did not finish in time — "
                            "no frames from the sensor?"
                        )
                        break
                    if self._stop_event.wait(0.05):
                        break
                    next_t = time.perf_counter()
                    continue

                # ── Flange mode: validate + freeze the tare (once) ────
                if self.flange_mode and not tare_done:
                    # The tare is only meaningful if the robot held its
                    # orientation through the window (nobody pushed it, no
                    # drive moved it).
                    tcp_pose = self.robot.get_tcp_pose()
                    R_post = np.array(rotation_matrix_zyx(
                        tcp_pose[3], tcp_pose[4], tcp_pose[5]
                    ))
                    if _rotation_angle_between(R_pre_tare, R_post) > self.TARE_ORIENT_TOL_RAD:
                        self._stop_reason = "tare_motion"
                        self._error(
                            "Robot orientation changed during the tare window — "
                            "tare invalid, not starting control"
                        )
                        break

                    now = time.perf_counter()
                    self._frame_tf.set_orientation(
                        tcp_pose[3], tcp_pose[4], tcp_pose[5]
                    )
                    pose_t = now
                    self._compensator.set_tcp_orientation(
                        tcp_pose[3], tcp_pose[4], tcp_pose[5]
                    )
                    self._compensator.capture_tare()
                    if self._wrench_filter is not None:
                        self._wrench_filter.reset()
                    tare_done = True
                    verify_deadline = now + self.TARE_VERIFY_SECONDS
                    verify_sum[:] = 0.0
                    verify_count = 0
                    verify_max_f = 0.0
                    verify_max_m = 0.0
                    self._info(
                        f"Tare captured (profile '{self._active_profile}') — "
                        "verifying residual, keep hands off"
                    )
                    next_t = time.perf_counter()
                    continue

                next_t += self.CONTROL_PERIOD
                now = time.perf_counter()

                # Real elapsed time for the dt-dependent stages below (slew
                # limiter, inertial accel, drift EMA, converter smoothing).
                # The loop TARGETS CONTROL_PERIOD (1 ms) but the achieved
                # tick is RPC-bound (~15 ms on the Windows dev box); feeding
                # the nominal value made the slew limiter ~15× stiffer than
                # its configured accel and stretched release-to-stop to ~1 s.
                # Clamped so a one-off stall cannot burst-release the whole
                # accumulated ramp in a single tick.
                if last_tick_t is None:
                    tick_dt = self.CONTROL_PERIOD
                else:
                    tick_dt = min(max(now - last_tick_t, 1e-4), 0.05)
                last_tick_t = now

                # 1. A frozen or absent sensor must never drive the robot.
                if not self._reader.is_running:
                    self._stop_reason = "reader_thread_died"
                    break

                raw, age = self._reader.get_latest_with_age()
                if age > self.MAX_SENSOR_AGE:
                    self._stop_reason = f"sensor_stale_{age * 1e3:.1f}ms"
                    break

                # Crosstalk correction first — everything downstream
                # (gravity model, filter, dead-zone) sees the same corrected
                # sensor the payload profile was calibrated on.
                if self._sensor_K is not None:
                    raw = np.asarray(raw, dtype=float)
                    raw = raw - self._sensor_K @ raw

                # 2. Refresh the robot-state RPCs on their own, slower
                #    deadlines — reading them every tick would not fit the
                #    1 ms budget.
                if now - pose_t >= self.POSE_REFRESH:
                    tcp_pose = self.robot.get_tcp_pose()
                    self._frame_tf.set_orientation(
                        tcp_pose[3], tcp_pose[4], tcp_pose[5]
                    )
                    if self.flange_mode:
                        self._compensator.set_tcp_orientation(
                            tcp_pose[3], tcp_pose[4], tcp_pose[5]
                        )
                    pose_t = now

                # A stale pose means the gravity model points the wrong way —
                # compensation would inject a phantom operator wrench.
                # Defensive: today a failed get_tcp_pose() raises and stops
                # the loop anyway, but this survives future refactors.
                if self.flange_mode and now - pose_t > self.POSE_MAX_AGE:
                    self._stop_reason = f"pose_stale_{(now - pose_t) * 1e3:.0f}ms"
                    break

                if self.use_joint_space and (self._q is None or now - q_t >= self.Q_REFRESH):
                    self._q = self.robot.get_current_joint_angles()
                    q_t = now

                # 2b. Profile switch requested by the DI watcher (load
                #     released / picked up).  Swap the gravity model, then
                #     hold zeros until the residual settles (phase logic in
                #     step 6b) — this also catches a pin that lies about the
                #     physical load state.
                if self.flange_mode and self._switch_request != self._active_profile:
                    req = self._switch_request
                    if phase == "verify":
                        # Load state changed while the tare was being taken /
                        # verified — the frozen reference is ambiguous now.
                        self._stop_reason = "profile_switch_during_startup"
                        break
                    self._compensator.set_profile(self._profiles[req])
                    self._active_profile = req
                    if self._wrench_filter is not None:
                        self._wrench_filter.reset()
                    # Anchor the give-up timeout to the FIRST entry into
                    # profile_verify — a chattering switch pin must not keep
                    # resetting it, or the 5 s give-up becomes unreachable.
                    if phase != "profile_verify":
                        switch_start = now
                    phase = "profile_verify"
                    hold_start = None
                    self._info(
                        f"Payload profile switching to '{req}' — holding zero "
                        "velocity until the reading settles"
                    )

                # 3. Sensor-frame processing: gravity compensation (flange
                #    mode), plausibility stop, optional low-pass — all before
                #    the frame transforms so the proven chain is untouched.
                if self.flange_mode:
                    w = self._compensator.compensate(raw)

                    res_f = float(np.linalg.norm(w[:3]))
                    res_m = float(np.linalg.norm(w[3:]))
                    # Cache the real (unclamped) compensated force so shutdown
                    # still knows the robot is loaded even if this tick exits
                    # through the hard over-force guard below.
                    self._last_force_norm = res_f
                    self._last_force_t = now
                    # Collision guard: collision detection is disabled while
                    # hand guiding, so an implausibly large "operator" wrench
                    # is the only crash indicator available here.  With
                    # overforce_policy "saturate", the band between the limit
                    # and hard_stop_multiplier × limit clamps instead of stopping.
                    # Beyond that hard threshold, control stops.
                    if res_f > self._max_op_f or res_m > self._max_op_m:
                        #hard = (res_f > 2.0 * self._max_op_f
                        #        or res_m > 2.0 * self._max_op_m)
                        hard = (
                            res_f > self._hard_stop_multiplier * self._max_op_f
                            or    
                            res_m > self._hard_stop_multiplier * self._max_op_m
                            )
                        if not self._overforce_saturate or hard:
                            self._stop_reason = (
                                f"wrench_implausible_F{res_f:.0f}N_M{res_m:.0f}Nm"
                            )
                            break
                        # Radial clamp preserves the push/twist direction.
                        if res_f > self._max_op_f:
                            w[:3] *= self._max_op_f / res_f
                        if res_m > self._max_op_m:
                            w[3:] *= self._max_op_m / res_m
                        if now - self._last_overforce_warn > self.DRIFT_WARN_INTERVAL:
                            self._info(
                                f"NOTE: operator wrench {res_f:.0f} N / "
                                f"{res_m:.0f} Nm exceeds the limit "
                                f"({self._max_op_f:.0f} N / {self._max_op_m:.0f} Nm)"
                                " — speed is capped, pushing harder does nothing"
                            )
                            self._last_overforce_warn = now

                    if self._inertial_ff:
                        w = w + self._compensator.inertial_wrench(accel_cmd)
                else:
                    w = np.array(raw, dtype=float)

                if self._wrench_filter is not None:
                    w = self._wrench_filter.filter(w)

                if self.flange_mode:
                    # Residuals of the FILTERED wrench — what the dead-zone
                    # actually sees — drive the verify phases and the drift
                    # monitor.
                    res_f = float(np.linalg.norm(w[:3]))
                    res_m = float(np.linalg.norm(w[3:]))

                # 3b. Sensor frame → TCP frame transform
                f_sensor = w[:3]
                m_sensor = w[3:]

                f_tcp = self.R_sensor_tcp @ f_sensor
                m_tcp = self.R_sensor_tcp @ m_sensor
                if self.has_lever_arm:
                    # Moment reference shift, sensor origin → TCP origin:
                    #   m_TCP = R·m_S + r_{S/TCP} × f,  r_{S/TCP} = −p_lever
                    # A pure force applied AT the TCP therefore yields zero
                    # TCP torque (the sensor's lever-arm moment cancels).
                    m_tcp -= np.cross(self.p_lever_tcp, f_tcp)

                fx_tcp, fy_tcp, fz_tcp = f_tcp
                mx_tcp, my_tcp, mz_tcp = m_tcp

                # 4. Rotate TCP frame → Base frame
                fx_b, fy_b, fz_b, mx_b, my_b, mz_b = self._frame_tf.transform(
                    fx_tcp, fy_tcp, fz_tcp, mx_tcp, my_tcp, mz_tcp
                )

                # 5. FREE DRIVE + continuous contact damping
                #
                # The normal ForceToVelocity mapping remains active.  There is
                # no contact latch, STOP, UNLOAD, or BLOCKED state.  High force
                # continuously reduces linear speed, and sudden direction reversal
                # is damped more strongly to suppress rebound.
                force_b = np.array([fx_b, fy_b, fz_b], dtype=float)
                force_norm = float(np.linalg.norm(force_b))
                self._last_force_norm = force_norm
                self._last_force_t = now

                if self.enable_orientation:
                    self._converter.update(
                        fx_b, fy_b, fz_b, mx_b, my_b, mz_b, dt=tick_dt
                    )
                else:
                    self._converter.update(
                        fx_b, fy_b, fz_b, 0.0, 0.0, 0.0, dt=tick_dt
                    )

                velocity = list(self._converter.get_velocity())

                if phase == "active":
                    velocity, damping_scale, damping_reversing = (
                        self._apply_contact_damping(
                            velocity, force_norm, prev_cmd_vel
                        )
                    )
                else:
                    damping_scale = 1.0
                    damping_reversing = False

                if now - last_force_print_t >= self.FORCE_PRINT_INTERVAL:
                    print(
                        f"[MODE=FREE_DRIVE] F={force_norm:.2f} N | "
                        f"Damp={damping_scale:.2f} | "
                        f"Reverse={'YES' if damping_reversing else 'NO'} | "
                        f"Fx={fx_b:.2f} N, "
                        f"Fy={fy_b:.2f} N, "
                        f"Fz={fz_b:.2f} N",
                        flush=True,
                    )
                    last_force_print_t = now

                # 6. Zero out rotation when orientation control is disabled
                if not self.enable_orientation:
                    velocity[3] = 0.0
                    velocity[4] = 0.0
                    velocity[5] = 0.0

                # 6b. Flange-mode phase logic: startup and profile-switch
                #     verification hold zeros until the compensated wrench
                #     proves the gravity model matches physical reality; the
                #     drift monitor watches for a degrading tare while idle.
                if self.flange_mode:
                    if phase == "verify":
                        # Gate on the MEAN residual over the window, not the
                        # peak: at the tare orientation the compensated wrench
                        # is zero-mean sensor noise, so the norm of a single
                        # sample (and thus its peak over the window) sits near
                        # the noise floor (~1 N here) and trips a 1 N gate even
                        # with a perfect tare.  Averaging cancels the noise
                        # (std/√N); a real sustained force — the tool touched
                        # during the tare, a snagged cable — survives it.
                        verify_sum += w
                        verify_count += 1
                        verify_max_f = max(verify_max_f, res_f)
                        verify_max_m = max(verify_max_m, res_m)
                        velocity = list(zero_vel)
                        if now >= verify_deadline:
                            mean_w = verify_sum / max(verify_count, 1)
                            mean_f = float(np.linalg.norm(mean_w[:3]))
                            mean_m = float(np.linalg.norm(mean_w[3:]))
                            if (mean_f > self._startup_max_res_f
                                    or mean_m > self._startup_max_res_m):
                                self._stop_reason = (
                                    f"tare_residual_F{mean_f:.2f}N"
                                    f"_M{mean_m:.2f}Nm"
                                )
                                self._error(
                                    f"Post-tare residual too large "
                                    f"(mean {mean_f:.2f} N / {mean_m:.2f} Nm, "
                                    f"peak {verify_max_f:.2f} N / "
                                    f"{verify_max_m:.2f} Nm) — wrong payload "
                                    "profile, or the tool was touched during "
                                    "the tare. Not starting control"
                                )
                                break
                            phase = "active"
                            drift_ema[:] = 0.0
                            self._info(
                                f"Tare verified (mean residual {mean_f:.2f} N / "
                                f"{mean_m:.2f} Nm, peak {verify_max_f:.2f} N / "
                                f"{verify_max_m:.2f} Nm) — hand guiding active"
                            )
                    elif phase == "profile_verify":
                        velocity = list(zero_vel)
                        if (res_f <= self._startup_max_res_f
                                and res_m <= self._startup_max_res_m):
                            if hold_start is None:
                                hold_start = now
                            elif now - hold_start >= self.PROFILE_VERIFY_HOLD:
                                phase = "active"
                                drift_ema[:] = 0.0
                                self._info(
                                    f"Profile '{self._active_profile}' verified — "
                                    "hand guiding active"
                                )
                        else:
                            hold_start = None
                        if phase == "profile_verify" and now - switch_start > self.PROFILE_VERIFY_TIMEOUT:
                            self._stop_reason = "profile_switch_verify_failed"
                            self._error(
                                f"Compensated wrench did not settle within "
                                f"{self.PROFILE_VERIFY_TIMEOUT:.0f} s of the profile "
                                "switch — does the switch input match the real "
                                "load state?"
                            )
                            break
                    else:
                        # Drift monitor: while FREE_DRIVE reads released, a slow
                        # EMA of the compensated wrench approaching the dead-zone
                        # means the tare is degrading.
                        if not any(velocity):
                            alpha = 1.0 - math.exp(-tick_dt / self.DRIFT_EMA_TAU)
                            drift_ema += alpha * (w - drift_ema)
                            if (np.linalg.norm(drift_ema[:3])
                                    > self.DRIFT_WARN_FRACTION * self._dead_zone_force
                                    or np.linalg.norm(drift_ema[3:])
                                    > self.DRIFT_WARN_FRACTION * self._dead_zone_torque):
                                if now - last_drift_warn > self.DRIFT_WARN_INTERVAL:
                                    self._info(
                                        "WARNING: compensated wrench is drifting "
                                        f"toward the dead-zone "
                                        f"({np.linalg.norm(drift_ema[:3]):.2f} N / "
                                        f"{np.linalg.norm(drift_ema[3:]):.2f} Nm at rest) "
                                        "— restart the app to re-tare"
                                    )
                                    last_drift_warn = now
                        else:
                            drift_ema[:] = 0.0

                    # 6c. Acceleration slew limit — caps the tool's m·a
                    #     inertial term so it can never cross the dead-zone
                    #     on its own (no self-sustained limit cycle).
                    velocity = self._slew.limit(velocity, tick_dt)
                    if self._inertial_ff:
                        accel_cmd = (np.array(velocity[:3])
                                     - np.array(prev_cmd_vel[:3])) / tick_dt
                    prev_cmd_vel = velocity

                # 7. Command every tick — the velocity while the operator
                #    pushes, zeros the moment they let go.  Nothing is gated
                #    on a previous state, so no missed transition can leave
                #    the robot running.
                if self._stop_event.is_set():
                    # Stop request: overwrite the last motion command with ZERO
                    # immediately.  Shutdown will continue streaming zeros.
                    try:
                        if self.use_joint_space:
                            self.robot.speed_j(zero_vel, zero_accel)
                        else:
                            self.robot.speed_x(zero_vel, zero_accel)
                    except Exception as ex:
                        self._error(f"Immediate zero-speed command failed: {ex}")
                    break

                if self.use_joint_space:
                    if any(velocity):
                        joint_vel = self._kin.cartesian_to_joint_velocity(
                            self._q, np.array(velocity), damping=self.DAMPING
                        )
                        jv_list = joint_vel.tolist()

                        # Ramp down near joint limits
                        if self._joint_limits:
                            jv_list = _clamp_joint_velocities(
                                self._q, jv_list, self._joint_limits,
                                self._buffer_rad, self._hard_stop_rad,
                            )

                        # Max joint velocity clamp (proportional)
                        max_abs = max(abs(v) for v in jv_list)
                        if max_abs > self.MAX_JOINT_VEL:
                            scale = self.MAX_JOINT_VEL / max_abs
                            jv_list = [v * scale for v in jv_list]
                    else:
                        jv_list = zero_vel

                    self.robot.speed_j(jv_list, zero_accel)
                else:
                    jv_list = None
                    self.robot.speed_x(velocity, zero_accel)

                if self._tick_log is not None:
                    self._tick_log.log(
                        now - loop_start, age, now - pose_t, phase,
                        raw, w, (fx_b, fy_b, fz_b, mx_b, my_b, mz_b),
                        velocity, jv_list, tcp_pose, self._q,
                    )

                ticks += 1

                # 8. Pace to the deadline with time.sleep(), NOT Event.wait():
                #    on Windows a sub-millisecond Event.wait() timeout rounds up
                #    to the ~15.6 ms system timer tick, which would silently cap
                #    this loop at ~64 Hz.  time.sleep() is sub-ms on both Linux
                #    and Windows.  A stop signalled during the sleep costs at
                #    most one CONTROL_PERIOD, checked at the top of the loop and
                #    again just before the next command is issued.
                slack = next_t - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    overruns += 1
                    next_t = time.perf_counter()  # late; resync, don't spiral

        except Exception as ex:
            self._stop_reason = self._stop_reason or f"control_loop_error: {ex}"
            traceback.print_exc()
            self._error(f"Error in FTS control loop: {ex}")
        finally:
            # First statement: no further commands will be issued.  A
            # concurrent _shutdown() (from finish()) is blocked on this
            # event before it streams its zeros, so those zeros provably
            # land after the loop's last command.
            self._loop_exited.set()
            elapsed = time.perf_counter() - loop_start
            mean_ms = (elapsed / ticks * 1e3) if ticks else 0.0
            self._info(f"Control loop ended — {ticks} ticks, {overruns} overruns, "
                  f"mean period {mean_ms:.3f} ms "
                  f"(target {self.CONTROL_PERIOD * 1e3:.3f} ms)")
            if self._tick_log is not None:
                self._tick_log.close()
                self._info(f"Tick log saved: {self._tick_log.path} "
                           f"({self._tick_log.rows} rows)")
            self._shutdown()

    # ─────────────────────────────────────────────────────────────────
    # Console mirroring — neurapy.robot_old.notify_* only reach the robot
    # controller (that module never writes to stdout), so echo every
    # operator-facing message to stderr too, or a stop looks silent here.
    # ─────────────────────────────────────────────────────────────────

    def _console(self, msg):
        """Echo to stderr without ever raising.  A Windows OEM console
        (cp850/cp437) can't encode the em-dash/ellipsis in these messages;
        a diagnostic print must degrade to '?' rather than become a new
        crash that masks the real stop reason."""
        try:
            print(msg, file=sys.stderr, flush=True)
        except (UnicodeEncodeError, OSError):
            enc = getattr(sys.stderr, "encoding", None) or "ascii"
            try:
                print(msg.encode(enc, "replace").decode(enc),
                      file=sys.stderr, flush=True)
            except Exception:
                pass

    def _info(self, msg):
        self._console(msg)
        self.robot.notify_info(msg)

    def _error(self, msg):
        self._console("ERROR: " + msg)
        self.robot.notify_error(msg)

    # ─────────────────────────────────────────────────────────────────
    # finish — cleanup
    # ─────────────────────────────────────────────────────────────────

    def finish(self):
        self._shutdown()

    @staticmethod
    def _rpy_pose_to_servox_pose(pose):
        """Convert [x,y,z,roll,pitch,yaw] to [x,y,z,qw,qx,qy,qz].

        Neurapy servo_x() uses a quaternion target pose.  Some robot_old
        get_tcp_pose() versions used by this application expose RPY instead,
        so convert locally and avoid depending on rpy_to_quaternion() being
        present on robot_old.
        """
        if len(pose) == 7:
            return [float(v) for v in pose]
        if len(pose) != 6:
            raise ValueError(
                f"Unsupported TCP pose length {len(pose)}; expected 6 (RPY) or 7 (quaternion)"
            )

        x, y, z, roll, pitch, yaw = [float(v) for v in pose]
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        # Quaternion for ZYX yaw-pitch-roll, returned as [w, x, y, z].
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return [x, y, z, qw, qx, qy, qz]

    def _position_hold_stop_servo(self):
        """One-DI fixed-position HOLD, then automatic EXIT request.

        State sequence:
          FREE_DRIVE --(first DI high)--> HOLD --(10 x servo_x, 100 ms)--> EXIT

        HOLD behavior:
          1. Overwrite the last speed_x/speed_j command with zero.
          2. Capture the CURRENT TCP pose ONCE.
          3. Send that exact SAME pose with servo_x() exactly 10 times.
          4. Wait POSITION_HOLD_INTERVAL (100 ms) after each command.
          5. Return True so _shutdown() proceeds directly to EXIT.

        No FTS-force threshold, DI release, or second DI press is used.
        """
        if not self._servo_active:
            self._error("Position HOLD requested but external servo is not active")
            return False

        zero = [0.0] * 6

        # Step 1: cancel the last velocity command immediately.
        try:
            if self.use_joint_space:
                self.robot.speed_j(zero, zero)
            else:
                self.robot.speed_x(zero, zero)
        except Exception as ex:
            self._error(f"HOLD: initial zero-speed command failed: {ex}")

        # Step 2: capture the pose ONCE. Do not refresh this target later.
        try:
            tcp_pose = self.robot.get_tcp_pose()
            hold_pose = self._rpy_pose_to_servox_pose(tcp_pose)
        except Exception as ex:
            self._error(
                f"HOLD: cannot capture/convert TCP pose ({ex}); "
                "falling back to zero-velocity HOLD before EXIT"
            )
            return self._stream_zero_velocity_before_exit()

        servo_x = getattr(self.robot, "servo_x", None)
        if not callable(servo_x):
            self._error(
                "HOLD: robot API has no servo_x(); "
                "falling back to zero-velocity HOLD before EXIT"
            )
            return self._stream_zero_velocity_before_exit()

        self._info(
            "STATE=HOLD — one DI press accepted; captured one TCP pose. "
            f"Sending the SAME pose {self.POSITION_HOLD_REPEAT} times, "
            f"{self.POSITION_HOLD_INTERVAL * 1000.0:.0f} ms apart "
            f"(gain={self.POSITION_HOLD_GAIN:.1f}), then AUTO EXIT."
        )

        for i in range(self.POSITION_HOLD_REPEAT):
            try:
                error_code = servo_x(
                    hold_pose, zero, zero, self.POSITION_HOLD_GAIN
                )
                if error_code not in (None, 0):
                    self._error(
                        f"HOLD: servo_x returned error code {error_code}; "
                        "switching to zero-velocity HOLD before EXIT"
                    )
                    return self._stream_zero_velocity_before_exit()
            except Exception as ex:
                self._error(
                    f"HOLD: servo_x failed ({ex}); "
                    "switching to zero-velocity HOLD before EXIT"
                )
                return self._stream_zero_velocity_before_exit()

            self._info(
                f"STATE=HOLD — servo_x {i + 1}/{self.POSITION_HOLD_REPEAT}"
            )
            time.sleep(self.POSITION_HOLD_INTERVAL)

        self._info(
            "STATE=EXIT — HOLD sequence complete; proceeding directly to EXIT "
            "without waiting for another DI press"
        )
        return True

    def _stream_zero_velocity_before_exit(self):
        """Fallback: send zero velocity 10 times, then request automatic EXIT."""
        zero = [0.0] * 6
        self._info(
            "STATE=HOLD_ZERO — sending zero velocity before automatic EXIT"
        )

        for i in range(self.POSITION_HOLD_REPEAT):
            try:
                if self.use_joint_space:
                    self.robot.speed_j(zero, zero)
                else:
                    self.robot.speed_x(zero, zero)
            except Exception as ex:
                self._error(f"HOLD_ZERO command failed: {ex}")

            self._info(
                f"STATE=HOLD_ZERO — zero command {i + 1}/{self.POSITION_HOLD_REPEAT}"
            )
            time.sleep(self.POSITION_HOLD_INTERVAL)

        self._info(
            "STATE=EXIT — zero-velocity fallback complete; proceeding to EXIT"
        )
        return True

    def _shutdown(self):
        """One-DI FREE_DRIVE -> HOLD -> EXIT shutdown sequence.

        The first Stop press ends FREE_DRIVE.  The captured TCP pose is then
        commanded 10 times at 100 ms intervals and the sequence proceeds
        directly to EXIT.  No DI release or second press is required.

        IMPORTANT: there is intentionally NO force-based EXIT condition.
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True

        self._stop_event.set()

        # Ensure the main loop's final command is finished before HOLD starts.
        if (self._loop_thread is not None
                and threading.current_thread() is not self._loop_thread):
            self._loop_exited.wait(timeout=1.0)

        self._info(
            "STATE transition: FREE_DRIVE -> HOLD -> EXIT from one DI press."
        )

        # Performs the fixed-position HOLD sequence, then requests EXIT automatically.
        exit_requested = self._position_hold_stop_servo()
        if not exit_requested:
            self._error(
                "HOLD ended without an EXIT request; leaving External Servo active"
            )
            return

        # Retire the original watcher thread after HOLD.
        if self._io_thread is not None:
            self._io_thread.join(timeout=0.5)
            self._io_thread = None

        # Stop FTS/converter resources before handing control back.
        if self._reader is not None:
            self._reader.stop()
        if self._converter is not None:
            self._converter.reset()
        if self._slew is not None:
            self._slew.reset()
        if self._wrench_filter is not None:
            self._wrench_filter.reset()
        if self._compensator is not None:
            self._compensator.reset()

        # Critical ordering: do NOT enable collision/reflex before deactivation.
        # This one-DI version exits automatically after the fixed-position HOLD
        # sequence.  Deactivate External Servo first, then restore controller
        # collision/reflex handling.
        if self._servo_active:
            try:
                self.robot.deactivate_servo_interface()
                self._servo_active = False
            except Exception as ex:
                self._error(f"EXIT: deactivate_servo_interface failed: {ex}")
                return

        try:
            self.robot.enable_collision_detection()
        except Exception as ex:
            self._error(f"EXIT: enable_collision_detection failed: {ex}")
        try:
            self.robot.enable_reflex()
        except Exception as ex:
            self._error(f"EXIT: enable_reflex failed: {ex}")
        try:
            self.robot.stop()
        except Exception as ex:
            self._error(f"EXIT: robot.stop() failed: {ex}")

        self._info(
            f"FTS App finished after EXIT (reason: {self._stop_reason or 'normal'})"
        )

    # ─────────────────────────────────────────────────────────────────
    # Stop signal thread
    # ─────────────────────────────────────────────────────────────────

    def _io_stop_watcher(self):
        """Poll the configured digital input; stop control the moment it reads high.

        Reads before the first wait so an input already held at startup is
        caught on tick zero, and waits on _stop_event rather than sleeping so
        another stop path retires this thread at once.

        In flange mode with payload.switch_io configured, the same thread
        also polls the profile-switch input and publishes the requested
        profile name via _switch_request (a plain attribute write — atomic
        under the GIL); the control loop performs the actual switch.
        """
        while True:
            try:
                pressed = bool(int(self.robot.get_digital_input(self.io_num)))
            except Exception as ex:
                # A watchdog we cannot read is a watchdog that is not
                # protecting anyone.  Fail safe.
                self._error(f"IO watchdog read failed on DI {self.io_num} — stopping: {ex}")
                self._stop_reason = f"io_read_error: {ex}"
                self._stop_event.set()
                return

            if pressed:
                self._info(f"Stop signal received (digital input {self.io_num})")
                self._stop_reason = "io_high"
                self._stop_event.set()
                return

            if self._switch_io >= 0:
                try:
                    high = bool(int(self.robot.get_digital_input(self._switch_io)))
                except Exception as ex:
                    # An unreadable switch pin means the gravity model may no
                    # longer match the physical load — that is a drift hazard,
                    # not a cosmetic problem.  Fail safe.
                    self._error(
                        f"Profile-switch input DI {self._switch_io} read failed — stopping: {ex}"
                    )
                    self._stop_reason = f"switch_io_read_error: {ex}"
                    self._stop_event.set()
                    return
                self._switch_request = "tool_load" if high else "tool"

            if self._stop_event.wait(self.IO_POLL_INTERVAL):
                return  # stopped by another path; nothing left to watch


# ── Module-level helper functions ────────────────────────────────────
# (Kept outside the class to match fts_hand_guid_ctrl.py structure)


def _rotation_angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    """Angle (rad) of the relative rotation R1ᵀ·R2."""
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def _deadzone_floors(
    mass_kg: float,
    com_norm: float,
    max_accel_linear: float,
    orientation_on: bool,
    gravity: float,
) -> Tuple[float, float]:
    """Recommended dead-zone floors (N, Nm) for flange mode.

    Error budget for the compensated wrench (safety factor S = 1.5):
      • q       = 0.2 N / 0.2 Nm  — ~2 LSB of sensor quantisation/noise
      • m·a_max — the commanded inertial term (always present)
    and, when orientation control is on (the wrist can rotate away from the
    tare orientation, re-exposing the gravity-model error):
      • 2·g·δm       with assumed mass error δm = 2 % of m
      • m·g·δθ       with assumed orientation error δθ = 1°
      • 2·m·g·δr     with assumed COM error δr = 5 mm (torque channel)

    With orientation off, the tare absorbs the whole model error and only
    the inertial + noise terms remain.
    """
    S = 1.5
    q_f = 0.2
    q_m = 0.2
    dm = 0.02 * mass_kg
    dtheta = math.radians(1.0)
    dr = 0.005

    floor_f = mass_kg * max_accel_linear + q_f
    floor_m = mass_kg * max_accel_linear * com_norm + q_m
    if orientation_on:
        floor_f += 2.0 * gravity * dm + mass_kg * gravity * dtheta
        floor_m += 2.0 * mass_kg * gravity * dr + 2.0 * gravity * dm * com_norm
    return S * floor_f, S * floor_m


def _clamp_joint_velocities(
    joint_angles: list,
    joint_velocities: list,
    joint_limits: list,
    buffer_rad: float,
    hard_stop_rad: float,
) -> list:
    """Ramp down joint velocities near limits.

    Velocity toward a limit is linearly reduced in the buffer zone
    and zeroed inside the hard-stop zone.  Velocity moving away from
    a limit is never restricted.
    """
    clamped = list(joint_velocities)
    for i, (q, qd, (q_min, q_max)) in enumerate(
        zip(joint_angles, joint_velocities, joint_limits)
    ):
        dist_to_min = q - q_min
        dist_to_max = q_max - q

        if qd < 0 and dist_to_min < buffer_rad:
            if dist_to_min <= hard_stop_rad:
                scale = 0.0
            else:
                scale = (dist_to_min - hard_stop_rad) / (buffer_rad - hard_stop_rad)
            clamped[i] = qd * scale

        elif qd > 0 and dist_to_max < buffer_rad:
            if dist_to_max <= hard_stop_rad:
                scale = 0.0
            else:
                scale = (dist_to_max - hard_stop_rad) / (buffer_rad - hard_stop_rad)
            clamped[i] = qd * scale

    return clamped


class _TickLogger:
    """Per-tick CSV logger for offline motion analysis (--log).

    One row per commanded control tick: loop time, staleness of the sensor
    frame and of the cached TCP pose, the wrench at each pipeline stage
    (raw sensor → after compensation/filtering → base frame), the commanded
    cartesian and joint velocities, and the cached robot state (TCP pose
    from get_tcp_pose() as [x,y,z,roll,pitch,yaw], joint angles).  Columns
    that do not apply to the current mode are left empty (read as NaN by
    pandas/numpy).  Writes are OS-buffered — ~0.4 MB/s at the full 1 kHz
    rate, no per-tick flush.
    """

    HEADER = (
        "t_s,sensor_age_ms,pose_age_ms,phase,"
        "raw_fx,raw_fy,raw_fz,raw_mx,raw_my,raw_mz,"
        "w_fx,w_fy,w_fz,w_mx,w_my,w_mz,"
        "base_fx,base_fy,base_fz,base_mx,base_my,base_mz,"
        "cmd_vx,cmd_vy,cmd_vz,cmd_wx,cmd_wy,cmd_wz,"
        "cmd_jv1,cmd_jv2,cmd_jv3,cmd_jv4,cmd_jv5,cmd_jv6,"
        "tcp_x,tcp_y,tcp_z,tcp_roll,tcp_pitch,tcp_yaw,"
        "q1,q2,q3,q4,q5,q6"
    )

    def __init__(self, path):
        self.path = str(path)
        self.rows = 0
        self._f = open(self.path, "w", buffering=1 << 16)
        self._f.write(self.HEADER + "\n")

    def log(self, t, sensor_age, pose_age, phase,
            raw, w, w_base, vel, jv, pose, q):
        parts = [
            "{:.4f}".format(t),
            "{:.2f}".format(sensor_age * 1e3),
            "{:.2f}".format(pose_age * 1e3),
            phase,
        ]
        for seq in (raw, w, w_base, vel, jv, pose, q):
            if seq is None:
                parts.extend([""] * 6)
            else:
                parts.extend("{:.6g}".format(float(v)) for v in seq)
        self._f.write(",".join(parts) + "\n")
        self.rows += 1

    def close(self):
        try:
            self._f.close()
        except OSError:
            pass


DEFAULT_CONFIG = "FTS_Free_Drive_config.json"


if __name__ == "__main__":
    import sys

    from neurapy.robot_old import Robot

    # Usage: python FTS_Free_Drive.py [config.json] [--log[=path]]
    #   --log        record every control tick (sensor wrench, commanded
    #                velocities, TCP pose) to logs/fts_ticklog_<timestamp>.csv
    #                at the project root, for offline motion analysis
    #   --log=path   same, to an explicit file
    log_path = None
    positional = []
    for arg in sys.argv[1:]:
        if arg == "--log":
            log_path = ""
        elif arg.startswith("--log="):
            log_path = arg[len("--log="):]
        else:
            positional.append(arg)

    if positional:
        config_path = positional[0]
    else:
        # Default config is beside this .py file, independent of current cwd.
        config_path = Path(__file__).resolve().parent / DEFAULT_CONFIG

    if log_path == "":
        logs_dir = Path(__file__).resolve().parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_path = str(logs_dir
                       / time.strftime("fts_ticklog_%Y%m%d_%H%M%S.csv"))

    # On Windows, time.sleep() granularity defaults to the ~15.6 ms system
    # tick.  That stretches the sensor reader's 0.5 ms poll-yield
    # (M4313SFA9A_helper.py) into ~15 ms gaps — frames go stale and the
    # MAX_SENSOR_AGE guard stops control — and paces the 1 kHz loop at ~64 Hz.
    # Request the 1 ms multimedia timer for the whole process so both run near
    # the production (Linux controller) rate.  No-op off Windows.
    _winmm = None
    if sys.platform == "win32":
        import ctypes
        _winmm = ctypes.windll.winmm
        _winmm.timeBeginPeriod(1)

    # app = FTS_Free_Drive(Robot(request_control=True), config_path)
    app = FTS_Free_Drive(Robot(), config_path, log_path=log_path)
    try:
        app.init()
        app.run()
    finally:
        # In this diagnostic build, a normal DI Stop enters an intentional
        # infinite position-hold inside run()/_shutdown(), so this block is not
        # reached on that path.  If startup fails before servo activation,
        # finish() remains safe/idempotent.
        app.finish()
        if _winmm is not None:
            _winmm.timeEndPeriod(1)
