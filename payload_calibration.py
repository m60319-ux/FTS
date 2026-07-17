"""
Payload calibration for the flange-mounted FTS (M4313SFA9A).

Identifies the tool(+load) mass and centre of mass by holding the robot at
several wrist orientations, averaging the raw sensor reading at each, and
solving the least-squares problem in payload_model.py.  The result is written
into the `payload` section of the FTS_Free_Drive JSON config, per profile.

Calibrate once per physical payload state:

    python payload_calibration.py config.json --profile tool        # bare tool
    python payload_calibration.py config.json --profile tool_load   # tool + load

Poses can come from two sources:

  TAUGHT POSES (recommended) — poses P1…Pn taught on the pendant and stored
  in the robot's point database.  The operator teaches them once (which
  guarantees reachability and collision-freedom in the cell); the script
  visits them by name via move_joint(<name>) and reads them back with
  get_point(<name>, representation="Cartesian") for the pre-flight
  conditioning check.  The LAST pose in the list is the holdout
  (verification) pose; the rest are used for the fit — so pass at least 4:

    python payload_calibration.py config.json --profile tool \
        --poses P1,P2,P3,P4,P5

  WRIST OFFSETS (fallback, no taught poses needed) — the default when
  --poses is not given: the script builds 8 + 2 poses by rotating J4-J6
  from the current pose.

Options:
    --poses A,B,…  comma-separated taught pose names; the LAST is the holdout
    --manual      operator jogs the robot between prompted captures instead
                  of automatic motion (fallback when move_joint is
                  unavailable; robot connection stays read-only)
    --quick       3 poses + 1 holdout instead of 8 + 2 (wrist-offset mode
                  only; ignored with --poses)
    --dry-run     print the planned poses and their predicted conditioning;
                  no motion, no capture, no write
    --no-write    identify and report, but leave the config untouched
    --speed N     move_joint speed override (default 5)

Safety:
  • collision detection is NEVER disabled by this script
  • every motion is announced and requires explicit confirmation
  • per-pose capture is gated on the robot being stationary AND the reading
    variance being sensor-noise-small ("is someone touching the tool?")

The identification needs >= 3 distinct gravity directions in the SENSOR
frame; the default pose set rotates J4-J6 so the flange axis sweeps the
sphere.  Start with the tool axis roughly HORIZONTAL — with the flange
pointing straight down, J6 rolls do not change the gravity direction and
the fit degenerates (the pre-flight check catches this).
"""

import argparse
import json
import math
import os
import sys
import tempfile
import threading
import time
from datetime import datetime

import numpy as np

from FTS_Free_DriveConfig import load_config
from FTS_Free_Drive import _build_sensor_to_tcp_transform
from M4313SFA9A_helper import FTSSensorReader, resolve_serial_port
from frame_transformer import rotation_matrix_zyx
import payload_model as pm

# ── Pose sets: wrist offsets (ΔJ4, ΔJ5, ΔJ6) in degrees ──────────────
FULL_OFFSETS = [
    (0, 0, 0), (0, 0, 90), (0, 0, 180), (0, 0, -90),
    (0, 60, 0), (0, -60, 0), (90, 60, 0), (-90, 60, 0),
]
FULL_HOLDOUT = [(45, 30, 45), (-45, -30, -45)]
QUICK_OFFSETS = [(0, 0, 0), (0, 0, 90), (0, 60, 0)]
QUICK_HOLDOUT = [(45, 30, 45)]

JOINT_LIMIT_BUFFER_RAD = math.radians(10.0)

# ── Arrival tolerance for taught poses ───────────────────────────────
POSE_ARRIVAL_POS_TOL = 0.010            # m — get_tcp_pose vs taught pose
POSE_ARRIVAL_ROT_TOL = math.radians(2.0)

# ── Capture parameters ───────────────────────────────────────────────
SETTLE_SECONDS = 1.0          # after motion stops, before capture
CAPTURE_SECONDS = 2.0         # averaging window (~4000 frames at 2 kHz)
STATIONARY_TOL_RAD = 1e-3     # max joint delta between two reads
STATIONARY_POLL = 0.2
MOTION_TIMEOUT = 30.0         # give up waiting for a move to finish
MAX_CAPTURE_RETRIES = 2
FORCE_STD_MAX = 0.2           # N  — variance gate ("hands off?")
TORQUE_STD_MAX = 0.05         # Nm
HOLDOUT_MAX_F = 0.5           # N  — end-to-end verification threshold
HOLDOUT_MAX_M = 0.1           # Nm

# ── Default payload section written when the config has none yet ─────
DEFAULT_PROFILE = {
    "mass_kg": 0.0, "com_x": 0.0, "com_y": 0.0, "com_z": 0.0,
    "identified_on": "", "residual_force_rms": 0.0, "residual_torque_rms": 0.0,
}
# enabled stays FALSE when this section is first created: switching the
# runtime into flange mode is an operator decision to be taken after the
# dead-zones / accel limits are reviewed against the flange-mode floors —
# a calibration utility must not arm it as a side effect.
DEFAULT_PAYLOAD_SECTION = {
    "enabled": False,
    "profile_tool": dict(DEFAULT_PROFILE),
    "profile_tool_load": dict(DEFAULT_PROFILE),
    "active_profile": "tool",
    "switch_io": -1,
    "gravity_x": 0.0, "gravity_y": 0.0, "gravity_z": -9.80665,
    "startup_max_residual_force": 1.0,
    "startup_max_residual_torque": 0.3,
    "max_operator_force": 80.0,
    "max_operator_torque": 20.0,
    "max_accel_linear": 0.25,
    "max_accel_angular": 0.5,
    "decel_multiplier": 4.0,
    "inertial_feedforward": False,
}


class _SampleCollector:
    """Accumulate full-rate sensor frames inside an arm/disarm window."""

    def __init__(self):
        self._lock = threading.Lock()
        self._samples = None

    def callback(self, fx, fy, fz, mx, my, mz):
        with self._lock:
            if self._samples is not None:
                self._samples.append((fx, fy, fz, mx, my, mz))

    def arm(self):
        with self._lock:
            self._samples = []

    def disarm(self):
        with self._lock:
            samples = self._samples
            self._samples = None
        return np.array(samples) if samples else np.zeros((0, 6))


# ── Small helpers ────────────────────────────────────────────────────


def _confirm(prompt):
    return input(f"{prompt} [y/N] ").strip().lower() == "y"


def _rotation_from_rpy(pose):
    return np.array(rotation_matrix_zyx(pose[3], pose[4], pose[5]))


def wait_until_stationary(robot, timeout=MOTION_TIMEOUT, target=None,
                          quiet_polls=3, target_tol_rad=0.01):
    """Poll joint angles until they stop changing (handles blocking AND
    non-blocking move_joint implementations).

    Requires *quiet_polls* CONSECUTIVE quiet intervals — a single quiet
    0.2 s window is not proof of rest: a non-blocking move_joint may not
    have started moving yet, and a slow decel tail creeps below any
    one-interval threshold.  When *target* is given, also requires the
    joints to actually be AT the target, so a rejected/dropped motion
    command times out here instead of silently capturing a duplicate pose.
    """
    deadline = time.time() + timeout
    prev = robot.get_current_joint_angles()
    quiet = 0
    while time.time() < deadline:
        time.sleep(STATIONARY_POLL)
        cur = robot.get_current_joint_angles()
        if max(abs(c - p) for c, p in zip(cur, prev)) < STATIONARY_TOL_RAD:
            quiet += 1
        else:
            quiet = 0
        prev = cur
        if quiet >= quiet_polls:
            if target is None or max(
                abs(c - t) for c, t in zip(cur, target)
            ) < target_tol_rad:
                return True
            quiet = 0  # at rest but not at the target — keep waiting
    return False


def abort_with_motion_warning(robot, label):
    """Best-effort halt + unmissable operator warning, then exit.

    Reached when a commanded motion never settled — the robot may STILL be
    moving, and a bare "aborting" invites the operator to walk up to a
    wrist mid-swing.
    """
    stopped = False
    if hasattr(robot, "stop"):
        try:
            robot.stop()
            stopped = True
        except Exception as ex:
            print(f"  robot.stop() failed: {ex}")
    print("\n" + "!" * 62)
    print(f"!! {label}: motion did not settle — ABORTING.")
    if stopped:
        print("!! A stop command was sent, but VERIFY the robot is at rest")
        print("!! from the pendant before approaching it.")
    else:
        print("!! The robot may STILL BE MOVING. Stay clear and stop it")
        print("!! from the pendant / e-stop before approaching.")
    print("!" * 62)
    sys.exit(1)


def fetch_named_poses(robot, names):
    """Read taught poses from the robot's point database.

    Returns {name: [x, y, z, r, p, y]} — used for the pre-flight
    conditioning prediction and the arrival check.  Returns None when the
    controller does not expose get_point (the script then degrades to
    capture-time measurement only).
    """
    if not hasattr(robot, "get_point"):
        return None
    poses = {}
    for name in names:
        try:
            poses[name] = list(robot.get_point(name, representation="Cartesian"))
        except Exception as ex:
            print(f"ERROR: pose '{name}' could not be read from the robot "
                  f"database: {ex}")
            sys.exit(1)
    return poses


def pose_arrival_error(robot, expected):
    """(position error m, orientation error rad) between the current TCP
    pose and a taught pose — guards against a silently rejected motion."""
    cur = robot.get_tcp_pose()
    dpos = math.sqrt(sum((c - e) ** 2 for c, e in zip(cur[:3], expected[:3])))
    R1 = _rotation_from_rpy(cur)
    R2 = _rotation_from_rpy(expected)
    c = (np.trace(R1.T @ R2) - 1.0) / 2.0
    dang = math.acos(max(-1.0, min(1.0, c)))
    return dpos, dang


def build_targets(q0, offsets, joint_limits):
    """q0 + wrist offsets on J4-J6, dropping targets outside the limits."""
    targets = []
    dropped = []
    for off in offsets:
        q = list(q0)
        ok = True
        for j, d_deg in zip((3, 4, 5), off):
            q[j] = q0[j] + math.radians(d_deg)
            if joint_limits is not None:
                lo, hi = joint_limits[j]
                if not (lo + JOINT_LIMIT_BUFFER_RAD <= q[j] <= hi - JOINT_LIMIT_BUFFER_RAD):
                    ok = False
        (targets if ok else dropped).append((off, q))
    return targets, dropped


def predict_g_dirs(kin, q0, targets, R_base_tcp0, R_sensor_tcp, g_base):
    """Predicted sensor-frame gravity directions for planned joint targets.

    Uses the FK flange rotation relative to q0's, composed with the MEASURED
    TCP rotation at q0 — exact for any tool/TCP definition, because the
    flange→TCP transform is constant.
    """
    R_base_fl0 = kin.forward_kinematics(list(q0))[-1][:3, :3]
    R_fl_tcp = R_base_fl0.T @ R_base_tcp0
    dirs = []
    for _, q in targets:
        R_base_fl = kin.forward_kinematics(list(q))[-1][:3, :3]
        R_base_sensor = R_base_fl @ R_fl_tcp @ R_sensor_tcp
        dirs.append(pm.gravity_dir_sensor(R_base_sensor, g_base))
    return dirs


def capture_pose(robot, collector, R_sensor_tcp, g_base, label):
    """Hands-off averaged capture at the current pose.

    Returns (mean_raw6, g_dir) or None after exhausting retries.
    """
    for attempt in range(1 + MAX_CAPTURE_RETRIES):
        if not wait_until_stationary(robot, timeout=5.0):
            print(f"  {label}: robot is still moving — waiting …")
            if not wait_until_stationary(robot):
                print(f"  {label}: robot never settled, skipping pose")
                return None
        print(f"  {label}: capturing {CAPTURE_SECONDS:.0f} s — HANDS OFF the tool")
        time.sleep(SETTLE_SECONDS)
        collector.arm()
        time.sleep(CAPTURE_SECONDS)
        samples = collector.disarm()

        if len(samples) < 100:
            print(f"  {label}: only {len(samples)} frames received — check the sensor")
            return None

        std = samples.std(axis=0)
        if max(std[:3]) > FORCE_STD_MAX or max(std[3:]) > TORQUE_STD_MAX:
            print(f"  {label}: reading is not quiet "
                  f"(std F {max(std[:3]):.2f} N / M {max(std[3:]):.3f} Nm) — "
                  "is someone touching the tool?")
            if attempt < MAX_CAPTURE_RETRIES:
                input("  Let go of the tool and press Enter to retry ")
                continue
            print(f"  {label}: giving up on this pose")
            return None

        pose = robot.get_tcp_pose()
        R_base_sensor = _rotation_from_rpy(pose) @ R_sensor_tcp
        g_dir = pm.gravity_dir_sensor(R_base_sensor, g_base)
        mean = samples.mean(axis=0)
        print(f"  {label}: ok ({len(samples)} frames, "
              f"|F| {np.linalg.norm(mean[:3]):.2f} N, "
              f"|M| {np.linalg.norm(mean[3:]):.3f} Nm)")
        return mean, g_dir
    return None


def write_profile_to_config(config_path, profile_name, profile_dict):
    """Update the payload profile in the JSON config, atomically.

    Creates the whole payload section (with conservative defaults) when the
    config predates flange mode.  The rewritten file is validated with
    load_config() BEFORE it replaces the original.
    """
    with open(str(config_path), "r") as f:
        data = json.load(f)

    if "payload" not in data:
        data["payload"] = json.loads(json.dumps(DEFAULT_PAYLOAD_SECTION))
        data["payload"]["active_profile"] = profile_name
    data["payload"][f"profile_{profile_name}"] = profile_dict

    fd, tmp_path = tempfile.mkstemp(
        suffix=".json", dir=os.path.dirname(os.path.abspath(str(config_path)))
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        load_config(tmp_path)  # must parse cleanly before it replaces the original
        os.replace(tmp_path, str(config_path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def print_report(result, status, messages, holdout_rows):
    p = result.params
    print("\n" + "=" * 62)
    print("Identification result")
    print("=" * 62)
    print(f"  mass          : {p.mass_kg:.4f} kg")
    print(f"  centre of mass: [{p.com[0] * 1e3:+.1f}, {p.com[1] * 1e3:+.1f}, "
          f"{p.com[2] * 1e3:+.1f}] mm (sensor frame)")
    print(f"  sensor bias   : |F0| {np.linalg.norm(p.bias_f):.2f} N, "
          f"|M0| {np.linalg.norm(p.bias_m):.3f} Nm (session-dependent, not stored)")
    print(f"  residual RMS  : {result.force_residual_rms:.3f} N / "
          f"{result.torque_residual_rms:.4f} Nm")
    print(f"  conditioning  : force {result.cond_force:.1f}, "
          f"torque {result.cond_torque:.1f}, spread {result.g_spread:.2f}")
    print("  per-pose residuals:")
    for i, (rf, rm) in enumerate(zip(result.pose_force_residuals,
                                     result.pose_torque_residuals)):
        print(f"    pose {i + 1:2d}: {rf:.3f} N / {rm:.4f} Nm")
    if holdout_rows:
        print("  holdout verification (poses NOT used in the fit):")
        for i, (df, dm, ok) in enumerate(holdout_rows):
            print(f"    holdout {i + 1}: {df:.3f} N / {dm:.4f} Nm "
                  f"[{'ok' if ok else 'FAIL'}]")
    print(f"  status        : {status}")
    for msg in messages:
        print(f"    - {msg}")
    print("=" * 62)


# ── Main flow ────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Identify FTS payload (mass, COM) and store it in the config."
    )
    ap.add_argument("config", nargs="?", default="FTS_Free_Drive_config.json")
    ap.add_argument("--profile", choices=("tool", "tool_load"), required=True,
                    help="which payload profile the CURRENT physical state is")
    ap.add_argument("--poses", default=None,
                    help="comma-separated taught pose names from the robot's "
                         "point database; the LAST one is the holdout "
                         "(verification) pose, so pass at least 4")
    ap.add_argument("--manual", action="store_true",
                    help="operator moves the robot between captures (no "
                         "script-commanded motion; combinable with --poses)")
    ap.add_argument("--quick", action="store_true",
                    help="3 poses + 1 holdout instead of 8 + 2 "
                         "(wrist-offset mode; ignored with --poses)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned poses + conditioning; no motion/write")
    ap.add_argument("--no-write", action="store_true",
                    help="identify and report only")
    ap.add_argument("--speed", type=float, default=5.0,
                    help="move_joint speed (default 5)")
    args = ap.parse_args()

    pose_names = None
    if args.poses:
        pose_names = [s.strip() for s in args.poses.split(",") if s.strip()]
        if len(pose_names) < 4:
            print("ERROR: --poses needs at least 4 names — 3+ for the fit "
                  "plus the LAST one as the holdout verification pose.")
            sys.exit(1)
        if len(set(pose_names)) != len(pose_names):
            print("ERROR: duplicate names in --poses.")
            sys.exit(1)
        if args.quick:
            print("NOTE: --quick is ignored when --poses is given.")

    config = load_config(args.config)
    R_sensor_tcp, _ = _build_sensor_to_tcp_transform(
        float(config.sensor_offset.sensor_x),
        float(config.sensor_offset.sensor_y),
        float(config.sensor_offset.sensor_z),
        float(config.sensor_offset.sensor_a),
        float(config.sensor_offset.sensor_b),
        float(config.sensor_offset.sensor_c),
    )
    if config.payload is not None:
        g_base = np.array([float(config.payload.gravity_x),
                           float(config.payload.gravity_y),
                           float(config.payload.gravity_z)])
    else:
        g_base = np.array([0.0, 0.0, -9.80665])
    g_mag = float(np.linalg.norm(g_base))

    offsets = QUICK_OFFSETS if args.quick else FULL_OFFSETS
    holdout_offsets = QUICK_HOLDOUT if args.quick else FULL_HOLDOUT

    # ── Robot connection ─────────────────────────────────────────────
    from neurapy.robot import Robot

    auto_mode = not args.manual
    need_control = auto_mode and not args.dry_run
    robot = Robot(request_control=need_control)
    print(f"Connected to robot: {robot.robot_name}")

    named_motion = False
    if pose_names is not None:
        named_motion = (not args.manual) and hasattr(robot, "move_joint")
        if not args.manual and not named_motion:
            print("This controller does not expose move_joint — you will be "
                  "prompted to run each taught pose from the pendant instead.")
            if getattr(robot, "is_write_owner", False):
                robot.close()
                robot = Robot(request_control=False)
    elif auto_mode and not hasattr(robot, "move_joint"):
        print("This controller does not expose move_joint — falling back to "
              "manual mode (you will be prompted to jog between captures).")
        auto_mode = False
        # Manual mode is documented read-only; keeping write control would
        # also lock the pendant out of jogging on some controllers.  A clean
        # close is the release mechanism; reconnect read-only.
        if getattr(robot, "is_write_owner", False):
            robot.close()
            robot = Robot(request_control=False)

    q0 = robot.get_current_joint_angles()
    pose0 = robot.get_tcp_pose()
    R_base_tcp0 = _rotation_from_rpy(pose0)

    # ── Plan poses + pre-flight conditioning ─────────────────────────
    targets = []
    holdout_targets = []
    fetched_poses = None
    if pose_names is not None:
        # Taught poses: read them back from the robot's point database so
        # the conditioning of the pose SET is known before any motion.
        fetched_poses = fetch_named_poses(robot, pose_names)
        if fetched_poses is None:
            print("NOTE: get_point is not available on this controller — "
                  "skipping the pre-flight conditioning check; orientations "
                  "are measured at capture time.")
        else:
            pred = [
                pm.gravity_dir_sensor(
                    _rotation_from_rpy(fetched_poses[n]) @ R_sensor_tcp, g_base
                )
                for n in pose_names[:-1]
            ]
            cond_f, cond_m = pm.pose_set_condition(pred)
            spread = pm.g_dir_spread(pred)
            print(f"Pre-flight: {len(pose_names) - 1} identification poses "
                  f"(holdout: '{pose_names[-1]}'), predicted conditioning "
                  f"force {cond_f:.1f} / torque {cond_m:.1f}, spread {spread:.2f}")
            if cond_m > pm.COND_TORQUE_REJECT or spread < 0.15:
                print("WARNING: the taught poses are poorly conditioned — "
                      "their orientations are too similar. Teach poses whose "
                      "TOOL AXIS points in clearly different directions "
                      "(up, down, sideways) and run again.")
                if not _confirm("Continue anyway?"):
                    sys.exit(1)

        if args.dry_run:
            print("\nTaught poses:")
            for n in pose_names:
                tag = "  (holdout)" if n == pose_names[-1] else ""
                if fetched_poses is not None:
                    p = fetched_poses[n]
                    print(f"  {n}: [{', '.join(f'{v:+.3f}' for v in p)}]{tag}")
                else:
                    print(f"  {n}{tag}")
            print("\nDry run — no motion, no capture, no write.")
            return
    else:
        kin = None
        try:
            from robot_kinematics import RobotKinematics
            kin = RobotKinematics(robot.robot_name.upper())
        except Exception as ex:
            print(f"NOTE: no DH model for '{robot.robot_name}' ({ex}) — "
                  "pre-flight conditioning prediction skipped.")

        joint_limits = kin.joint_limits if kin is not None else None
        targets, dropped = build_targets(q0, offsets, joint_limits)
        holdout_targets, holdout_dropped = build_targets(q0, holdout_offsets, joint_limits)

        # The planned wrist offsets only matter in auto mode — a manual-mode
        # operator jogs to arbitrary orientations, and the >= 3 requirement is
        # re-enforced on the actual captures below.
        if auto_mode:
            for off, _ in dropped + holdout_dropped:
                print(f"NOTE: pose offset {off} dropped — outside joint limits")
            if len(targets) < 3:
                print("ERROR: fewer than 3 reachable identification poses — "
                      "reposition the robot and retry.")
                sys.exit(1)
            if not holdout_targets and not args.dry_run:
                print("ERROR: no reachable holdout pose — out-of-fit verification "
                      "is impossible from this start configuration. Reposition "
                      "the robot and retry.")
                sys.exit(1)

        if auto_mode and kin is not None:
            pred = predict_g_dirs(kin, q0, targets, R_base_tcp0, R_sensor_tcp, g_base)
            cond_f, cond_m = pm.pose_set_condition(pred)
            spread = pm.g_dir_spread(pred)
            print(f"Pre-flight: {len(targets)} identification poses, "
                  f"predicted conditioning force {cond_f:.1f} / torque {cond_m:.1f}, "
                  f"spread {spread:.2f}")
            if cond_m > pm.COND_TORQUE_REJECT or spread < 0.15:
                print("WARNING: the planned poses are poorly conditioned. This "
                      "usually means the tool axis points straight up/down, so "
                      "J6 rolls do not change the gravity direction.\n"
                      "Jog the robot so the tool axis is roughly HORIZONTAL and "
                      "run again.")
                if not _confirm("Continue anyway?"):
                    sys.exit(1)

        if args.dry_run:
            print("\nPlanned identification poses (deg, J4/J5/J6 offsets):")
            for off, q in targets:
                print(f"  {off}  ->  [{', '.join(f'{math.degrees(v):+.1f}' for v in q)}]")
            print("Holdout poses:")
            for off, q in holdout_targets:
                print(f"  {off}  ->  [{', '.join(f'{math.degrees(v):+.1f}' for v in q)}]")
            print("\nDry run — no motion, no capture, no write.")
            return

    # ── Sensor ───────────────────────────────────────────────────────
    collector = _SampleCollector()
    reader = FTSSensorReader(
        port=resolve_serial_port(config.param.serial_port),
        callback=collector.callback,
        auto_zero=False,          # calibration needs RAW readings
    )
    reader.start()
    time.sleep(1.0)
    if not reader.is_running or reader.frames_parsed == 0:
        print("ERROR: no sensor frames — check the serial port "
              f"({resolve_serial_port(config.param.serial_port)}).")
        reader.stop()
        sys.exit(1)
    print(f"Sensor ok: {reader.fps:.0f} frames/s")

    captures = []          # (mean_raw6, g_dir) — identification poses
    holdout_captures = []  # same — holdout poses

    try:
        if pose_names is not None:
            n_total = len(pose_names)
            print(f"\n{n_total} taught poses: {n_total - 1} for the fit, "
                  f"'{pose_names[-1]}' as the holdout verification pose.")
            if named_motion:
                print(f"The robot will move to each pose at speed "
                      f"{args.speed:.0f}. Make sure the workspace is clear "
                      "and NOBODY touches the tool during captures.")
                if not _confirm("Start the calibration motion?"):
                    sys.exit(1)

            for idx, name in enumerate(pose_names):
                is_holdout = idx == n_total - 1
                label = (f"{'holdout' if is_holdout else 'pose'} "
                         f"{idx + 1}/{n_total} '{name}'")
                if named_motion:
                    input(f"\n{label}: press Enter to move ")
                    robot.move_joint(name, speed=args.speed)
                    if not wait_until_stationary(robot):
                        abort_with_motion_warning(robot, label)
                else:
                    input(f"\n{label}: run/jog the robot to pose '{name}' "
                          "from the pendant, hands off, then press Enter ")

                # Guard against a silently rejected/aborted motion: the
                # robot must actually be AT the taught pose it claims.
                if fetched_poses is not None:
                    dpos, dang = pose_arrival_error(robot, fetched_poses[name])
                    if dpos > POSE_ARRIVAL_POS_TOL or dang > POSE_ARRIVAL_ROT_TOL:
                        print(f"  {label}: robot is {dpos * 1e3:.0f} mm / "
                              f"{math.degrees(dang):.1f} deg away from the "
                              "taught pose — was the motion rejected?")
                        if not _confirm("  Capture here anyway?"):
                            sys.exit(1)

                cap = capture_pose(robot, collector, R_sensor_tcp, g_base, label)
                if cap is not None:
                    (holdout_captures if is_holdout else captures).append(cap)

            if named_motion:
                input(f"\nReturning to '{pose_names[0]}' — press Enter ")
                robot.move_joint(pose_names[0], speed=args.speed)
                if not wait_until_stationary(robot):
                    abort_with_motion_warning(robot, "return to start")
        elif auto_mode:
            max_delta = max(
                abs(q[j] - q0[j])
                for _, q in targets + holdout_targets for j in (3, 4, 5)
            )
            print(f"\nThe robot will move through "
                  f"{len(targets) + len(holdout_targets)} wrist poses "
                  f"(max joint delta {math.degrees(max_delta):.0f} deg) at "
                  f"speed {args.speed:.0f}, then return to the start pose.")
            print("Make sure the workspace is clear and NOBODY touches the tool "
                  "during captures.")
            if not _confirm("Start the calibration motion?"):
                sys.exit(1)

            for kind, tgts, store in (("pose", targets, captures),
                                      ("holdout", holdout_targets, holdout_captures)):
                for i, (off, q) in enumerate(tgts):
                    label = f"{kind} {i + 1}/{len(tgts)} {off}"
                    input(f"\n{label}: press Enter to move ")
                    robot.move_joint(list(q), speed=args.speed)
                    if not wait_until_stationary(robot, target=list(q)):
                        abort_with_motion_warning(robot, label)
                    cap = capture_pose(robot, collector, R_sensor_tcp, g_base, label)
                    if cap is not None:
                        store.append(cap)

            input("\nReturning to the start pose — press Enter ")
            robot.move_joint(list(q0), speed=args.speed)
            if not wait_until_stationary(robot, target=list(q0)):
                abort_with_motion_warning(robot, "return to start")
        else:
            print("\nManual mode: jog the robot to each orientation, keep "
                  "hands off, and press Enter to capture.")
            print("Suggested orientations (tool axis): +X up, -X up, +Y up, "
                  "-Y up, Z up, Z down, plus one diagonal.")
            n_ident = max(3, len(offsets))
            for i in range(n_ident):
                input(f"\npose {i + 1}/{n_ident}: position the robot, hands off, "
                      "Enter to capture ")
                cap = capture_pose(robot, collector, R_sensor_tcp, g_base,
                                   f"pose {i + 1}")
                if cap is not None:
                    captures.append(cap)
            for i in range(len(holdout_offsets)):
                input(f"\nholdout {i + 1}/{len(holdout_offsets)}: NEW orientation, "
                      "hands off, Enter to capture ")
                cap = capture_pose(robot, collector, R_sensor_tcp, g_base,
                                   f"holdout {i + 1}")
                if cap is not None:
                    holdout_captures.append(cap)
    finally:
        reader.stop()

    # ── Identify ─────────────────────────────────────────────────────
    if len(captures) < 3:
        print(f"\nERROR: only {len(captures)} usable poses — need >= 3.")
        sys.exit(1)

    g_dirs = [c[1] for c in captures]
    wrenches = [c[0] for c in captures]
    result = pm.identify_payload(g_dirs, wrenches, gravity=g_mag)
    status, messages = pm.assess_identification(result)

    holdout_rows = []
    # Non-vacuous: zero successful holdout captures means the model was
    # never verified outside the fit — that is a FAIL, not a pass.
    holdout_ok = len(holdout_captures) > 0
    if not holdout_captures:
        print("\nWARNING: no holdout pose was captured — the identification "
              "cannot be verified outside its own fit.")
    for mean, g_dir in holdout_captures:
        pred = pm.predicted_wrench(result.params, g_dir)
        df = float(np.linalg.norm(mean[:3] - pred[:3]))
        dm = float(np.linalg.norm(mean[3:] - pred[3:]))
        ok = df <= HOLDOUT_MAX_F and dm <= HOLDOUT_MAX_M
        holdout_ok = holdout_ok and ok
        holdout_rows.append((df, dm, ok))

    print_report(result, status, messages, holdout_rows)

    if status == "REJECT":
        print("\nIdentification REJECTED — nothing written. Fix the setup "
              "(mount transform, cabling, quiet captures) and rerun.")
        sys.exit(1)
    if not holdout_ok:
        print("\nHoldout verification FAILED — the model does not generalise "
              "to orientations outside the fit. Check sensor_offset "
              "(sensor→TCP rotation) and the TCP configuration. Nothing written.")
        sys.exit(1)
    if status == "WARN" and not args.no_write:
        if not _confirm("Identification has warnings — write it anyway?"):
            print("Nothing written.")
            sys.exit(1)

    if args.no_write:
        print("\n--no-write: config untouched.")
        return

    profile_dict = pm.profile_to_dict(
        result, identified_on=datetime.now().isoformat()[:19]
    )
    write_profile_to_config(args.config, args.profile, profile_dict)
    print(f"\nProfile 'profile_{args.profile}' written to {args.config}.")
    print("Flange mode is NOT armed by this script: review the "
          "force_to_velocity dead-zones and payload.* accel limits against "
          "the flange-mode floors (see FTS_Free_DriveConfig.py), then set "
          "payload.enabled to true yourself.")


if __name__ == "__main__":
    main()
