"""
End-to-end integration tests for the flange-mounted mode of FTS_Free_Drive.

Runs the REAL 1 kHz control loop against a simulated robot and a simulated
sensor whose raw readings follow the physical model

    raw = bias + gravity_wrench(truth, orientation) + operator

so the whole chain — blind tare, delta-gravity compensation, filtering,
dead-zone, slew limiting, verify phases, profile switching, stop paths — is
exercised exactly as on hardware, minus the serial port and RPC transport.

These tests run in real time (the loop paces itself); the full module takes
roughly 15-20 s.

Run:
    python -m unittest test_flange_integration -v
"""

import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest

import numpy as np


def setUpModule():
    # On Windows, time.sleep() granularity defaults to the ~15.6 ms system
    # tick on Python 3.6, which would run the 1 kHz loop at ~64 Hz and make
    # the timing assertions meaningless.  Request the 1 ms multimedia timer
    # so the loop paces close to the production (Linux controller) rate.
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)


def tearDownModule():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.winmm.timeEndPeriod(1)

import payload_model as pm
from FTS_Free_Drive import FTS_Free_Drive
from frame_transformer import rotation_matrix_zyx

G_BASE = np.array([0.0, 0.0, -9.80665])

TOOL_TRUTH = pm.PayloadParams(
    mass_kg=2.0, com=np.array([0.0, 0.0, 0.05]),
    bias_f=np.array([1.5, -0.7, 2.2]), bias_m=np.array([0.04, -0.01, 0.02]),
    gravity=9.80665,
)
TOOL_LOAD_TRUTH = TOOL_TRUTH._replace(
    mass_kg=4.0, com=np.array([0.0, 0.0, 0.10])
)


class FakePhysics:
    """Ground truth: robot orientation + what physically hangs on the sensor."""

    def __init__(self):
        self.rpy = [0.0, 0.0, 0.0]
        self.truth = TOOL_TRUTH
        self.operator = np.zeros(6)     # operator wrench, sensor frame
        self.R_sensor_tcp = np.eye(3)   # test configs use zero sensor rotation

    def raw(self):
        R_base_tcp = np.array(rotation_matrix_zyx(*self.rpy))
        g_dir = pm.gravity_dir_sensor(R_base_tcp @ self.R_sensor_tcp, G_BASE)
        return pm.predicted_wrench(self.truth, g_dir) + self.operator


class FakeReader:
    """Duck-type of FTSSensorReader with the blind tare semantics."""

    def __init__(self, physics):
        self.physics = physics
        self._tare = None
        self._running = False

    def start(self):
        self._tare = self.physics.raw().copy()
        self._running = True

    def stop(self, join_timeout=None):
        self._running = False

    @property
    def is_running(self):
        return self._running

    @property
    def zero_done(self):
        return self._tare is not None

    def get_latest_with_age(self):
        return tuple(self.physics.raw() - self._tare), 0.0


class FakeRobot:
    """Minimal RPC surface used by FTS_Free_Drive."""

    robot_name = "ds6-800"

    def __init__(self, physics):
        self.physics = physics
        self.pins = {0: 0}
        self.cmds = []      # (perf_counter, velocity list) from speed_x
        self.msgs = []      # (level, text) from notify_info/notify_error

    # -- state --
    def get_tcp_pose(self):
        return [0.5, 0.0, 0.5] + list(self.physics.rpy)

    def get_current_joint_angles(self):
        return [0.0] * 6

    def get_digital_input(self, pin):
        return self.pins.get(pin, 0)

    # -- motion --
    def speed_x(self, velocity, accel):
        self.cmds.append((time.perf_counter(), list(velocity)))

    def speed_j(self, velocity, accel):
        self.cmds.append((time.perf_counter(), list(velocity)))

    # -- housekeeping --
    def notify_info(self, msg):
        self.msgs.append(("info", str(msg)))

    def notify_error(self, msg):
        self.msgs.append(("error", str(msg)))

    def disable_collision_detection(self):
        pass

    def enable_collision_detection(self):
        pass

    def enable_reflex(self):
        pass

    def activate_servo_interface(self, mode):
        pass

    def deactivate_servo_interface(self):
        pass


def make_config(flange=True, switch_io=-1, active_profile="tool",
                sensor_z=0.0, orientation=False):
    profile_tool = {
        "mass_kg": TOOL_TRUTH.mass_kg, "com_x": 0.0, "com_y": 0.0, "com_z": 0.05,
        "identified_on": "2026-07-16T00:00:00",
        "residual_force_rms": 0.02, "residual_torque_rms": 0.005,
    }
    profile_tool_load = {
        "mass_kg": TOOL_LOAD_TRUTH.mass_kg, "com_x": 0.0, "com_y": 0.0, "com_z": 0.10,
        "identified_on": "2026-07-16T00:00:00",
        "residual_force_rms": 0.02, "residual_torque_rms": 0.005,
    }
    data = {
        "param": {"serial_port": "10", "auto_zero_seconds": 0.1, "io_num": 0},
        "control": {"control_space": False, "orientation_control": orientation},
        "sensor_offset": {"sensor_x": 0.0, "sensor_y": 0.0, "sensor_z": sensor_z,
                          "sensor_a": 0.0, "sensor_b": 0.0, "sensor_c": 0.0},
        "force_to_velocity": {"max_linear_vel": 0.05, "max_angular_vel": 0.05,
                              "dead_zone_force": 1.0, "dead_zone_torque": 0.5,
                              "gain_force": 0.01, "gain_torque": 0.1},
    }
    if flange:
        data["payload"] = {
            "enabled": True,
            "profile_tool": profile_tool,
            "profile_tool_load": profile_tool_load,
            "active_profile": active_profile,
            "switch_io": switch_io,
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
        data["wrench_filter"] = {"filter_type": "butter2", "cutoff_hz": 10.0}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


class FlangeIntegrationBase(unittest.TestCase):

    def setUp(self):
        self.physics = FakePhysics()
        self.app = None
        self.thread = None
        self.config_path = None

    def tearDown(self):
        if self.app is not None:
            self.app.finish()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
            self.assertFalse(self.thread.is_alive(), "control loop did not exit")
        if self.config_path:
            os.unlink(self.config_path)

    def start_app(self, pins=None, reader_cls=FakeReader, **config_kwargs):
        self.config_path = make_config(**config_kwargs)
        self.robot = FakeRobot(self.physics)
        if pins:
            self.robot.pins.update(pins)
        self.app = FTS_Free_Drive(self.robot, self.config_path)
        self.app.init()
        self.app._reader = reader_cls(self.physics)  # replace the serial reader
        self.thread = threading.Thread(target=self.app.run, daemon=True)
        self.thread.start()

    def wait_for_msg(self, fragment, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if any(fragment in text for _, text in self.robot.msgs):
                return True
            time.sleep(0.005)
        return False

    def wait_for_stop(self, timeout=8.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.app._stop_event.is_set():
                return True
            time.sleep(0.01)
        return False

    def cmds_since(self, t):
        return [v for (ts, v) in self.robot.cmds if ts >= t]

    @staticmethod
    def all_zero(cmds):
        return all(all(abs(x) < 1e-12 for x in v) for v in cmds)


class TestFlangeMode(FlangeIntegrationBase):

    def test_no_drift_across_orientation_change(self):
        """Rotating the wrist hands-off must not move the robot."""
        self.physics.truth = TOOL_TRUTH
        self.start_app(flange=True)
        self.assertTrue(self.wait_for_msg("hand guiding active"),
                        self.robot.msgs)

        # Step the orientation far from the tare pose, hands off.  (A step
        # is unphysically fast — real guiding rotates at <= 0.05 rad/s — so
        # skip the pose-refresh latency + filter transient before asserting.)
        t = time.perf_counter()
        self.physics.rpy = [0.3, 0.6, 0.2]
        time.sleep(0.6)
        cmds = self.cmds_since(t + 0.2)
        self.assertTrue(len(cmds) > 50, f"only {len(cmds)} commands in 0.4 s")
        self.assertTrue(self.all_zero(cmds),
                        f"drift detected: max {max(max(abs(x) for x in v) for v in cmds)}")

    def test_push_moves_release_stops(self):
        self.start_app(flange=True)
        self.assertTrue(self.wait_for_msg("hand guiding active"))

        # Push 5 N along sensor x — well above the 1 N dead-zone.
        self.physics.operator = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(0.4)
        t = time.perf_counter()
        moving = self.cmds_since(t - 0.1)
        self.assertTrue(any(abs(v[0]) > 1e-3 for v in moving),
                        "no motion despite 5 N push")
        # +x push ⇒ +x velocity (identity orientation, identity mount)
        self.assertTrue(all(v[0] >= 0.0 for v in moving))

        # Release: must decay to exactly zero (slew-limited; < 150 ms at the
        # nominal 1 kHz — allow slack for timer jitter on the test host).
        self.physics.operator = np.zeros(6)
        time.sleep(0.6)
        tail = self.cmds_since(time.perf_counter() - 0.1)
        self.assertTrue(self.all_zero(tail),
                        "still moving after release: "
                        f"{[v for v in tail if any(v)][:3]}")

    def test_operator_touch_during_verify_stops(self):
        """A wrench appearing inside the verify window must abort startup."""
        self.start_app(flange=True)
        self.assertTrue(self.wait_for_msg("verifying residual"))
        self.physics.operator = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(self.wait_for_stop())
        self.assertTrue(str(self.app._stop_reason).startswith("tare_residual"),
                        self.app._stop_reason)
        # Motion must never have been enabled: every command was zero.
        self.assertTrue(self.all_zero([v for _, v in self.robot.cmds]))

    def test_profile_switch_on_load_release(self):
        """DI toggles as the load is released: swap, settle, resume, no drift."""
        self.physics.truth = TOOL_LOAD_TRUTH
        # Pin HIGH from the start: run() reads it before the tare and picks
        # the tool_load profile, matching the physical state.
        self.start_app(flange=True, switch_io=5, active_profile="tool",
                       pins={5: 1})
        self.assertTrue(self.wait_for_msg("Profile from switch input"),
                        self.robot.msgs)
        self.assertTrue(self.wait_for_msg("hand guiding active"), self.robot.msgs)
        self.assertEqual(self.app._active_profile, "tool_load")

        # Release the load: physical truth and pin change together.
        t = time.perf_counter()
        self.physics.truth = TOOL_TRUTH
        self.robot.pins[5] = 0
        self.assertTrue(self.wait_for_msg("Profile 'tool' verified", timeout=3.0),
                        self.robot.msgs)

        # Transient while the watcher caught up (≤ 50 ms) was slew-limited.
        during = self.cmds_since(t)
        max_v = max((max(abs(x) for x in v) for v in during), default=0.0)
        self.assertLess(max_v, 0.03, "switch transient exceeded the slew budget")

        time.sleep(0.3)
        tail = self.cmds_since(time.perf_counter() - 0.1)
        self.assertTrue(self.all_zero(tail), "drift after profile switch")

    def test_lying_switch_pin_stops(self):
        """Pin claims tool_load but the load is not there ⇒ verify timeout."""
        self.physics.truth = TOOL_TRUTH
        self.start_app(flange=True, switch_io=5, active_profile="tool")
        self.robot.pins[5] = 0
        self.assertTrue(self.wait_for_msg("hand guiding active"))

        self.robot.pins[5] = 1          # lie: nothing was attached
        self.assertTrue(self.wait_for_stop(timeout=8.0))
        self.assertEqual(self.app._stop_reason, "profile_switch_verify_failed")
        # While waiting for verification the robot held still.
        tail = [v for ts, v in self.robot.cmds if ts > time.perf_counter() - 4.0]
        self.assertTrue(self.all_zero(tail))

    def test_chattering_switch_pin_still_times_out(self):
        """A bouncing switch contact must not reset the 5 s give-up forever."""
        self.physics.truth = TOOL_TRUTH
        self.start_app(flange=True, switch_io=5, active_profile="tool")
        self.robot.pins[5] = 0
        self.assertTrue(self.wait_for_msg("hand guiding active"))

        # Chatter at ~7 Hz with the physical load state constant: the
        # residual never settles for either profile, and the timeout must
        # still fire (anchored to the FIRST profile_verify entry).
        deadline = time.time() + 7.5
        state = 1
        while time.time() < deadline and not self.app._stop_event.is_set():
            self.robot.pins[5] = state
            state = 1 - state
            time.sleep(0.15)
        self.assertTrue(self.app._stop_event.is_set(),
                        "chattering pin livelocked the phase machine")
        self.assertEqual(self.app._stop_reason, "profile_switch_verify_failed")

    def test_pure_force_at_tcp_produces_no_rotation(self):
        """Lever-arm transport regression: a force applied AT the TCP has
        zero moment about the TCP — the sensor's lever-arm torque must
        cancel, not double (sign of the p × f term)."""
        self.start_app(flange=True, sensor_z=0.15, orientation=True)
        self.assertTrue(self.wait_for_msg("hand guiding active"))

        # 10 N along sensor x applied at the TCP: about the SENSOR origin
        # this produces m_s = p × F with p = (0, 0, 0.15).
        F = np.array([10.0, 0.0, 0.0])
        p = np.array([0.0, 0.0, 0.15])
        self.physics.operator = np.concatenate((F, np.cross(p, F)))
        time.sleep(0.4)
        cmds = self.cmds_since(time.perf_counter() - 0.2)
        self.assertTrue(any(v[0] > 1e-3 for v in cmds), "no translation")
        max_ang = max(max(abs(v[i]) for i in (3, 4, 5)) for v in cmds)
        self.assertLess(max_ang, 1e-9,
                        f"phantom rotation commanded: {max_ang}")

    def test_reader_death_during_zero_wait_stops(self):
        """A reader that dies before zero_done must stop the app, not hang it."""

        class DeadReader(FakeReader):
            def start(self):
                self._running = False   # thread died instantly

            @property
            def zero_done(self):
                return False

        self.start_app(flange=True, reader_cls=DeadReader)
        self.assertTrue(self.wait_for_stop(timeout=3.0))
        self.assertEqual(self.app._stop_reason, "reader_thread_died")
        self.assertTrue(self.all_zero([v for _, v in self.robot.cmds]))


class TestHandleModeRegression(FlangeIntegrationBase):
    """Legacy config: no payload section — behavior as before the extension."""

    def test_handle_mode_push_and_release(self):
        # Handle rig: nothing heavy on the sensor → gravity-free raw signal.
        self.physics.truth = TOOL_TRUTH._replace(
            mass_kg=0.0, com=np.zeros(3))
        self.start_app(flange=False)
        time.sleep(0.3)

        # No flange-mode messages, no verify phase.
        self.assertFalse(any("Tare captured" in m for _, m in self.robot.msgs))
        self.assertFalse(self.app.flange_mode)

        self.physics.operator = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(0.3)
        self.assertTrue(any(abs(v[0]) > 1e-3 for _, v in self.robot.cmds))

        self.physics.operator = np.zeros(6)
        time.sleep(0.2)
        tail = self.cmds_since(time.perf_counter() - 0.05)
        self.assertTrue(self.all_zero(tail))


if __name__ == "__main__":
    unittest.main()
