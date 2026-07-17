"""
Unit tests for payload_model.py (and, further down, the extended config
loader).  Pure stdlib unittest + numpy — no hardware, no robot.

Run:
    python -m unittest test_payload_model -v
"""

import json
import math
import os
import tempfile
import unittest

import numpy as np

import payload_model as pm


def _make_measurements(truth, dirs, rng=None, noise_std=0.0, quantise=0.0,
                       n_samples=1):
    """Synthesise averaged raw readings from a ground-truth model."""
    wrenches = []
    for g in dirs:
        w = pm.predicted_wrench(truth, g)
        if rng is not None and (noise_std > 0.0 or quantise > 0.0):
            samples = w[None, :] + rng.normal(0.0, noise_std, size=(n_samples, 6))
            if quantise > 0.0:
                samples = np.round(samples / quantise) * quantise
            w = samples.mean(axis=0)
        wrenches.append(w)
    return wrenches


WELL_SPREAD_DIRS = [
    np.array([0.0, 0.0, -1.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0),
    np.array([-1.0, 1.0, -1.0]) / math.sqrt(3.0),
]

TRUTH = pm.PayloadParams(
    mass_kg=2.4,
    com=np.array([0.003, -0.012, 0.065]),
    bias_f=np.array([1.2, -0.4, 3.1]),
    bias_m=np.array([0.05, -0.02, 0.01]),
    gravity=pm.GRAVITY,
)


class TestIdentification(unittest.TestCase):

    def test_exact_round_trip(self):
        """Noise-free measurements recover the truth to machine precision."""
        wrenches = _make_measurements(TRUTH, WELL_SPREAD_DIRS)
        res = pm.identify_payload(WELL_SPREAD_DIRS, wrenches)
        p = res.params
        self.assertAlmostEqual(p.mass_kg, TRUTH.mass_kg, places=9)
        np.testing.assert_allclose(p.com, TRUTH.com, atol=1e-9)
        np.testing.assert_allclose(p.bias_f, TRUTH.bias_f, atol=1e-9)
        np.testing.assert_allclose(p.bias_m, TRUTH.bias_m, atol=1e-9)
        self.assertLess(res.force_residual_rms, 1e-9)
        self.assertLess(res.torque_residual_rms, 1e-9)

    def test_noisy_quantised_round_trip(self):
        """Sensor-realistic noise + 0.1 LSB quantisation, 2000-sample average."""
        rng = np.random.RandomState(7)
        wrenches = _make_measurements(TRUTH, WELL_SPREAD_DIRS, rng,
                                      noise_std=0.05, quantise=0.1,
                                      n_samples=2000)
        res = pm.identify_payload(WELL_SPREAD_DIRS, wrenches)
        p = res.params
        self.assertLess(abs(p.mass_kg - TRUTH.mass_kg), 0.05)
        self.assertLess(np.linalg.norm(p.com - TRUTH.com), 0.005)
        self.assertLess(np.linalg.norm(p.bias_f - TRUTH.bias_f), 0.1)
        self.assertLess(np.linalg.norm(p.bias_m - TRUTH.bias_m), 0.03)
        status, messages = pm.assess_identification(res)
        self.assertEqual(status, "PASS", messages)

    def test_minimum_three_poses(self):
        """3 distinct non-coplanar directions is the theoretical minimum."""
        dirs = WELL_SPREAD_DIRS[:1] + WELL_SPREAD_DIRS[2:3] + WELL_SPREAD_DIRS[4:5]
        wrenches = _make_measurements(TRUTH, dirs)
        res = pm.identify_payload(dirs, wrenches)
        self.assertAlmostEqual(res.params.mass_kg, TRUTH.mass_kg, places=6)
        np.testing.assert_allclose(res.params.com, TRUTH.com, atol=1e-6)

    def test_too_few_poses_raises(self):
        dirs = WELL_SPREAD_DIRS[:2]
        wrenches = _make_measurements(TRUTH, dirs)
        with self.assertRaises(ValueError):
            pm.identify_payload(dirs, wrenches)

    def test_identical_directions_raise(self):
        dirs = [WELL_SPREAD_DIRS[0]] * 5
        wrenches = _make_measurements(TRUTH, dirs)
        with self.assertRaises(ValueError):
            pm.identify_payload(dirs, wrenches)

    def test_two_distinct_directions_raise(self):
        """Two distinct directions leave the torque block rank deficient."""
        dirs = [WELL_SPREAD_DIRS[0], WELL_SPREAD_DIRS[2],
                WELL_SPREAD_DIRS[0], WELL_SPREAD_DIRS[2]]
        wrenches = _make_measurements(TRUTH, dirs)
        with self.assertRaises(ValueError):
            pm.identify_payload(dirs, wrenches)

    def test_near_collinear_poor_conditioning(self):
        """Almost-identical directions must at least fire a quality warning."""
        base = np.array([0.0, 0.0, -1.0])
        eps = 1e-3
        dirs = [
            base,
            base + np.array([eps, 0.0, 0.0]),
            base + np.array([0.0, eps, 0.0]),
        ]
        dirs = [d / np.linalg.norm(d) for d in dirs]
        wrenches = _make_measurements(TRUTH, dirs)
        try:
            res = pm.identify_payload(dirs, wrenches)
        except ValueError:
            return  # rank-deficiency rejection is equally acceptable
        status, _ = pm.assess_identification(res)
        self.assertIn(status, ("WARN", "REJECT"))

    def test_length_mismatch_raises(self):
        wrenches = _make_measurements(TRUTH, WELL_SPREAD_DIRS)
        with self.assertRaises(ValueError):
            pm.identify_payload(WELL_SPREAD_DIRS[:-1], wrenches)

    def test_negligible_mass_zeroes_com(self):
        """m below MIN_MASS_FOR_COM must not blow up the COM estimate."""
        truth = TRUTH._replace(mass_kg=0.01)
        wrenches = _make_measurements(truth, WELL_SPREAD_DIRS)
        res = pm.identify_payload(WELL_SPREAD_DIRS, wrenches)
        np.testing.assert_allclose(res.params.com, np.zeros(3), atol=1e-12)


class TestGravityModel(unittest.TestCase):

    def test_cross_product_sign(self):
        """com = +z, gravity along −x ⇒ My = −m·g·d (guards the convention)."""
        d = 0.06
        m = 2.0
        w = pm.gravity_wrench(m, np.array([0.0, 0.0, d]),
                              np.array([-1.0, 0.0, 0.0]))
        self.assertAlmostEqual(w[0], -m * pm.GRAVITY)           # Fx
        self.assertAlmostEqual(w[4], -m * pm.GRAVITY * d)       # My
        self.assertAlmostEqual(w[3], 0.0)                       # Mx
        self.assertAlmostEqual(w[5], 0.0)                       # Mz

    def test_skew_matches_cross(self):
        rng = np.random.RandomState(3)
        for _ in range(10):
            a, b = rng.normal(size=3), rng.normal(size=3)
            np.testing.assert_allclose(pm.skew(a) @ b, np.cross(a, b),
                                       atol=1e-12)

    def test_gravity_dir_identity(self):
        g = pm.gravity_dir_sensor(np.eye(3))
        np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-12)

    def test_gravity_dir_normalises(self):
        g = pm.gravity_dir_sensor(np.eye(3), g_base=(0.0, 0.0, -9.81))
        self.assertAlmostEqual(np.linalg.norm(g), 1.0, places=12)

    def test_gravity_dir_rotated(self):
        """Sensor x-axis pointing straight down ⇒ gravity along sensor +x."""
        # R_base_sensor with sensor x = base -z, sensor z = base x
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        g = pm.gravity_dir_sensor(R)
        np.testing.assert_allclose(g, [1.0, 0.0, 0.0], atol=1e-12)

    def test_predicted_wrench_includes_bias(self):
        g = np.array([0.0, 0.0, -1.0])
        w = pm.predicted_wrench(TRUTH, g)
        grav = pm.gravity_wrench(TRUTH.mass_kg, TRUTH.com, g, TRUTH.gravity)
        np.testing.assert_allclose(w[:3], grav[:3] + TRUTH.bias_f, atol=1e-12)
        np.testing.assert_allclose(w[3:], grav[3:] + TRUTH.bias_m, atol=1e-12)


class TestConventionRoundTrip(unittest.TestCase):
    """FK flange rotation vs the ZYX-RPY convention used by get_tcp_pose."""

    def test_fk_vs_rpy(self):
        from frame_transformer import rotation_matrix_zyx
        from robot_kinematics import RobotKinematics

        kin = RobotKinematics("DS6-800")
        rng = np.random.RandomState(11)
        for _ in range(20):
            q = list(rng.uniform(-1.5, 1.5, size=6))
            R = kin.forward_kinematics(q)[-1][:3, :3]
            # Extract ZYX RPY from R (R = Rz(yaw)·Ry(pitch)·Rx(roll))
            pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
            roll = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
            R2 = np.array(rotation_matrix_zyx(roll, pitch, yaw))
            np.testing.assert_allclose(R2, R, atol=1e-9)


class TestAssessment(unittest.TestCase):

    def _clean_result(self):
        wrenches = _make_measurements(TRUTH, WELL_SPREAD_DIRS)
        return pm.identify_payload(WELL_SPREAD_DIRS, wrenches)

    def test_clean_pass(self):
        status, messages = pm.assess_identification(self._clean_result())
        self.assertEqual(status, "PASS")
        self.assertEqual(messages, [])

    def test_implausible_mass_rejects(self):
        res = self._clean_result()
        bad = res._replace(params=res.params._replace(mass_kg=80.0))
        status, messages = pm.assess_identification(bad)
        self.assertEqual(status, "REJECT")
        self.assertTrue(any("mass" in m for m in messages))

    def test_implausible_com_rejects(self):
        res = self._clean_result()
        bad = res._replace(params=res.params._replace(com=np.array([0.0, 0.0, 0.8])))
        status, messages = pm.assess_identification(bad)
        self.assertEqual(status, "REJECT")
        self.assertTrue(any("COM" in m for m in messages))

    def test_high_residual_warns_then_rejects(self):
        res = self._clean_result()
        warn = res._replace(force_residual_rms=0.2)
        self.assertEqual(pm.assess_identification(warn)[0], "WARN")
        reject = res._replace(force_residual_rms=0.5)
        self.assertEqual(pm.assess_identification(reject)[0], "REJECT")


class TestProfileBridge(unittest.TestCase):

    def test_profile_round_trip(self):
        wrenches = _make_measurements(TRUTH, WELL_SPREAD_DIRS)
        res = pm.identify_payload(WELL_SPREAD_DIRS, wrenches)
        d = pm.profile_to_dict(res, identified_on="2026-07-16T12:00:00")
        self.assertAlmostEqual(d["mass_kg"], TRUTH.mass_kg, places=3)
        self.assertEqual(d["identified_on"], "2026-07-16T12:00:00")

        class FakeProfile:
            mass_kg = d["mass_kg"]
            com_x = d["com_x"]
            com_y = d["com_y"]
            com_z = d["com_z"]

        params = pm.params_from_profile(FakeProfile)
        self.assertAlmostEqual(params.mass_kg, TRUTH.mass_kg, places=3)
        np.testing.assert_allclose(params.com, TRUTH.com, atol=1e-4)
        np.testing.assert_allclose(params.bias_f, np.zeros(3), atol=1e-12)


class TestVelocitySlewLimiter(unittest.TestCase):

    def _limiter(self, accel=1.0, ang=2.0, mult=4.0):
        from force_to_velocity import VelocitySlewLimiter
        return VelocitySlewLimiter(accel, ang, mult)

    def test_accel_clamped(self):
        s = self._limiter(accel=1.0)
        v = s.limit([1.0, 0, 0, 0, 0, 0], dt=0.001)
        self.assertAlmostEqual(v[0], 0.001)          # 1 m/s² × 1 ms
        v = s.limit([1.0, 0, 0, 0, 0, 0], dt=0.001)
        self.assertAlmostEqual(v[0], 0.002)

    def test_decel_faster(self):
        s = self._limiter(accel=1.0, mult=4.0)
        for _ in range(100):                          # ramp up to 0.1
            v = s.limit([0.1, 0, 0, 0, 0, 0], dt=0.001)
        self.assertAlmostEqual(v[0], 0.1)
        v = s.limit([0.0, 0, 0, 0, 0, 0], dt=0.001)   # release
        self.assertAlmostEqual(v[0], 0.1 - 0.004)     # 4×1 m/s² × 1 ms

    def test_release_to_stop_time(self):
        """From 0.05 m/s at accel 0.25, mult 4 ⇒ zero in 50 ms."""
        s = self._limiter(accel=0.25, mult=4.0)
        v = [0.05, 0, 0, 0, 0, 0]
        s._prev = list(v)
        ticks = 0
        while any(s._prev) and ticks < 1000:
            s.limit([0.0] * 6, dt=0.001)
            ticks += 1
        self.assertLessEqual(ticks, 51)

    def test_sign_change_splits_at_zero(self):
        """Crossing zero must not carry the decel rate into the new sign."""
        s = self._limiter(accel=1.0, mult=4.0)
        s._prev = [0.001, 0, 0, 0, 0, 0]
        v = s.limit([-1.0, 0, 0, 0, 0, 0], dt=0.001)
        # decel to zero takes 0.001/4 = 0.25 ms; remaining 0.75 ms at accel 1
        self.assertAlmostEqual(v[0], -0.00075)

    def test_angular_axes_use_angular_limit(self):
        s = self._limiter(accel=1.0, ang=2.0)
        v = s.limit([0, 0, 0, 1.0, 1.0, 1.0], dt=0.001)
        self.assertAlmostEqual(v[3], 0.002)

    def test_no_overshoot(self):
        s = self._limiter(accel=1000.0)
        v = s.limit([0.01, 0, 0, 0, 0, 0], dt=0.001)
        self.assertAlmostEqual(v[0], 0.01)            # reachable in one tick

    def test_validation(self):
        from force_to_velocity import VelocitySlewLimiter
        with self.assertRaises(ValueError):
            VelocitySlewLimiter(0.0, 1.0)
        with self.assertRaises(ValueError):
            VelocitySlewLimiter(1.0, 1.0, decel_multiplier=0.5)

    def test_reset(self):
        s = self._limiter()
        s.limit([1.0] * 6, dt=0.001)
        s.reset()
        self.assertEqual(s._prev, [0.0] * 6)


class TestConfigLoader(unittest.TestCase):
    """Backward compatibility and strictness of the extended config loader."""

    LEGACY = {
        "param": {"serial_port": "10", "auto_zero_seconds": 1.0, "io_num": 0},
        "control": {"control_space": False, "orientation_control": False},
        "sensor_offset": {"sensor_x": 0.11, "sensor_y": 0.0, "sensor_z": 0.0,
                          "sensor_a": 180.0, "sensor_b": 0.0, "sensor_c": 0.0},
        "force_to_velocity": {"max_linear_vel": 0.05, "max_angular_vel": 0.05,
                              "dead_zone_force": 0.5, "dead_zone_torque": 0.1,
                              "gain_force": 0.01, "gain_torque": 0.1},
    }

    PROFILE = {"mass_kg": 2.4, "com_x": 0.003, "com_y": -0.012, "com_z": 0.065,
               "identified_on": "2026-07-16T12:00:00",
               "residual_force_rms": 0.05, "residual_torque_rms": 0.01}

    PAYLOAD = {"enabled": True,
               "profile_tool": PROFILE,
               "profile_tool_load": dict(PROFILE, mass_kg=4.1),
               "active_profile": "tool", "switch_io": -1,
               "gravity_x": 0.0, "gravity_y": 0.0, "gravity_z": -9.80665,
               "startup_max_residual_force": 1.0,
               "startup_max_residual_torque": 0.3,
               "max_operator_force": 80.0, "max_operator_torque": 20.0,
               "max_accel_linear": 0.25, "max_accel_angular": 0.5,
               "decel_multiplier": 4.0, "inertial_feedforward": False}

    def _load(self, data):
        from FTS_Free_DriveConfig import load_config
        fd, path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            return load_config(path)
        finally:
            os.unlink(path)

    def test_legacy_config_loads_with_none_sections(self):
        cfg = self._load(self.LEGACY)
        self.assertIsNone(cfg.payload)
        self.assertIsNone(cfg.wrench_filter)
        self.assertEqual(cfg.param.serial_port, "10")
        self.assertEqual(cfg.force_to_velocity.dead_zone_force, 0.5)

    def test_shipped_legacy_config_still_loads(self):
        from FTS_Free_DriveConfig import load_config
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = load_config(os.path.join(here, "FTS_Free_Drive_config.json"))
        self.assertIsNone(cfg.payload)

    def test_flange_config_loads(self):
        data = dict(self.LEGACY)
        data["payload"] = self.PAYLOAD
        data["wrench_filter"] = {"filter_type": "butter2", "cutoff_hz": 10.0}
        cfg = self._load(data)
        self.assertTrue(cfg.payload.enabled)
        self.assertEqual(cfg.payload.profile_tool.mass_kg, 2.4)
        self.assertEqual(cfg.payload.profile_tool_load.mass_kg, 4.1)
        self.assertEqual(cfg.wrench_filter.filter_type, "butter2")

    def test_partial_payload_section_rejected(self):
        data = dict(self.LEGACY)
        data["payload"] = {"enabled": True}   # missing everything else
        with self.assertRaises(ValueError) as ctx:
            self._load(data)
        self.assertIn("payload", str(ctx.exception))

    def test_unknown_key_in_payload_rejected(self):
        data = dict(self.LEGACY)
        data["payload"] = dict(self.PAYLOAD, bogus_key=1)
        with self.assertRaises(ValueError) as ctx:
            self._load(data)
        self.assertIn("bogus_key", str(ctx.exception))

    def test_partial_profile_rejected(self):
        data = dict(self.LEGACY)
        data["payload"] = dict(self.PAYLOAD, profile_tool={"mass_kg": 1.0})
        with self.assertRaises(ValueError) as ctx:
            self._load(data)
        self.assertIn("profile_tool", str(ctx.exception))

    def test_unknown_top_level_key_still_rejected(self):
        data = dict(self.LEGACY)
        data["bogus_section"] = {}
        with self.assertRaises(ValueError) as ctx:
            self._load(data)
        self.assertIn("bogus_section", str(ctx.exception))

    def test_missing_legacy_section_still_rejected(self):
        data = dict(self.LEGACY)
        del data["control"]
        with self.assertRaises(ValueError) as ctx:
            self._load(data)
        self.assertIn("control", str(ctx.exception))

    def test_shipped_flange_example_loads(self):
        from FTS_Free_DriveConfig import load_config
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = load_config(os.path.join(here, "FTS_Free_Drive_config_flange_example.json"))
        self.assertTrue(cfg.payload.enabled)
        self.assertEqual(cfg.payload.profile_tool.identified_on, "")
        self.assertEqual(cfg.wrench_filter.filter_type, "butter2")


class TestCalibrationConfigWriter(unittest.TestCase):
    """payload_calibration.write_profile_to_config on legacy and new configs."""

    PROFILE_DICT = {"mass_kg": 2.4, "com_x": 0.003, "com_y": -0.012,
                    "com_z": 0.065, "identified_on": "2026-07-16T12:00:00",
                    "residual_force_rms": 0.05, "residual_torque_rms": 0.01}

    def _tmp_config(self, data):
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        self.addCleanup(os.unlink, path)
        return path

    def test_creates_payload_section_on_legacy_config(self):
        from payload_calibration import write_profile_to_config
        from FTS_Free_DriveConfig import load_config
        path = self._tmp_config(TestConfigLoader.LEGACY)
        write_profile_to_config(path, "tool", dict(self.PROFILE_DICT))
        cfg = load_config(path)
        # Calibration must NOT arm flange mode as a side effect — enabling
        # it is an operator decision after reviewing the dead-zones.
        self.assertFalse(cfg.payload.enabled)
        self.assertEqual(cfg.payload.active_profile, "tool")
        self.assertAlmostEqual(cfg.payload.profile_tool.mass_kg, 2.4)
        self.assertEqual(cfg.payload.profile_tool_load.identified_on, "")

    def test_updates_only_the_named_profile(self):
        from payload_calibration import write_profile_to_config
        from FTS_Free_DriveConfig import load_config
        data = dict(TestConfigLoader.LEGACY)
        data["payload"] = dict(TestConfigLoader.PAYLOAD, switch_io=7)
        path = self._tmp_config(data)
        write_profile_to_config(path, "tool_load", dict(self.PROFILE_DICT,
                                                        mass_kg=4.4))
        cfg = load_config(path)
        self.assertAlmostEqual(cfg.payload.profile_tool_load.mass_kg, 4.4)
        # Untouched: the other profile and the surrounding settings.
        self.assertAlmostEqual(cfg.payload.profile_tool.mass_kg, 2.4)
        self.assertEqual(cfg.payload.switch_io, 7)

    def test_never_corrupts_on_invalid_result(self):
        """If the rewritten config would not parse, the original survives."""
        from payload_calibration import write_profile_to_config
        path = self._tmp_config(TestConfigLoader.LEGACY)
        with open(path) as f:
            original = f.read()
        with self.assertRaises(Exception):
            # A profile dict with a surplus key must fail load_config
            # validation inside the writer and leave the file untouched.
            write_profile_to_config(path, "tool",
                                    dict(self.PROFILE_DICT, bogus=1))
        with open(path) as f:
            self.assertEqual(f.read(), original)


if __name__ == "__main__":
    unittest.main()
