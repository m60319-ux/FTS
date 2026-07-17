"""
Payload model and least-squares identification for a flange-mounted FTS.

When the force-torque sensor is bolted to the robot flange with the tool on
its measurement side, a static reading in the sensor frame is

    F_raw = F0 + m·g·ĝ_s                    (force channels)
    M_raw = M0 + r_com × (m·g·ĝ_s)          (torque channels)

where F0/M0 is the intrinsic sensor bias, m the tool(+load) mass, r_com the
centre of mass in the sensor frame and ĝ_s the unit gravity direction in the
sensor frame.  Both equations are linear in the unknowns:

    force block:   [ g·ĝ_i | I₃ ] · [m, F0]ᵀ        = F_raw,i
    torque block:  [ −g·[ĝ_i]× | I₃ ] · [c, M0]ᵀ    = M_raw,i ,  c := m·r_com

so N static poses with distinct gravity directions give two small stacked
least-squares problems (≥ 3 distinct directions for full rank).

This module is pure math on numpy arrays — no I/O, no robot, no serial — so
it is unit-testable and shared between the calibration script
(payload_calibration.py) and the runtime compensator (gravity_compensator.py).

Conventions
-----------
g_dir    unit gravity direction expressed in the SENSOR frame (points "down")
gravity  gravitational acceleration magnitude (m/s², > 0)
com      centre of mass in the sensor frame, metres, from the sensor origin
wrench   6-vector [Fx, Fy, Fz, Mx, My, Mz] (N, Nm), sensor frame
"""

import math
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np


GRAVITY = 9.80665  # standard gravity (m/s²)

# ── Quality thresholds (anchored to the sensor's 0.1 N / 0.1 Nm LSB) ─
COND_FORCE_WARN = 20.0
COND_FORCE_REJECT = 50.0
COND_TORQUE_WARN = 30.0
COND_TORQUE_REJECT = 100.0
FORCE_RMS_WARN = 0.15        # N
FORCE_RMS_REJECT = 0.30      # N  (3× the 0.1 N LSB)
TORQUE_RMS_WARN = 0.05       # Nm
TORQUE_RMS_REJECT = 0.10     # Nm (1 LSB — torque LSB is coarse)
MASS_MAX = 50.0              # kg — plausibility bound
COM_NORM_MAX = 0.5           # m  — plausibility bound
G_SPREAD_WARN = 0.3          # min singular value of ĝ set, √N-normalised
POSE_OUTLIER_FACTOR = 3.0    # per-pose residual > this × RMS → flagged
MIN_MASS_FOR_COM = 0.05      # kg — below this, com is numerically meaningless


class PayloadParams(NamedTuple):
    mass_kg: float       # tool(+load) mass
    com: np.ndarray      # (3,) centre of mass, sensor frame (m)
    bias_f: np.ndarray   # (3,) sensor force bias F0 (N) — session-dependent
    bias_m: np.ndarray   # (3,) sensor torque bias M0 (Nm)
    gravity: float       # gravity magnitude used during identification (m/s²)


class IdentificationResult(NamedTuple):
    params: PayloadParams
    force_residual_rms: float          # N,  over all 3N force components
    torque_residual_rms: float         # Nm, over all 3N torque components
    cond_force: float                  # of the unit-normalised force regressor
    cond_torque: float                 # of the unit-normalised torque regressor
    g_spread: float                    # σ_min(stacked ĝ) / √N — direction coverage
    pose_force_residuals: np.ndarray   # (N,) per-pose ‖ΔF‖
    pose_torque_residuals: np.ndarray  # (N,) per-pose ‖ΔM‖


# ── Elementary helpers ───────────────────────────────────────────────


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric cross-product matrix: skew(a) @ b == np.cross(a, b)."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def gravity_dir_sensor(R_base_sensor: np.ndarray,
                       g_base: Sequence[float] = (0.0, 0.0, -1.0)) -> np.ndarray:
    """Unit gravity direction in the sensor frame.

    Parameters
    ----------
    R_base_sensor : np.ndarray, shape (3, 3)
        Rotation mapping sensor-frame vectors into the base frame
        (columns are the sensor axes expressed in base).
    g_base : sequence of 3 floats
        Gravity direction in the base frame.  Default (0, 0, -1) for a
        floor-mounted robot; pass the configured vector for wall/ceiling
        mounts.  Magnitude is ignored — the result is normalised.
    """
    g = np.asarray(g_base, dtype=float)
    norm = np.linalg.norm(g)
    if norm < 1e-12:
        raise ValueError("g_base must be a non-zero vector")
    return np.asarray(R_base_sensor, dtype=float).T @ (g / norm)


def gravity_wrench(mass_kg: float, com: np.ndarray, g_dir: np.ndarray,
                   gravity: float = GRAVITY) -> np.ndarray:
    """Gravity wrench of the payload in the sensor frame (no bias).

    Returns (6,) = [m·g·ĝ, com × (m·g·ĝ)].
    """
    f = mass_kg * gravity * np.asarray(g_dir, dtype=float)
    m = np.cross(np.asarray(com, dtype=float), f)
    return np.concatenate((f, m))


def predicted_wrench(params: PayloadParams, g_dir: np.ndarray) -> np.ndarray:
    """Predicted RAW sensor reading (bias + gravity) at orientation g_dir."""
    w = gravity_wrench(params.mass_kg, params.com, g_dir, params.gravity)
    w[:3] += params.bias_f
    w[3:] += params.bias_m
    return w


# ── Identification ───────────────────────────────────────────────────


def _stacked_regressors(g_dirs: List[np.ndarray],
                        gravity: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build the stacked (3N×4) force and (3N×6) torque regressor matrices."""
    n = len(g_dirs)
    eye = np.eye(3)
    A_f = np.zeros((3 * n, 4))
    A_m = np.zeros((3 * n, 6))
    for i, g in enumerate(g_dirs):
        r = slice(3 * i, 3 * i + 3)
        A_f[r, 0] = gravity * g
        A_f[r, 1:4] = eye
        A_m[r, 0:3] = -gravity * skew(g)
        A_m[r, 3:6] = eye
    return A_f, A_m


def pose_set_condition(g_dirs: Sequence[np.ndarray]) -> Tuple[float, float]:
    """Condition numbers of the unit-normalised regressors for a pose set.

    Needs only the gravity directions — no measurements — so the calibration
    script can evaluate a planned pose set before any motion.

    Returns
    -------
    (cond_force, cond_torque)
    """
    dirs = [np.asarray(g, dtype=float) for g in g_dirs]
    A_f, A_m = _stacked_regressors(dirs, gravity=1.0)
    return float(np.linalg.cond(A_f)), float(np.linalg.cond(A_m))


def g_dir_spread(g_dirs: Sequence[np.ndarray]) -> float:
    """Direction-coverage metric: σ_min of the stacked ĝ matrix, √N-normalised.

    0 when all directions are coplanar through the origin's normal (rank
    deficient); the theoretical maximum 1/√3 ≈ 0.577 for perfectly isotropic
    coverage.  Below ~0.3 the identification is poorly conditioned.
    """
    G = np.array([np.asarray(g, dtype=float) for g in g_dirs])
    sv = np.linalg.svd(G, compute_uv=False)
    return float(sv[-1] / math.sqrt(len(g_dirs)))


def identify_payload(g_dirs: Sequence[np.ndarray],
                     wrenches: Sequence[np.ndarray],
                     gravity: float = GRAVITY) -> IdentificationResult:
    """Least-squares payload identification from N static raw readings.

    Parameters
    ----------
    g_dirs : sequence of (3,) arrays
        Unit gravity direction in the sensor frame at each pose.
    wrenches : sequence of (6,) arrays
        Time-averaged RAW sensor reading [F, M] at each pose (no tare
        subtracted — the bias is one of the identified unknowns).
    gravity : float
        Gravity magnitude (m/s²).

    Raises
    ------
    ValueError
        Fewer than 3 poses, mismatched lengths, or a rank-deficient pose
        set (gravity directions do not span enough of the sphere).
    """
    dirs = [np.asarray(g, dtype=float) for g in g_dirs]
    meas = [np.asarray(w, dtype=float) for w in wrenches]
    n = len(dirs)
    if n != len(meas):
        raise ValueError(
            f"g_dirs and wrenches length mismatch: {n} vs {len(meas)}"
        )
    if n < 3:
        raise ValueError(f"payload identification needs >= 3 poses, got {n}")
    for i, (g, w) in enumerate(zip(dirs, meas)):
        if g.shape != (3,):
            raise ValueError(f"g_dirs[{i}] must have shape (3,), got {g.shape}")
        if w.shape != (6,):
            raise ValueError(f"wrenches[{i}] must have shape (6,), got {w.shape}")

    A_f, A_m = _stacked_regressors(dirs, gravity)
    b_f = np.concatenate([w[:3] for w in meas])
    b_m = np.concatenate([w[3:] for w in meas])

    sol_f, _, rank_f, _ = np.linalg.lstsq(A_f, b_f, rcond=None)
    if rank_f < 4:
        raise ValueError(
            "force regressor is rank deficient — the gravity directions are "
            "too similar (need >= 2 distinct orientations)"
        )
    sol_m, _, rank_m, _ = np.linalg.lstsq(A_m, b_m, rcond=None)
    if rank_m < 6:
        raise ValueError(
            "torque regressor is rank deficient — the gravity directions are "
            "too similar (need >= 3 distinct orientations)"
        )

    mass = float(sol_f[0])
    bias_f = sol_f[1:4]
    c = sol_m[0:3]              # first moment m·r_com
    bias_m = sol_m[3:6]
    # Below MIN_MASS_FOR_COM the division amplifies noise into metres of COM
    # error; report a zero COM — torque compensation degenerates to bias-only.
    com = c / mass if mass > MIN_MASS_FOR_COM else np.zeros(3)

    params = PayloadParams(
        mass_kg=mass, com=com, bias_f=bias_f, bias_m=bias_m, gravity=gravity
    )

    # ── Residuals ────────────────────────────────────────────────────
    res_f = (A_f @ sol_f - b_f).reshape(n, 3)
    res_m = (A_m @ sol_m - b_m).reshape(n, 3)
    pose_res_f = np.linalg.norm(res_f, axis=1)
    pose_res_m = np.linalg.norm(res_m, axis=1)
    rms_f = float(np.sqrt(np.mean(res_f ** 2)))
    rms_m = float(np.sqrt(np.mean(res_m ** 2)))

    # Condition numbers on unit-normalised regressors so the numbers are
    # comparable across gravity conventions.
    cond_f, cond_m = pose_set_condition(dirs)

    return IdentificationResult(
        params=params,
        force_residual_rms=rms_f,
        torque_residual_rms=rms_m,
        cond_force=cond_f,
        cond_torque=cond_m,
        g_spread=g_dir_spread(dirs),
        pose_force_residuals=pose_res_f,
        pose_torque_residuals=pose_res_m,
    )


# ── Quality assessment ───────────────────────────────────────────────


def assess_identification(result: IdentificationResult) -> Tuple[str, List[str]]:
    """Grade an identification as PASS / WARN / REJECT with reasons.

    Returns
    -------
    (status, messages)
        status is "PASS", "WARN" or "REJECT"; messages explains every
        threshold that fired (empty for a clean PASS).
    """
    warns = []
    rejects = []

    m = result.params.mass_kg
    if m <= 0.0 or m > MASS_MAX:
        rejects.append(f"implausible mass {m:.3f} kg (expected 0 < m <= {MASS_MAX})")
    com_norm = float(np.linalg.norm(result.params.com))
    if com_norm > COM_NORM_MAX:
        rejects.append(f"implausible COM distance {com_norm:.3f} m (> {COM_NORM_MAX})")

    if result.cond_force > COND_FORCE_REJECT:
        rejects.append(f"force conditioning {result.cond_force:.1f} > {COND_FORCE_REJECT}")
    elif result.cond_force > COND_FORCE_WARN:
        warns.append(f"force conditioning {result.cond_force:.1f} > {COND_FORCE_WARN}")

    if result.cond_torque > COND_TORQUE_REJECT:
        rejects.append(f"torque conditioning {result.cond_torque:.1f} > {COND_TORQUE_REJECT}")
    elif result.cond_torque > COND_TORQUE_WARN:
        warns.append(f"torque conditioning {result.cond_torque:.1f} > {COND_TORQUE_WARN}")

    if result.force_residual_rms > FORCE_RMS_REJECT:
        rejects.append(f"force residual RMS {result.force_residual_rms:.3f} N > {FORCE_RMS_REJECT}")
    elif result.force_residual_rms > FORCE_RMS_WARN:
        warns.append(f"force residual RMS {result.force_residual_rms:.3f} N > {FORCE_RMS_WARN}")

    if result.torque_residual_rms > TORQUE_RMS_REJECT:
        rejects.append(f"torque residual RMS {result.torque_residual_rms:.3f} Nm > {TORQUE_RMS_REJECT}")
    elif result.torque_residual_rms > TORQUE_RMS_WARN:
        warns.append(f"torque residual RMS {result.torque_residual_rms:.3f} Nm > {TORQUE_RMS_WARN}")

    if result.g_spread < G_SPREAD_WARN:
        warns.append(f"gravity-direction spread {result.g_spread:.2f} < {G_SPREAD_WARN} "
                     "— orientations cover the sphere poorly")

    # Per-pose outliers (likely touched / not settled during capture)
    for label, pose_res, rms in (
        ("force", result.pose_force_residuals, result.force_residual_rms),
        ("torque", result.pose_torque_residuals, result.torque_residual_rms),
    ):
        if rms <= 0.0:
            continue
        for i, r in enumerate(pose_res):
            if r > POSE_OUTLIER_FACTOR * rms * math.sqrt(3.0):
                warns.append(f"pose {i + 1} {label} residual {r:.3f} is an outlier "
                             "— was the tool touched during capture?")

    if rejects:
        return "REJECT", rejects + warns
    if warns:
        return "WARN", warns
    return "PASS", warns


# ── Config bridging ──────────────────────────────────────────────────


def params_from_profile(profile, gravity: float = GRAVITY) -> PayloadParams:
    """Build PayloadParams from a config PayloadProfile (bias-free).

    The runtime compensator uses the delta-gravity formulation, so the
    session bias never enters — bias fields are zeroed.
    """
    return PayloadParams(
        mass_kg=float(profile.mass_kg),
        com=np.array([profile.com_x, profile.com_y, profile.com_z], dtype=float),
        bias_f=np.zeros(3),
        bias_m=np.zeros(3),
        gravity=float(gravity),
    )


def profile_to_dict(result: IdentificationResult, identified_on: str) -> dict:
    """Serialise an identification into the config's payload-profile keys."""
    p = result.params
    return {
        "mass_kg": round(p.mass_kg, 4),
        "com_x": round(float(p.com[0]), 5),
        "com_y": round(float(p.com[1]), 5),
        "com_z": round(float(p.com[2]), 5),
        "identified_on": identified_on,
        "residual_force_rms": round(result.force_residual_rms, 4),
        "residual_torque_rms": round(result.torque_residual_rms, 4),
    }


# ── Self-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.RandomState(42)

    truth = PayloadParams(
        mass_kg=2.4,
        com=np.array([0.003, -0.012, 0.065]),
        bias_f=np.array([1.2, -0.4, 3.1]),
        bias_m=np.array([0.05, -0.02, 0.01]),
        gravity=GRAVITY,
    )

    # Eight well-spread gravity directions (unit vectors)
    dirs = [
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0),
        np.array([-1.0, 1.0, -1.0]) / math.sqrt(3.0),
    ]

    # Simulated capture: 2000 samples per pose, gaussian noise + 0.1 quantisation
    wrenches = []
    for g in dirs:
        w = predicted_wrench(truth, g)
        samples = w[None, :] + rng.normal(0.0, 0.05, size=(2000, 6))
        samples = np.round(samples / 0.1) * 0.1
        wrenches.append(samples.mean(axis=0))

    res = identify_payload(dirs, wrenches)
    status, messages = assess_identification(res)

    p = res.params
    print(f"mass     : {p.mass_kg:8.4f} kg   (truth {truth.mass_kg})")
    print(f"com      : [{p.com[0]:+.4f}, {p.com[1]:+.4f}, {p.com[2]:+.4f}] m "
          f"(truth [{truth.com[0]:+.4f}, {truth.com[1]:+.4f}, {truth.com[2]:+.4f}])")
    print(f"bias F   : [{p.bias_f[0]:+.3f}, {p.bias_f[1]:+.3f}, {p.bias_f[2]:+.3f}] N")
    print(f"bias M   : [{p.bias_m[0]:+.3f}, {p.bias_m[1]:+.3f}, {p.bias_m[2]:+.3f}] Nm")
    print(f"residuals: {res.force_residual_rms:.4f} N / {res.torque_residual_rms:.4f} Nm")
    print(f"cond     : force {res.cond_force:.1f}, torque {res.cond_torque:.1f}, "
          f"spread {res.g_spread:.3f}")
    print(f"status   : {status}")
    for msg in messages:
        print(f"  - {msg}")

    assert abs(p.mass_kg - truth.mass_kg) < 0.01, "mass recovery failed"
    assert np.linalg.norm(p.com - truth.com) < 0.002, "COM recovery failed"
    assert np.linalg.norm(p.bias_f - truth.bias_f) < 0.05, "force bias recovery failed"
    assert np.linalg.norm(p.bias_m - truth.bias_m) < 0.02, "torque bias recovery failed"
    assert status == "PASS", f"expected PASS, got {status}: {messages}"
    print("\nself-test OK")
