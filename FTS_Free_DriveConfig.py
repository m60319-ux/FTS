"""Configuration schema for FTS_Free_Drive, loaded from a JSON file.

The nesting mirrors the sections the values used to arrive in, so a config file
reads section-by-section:

    {
      "param":             {"serial_port": "0", "auto_zero_seconds": 1.0, "io_num": 0},
      "control":           {"control_space": false, "orientation_control": false},
      "sensor_offset":     {"sensor_x": 0.0, ...},
      "force_to_velocity": {"max_linear_vel": 0.05, ...},

      // Optional — flange-mounted sensor mode (see FTS_Free_Drive_config_flange_example.json)
      "payload":           {"enabled": true, "profile_tool": {...}, ...},
      "wrench_filter":     {"filter_type": "butter2", "cutoff_hz": 10.0}
    }

Quoted numbers ("0.5") are accepted as well as bare ones — the app casts every
numeric field on read.

The `payload` and `wrench_filter` sections are OPTIONAL: a legacy config
without them loads exactly as before (both come back as None → handle mode).
When a section IS present it is parsed strictly like every other section —
unknown or missing keys raise, naming the section and key.

Flange mode (payload.enabled == true) changes the sensor mounting assumption:
the sensor is bolted to the flange with the tool on its measurement side, so
the app subtracts the orientation-dependent gravity wrench of the tool.
Payload profiles are written by payload_calibration.py — do not hand-edit
mass/COM values.

Dead-zone sizing for flange mode (see FTS_Free_Drive.py for the derivation):
with orientation control ON the gravity-model error re-enters as the wrist
rotates — budget  dz_F ≥ 1.5·(2g·δm + m·g·δθ + m·a_max + 0.2)  which for a
2.4 kg tool lands near 3 N (the legacy 0.5 N default is far too small).
With orientation OFF the tare absorbs the model error and
dz_F ≥ 1.5·(m·a_max + 0.2) suffices.

Target runtime is Python 3.6, so the sections are NamedTuples rather than
dataclasses (stdlib dataclasses arrived in 3.7).
"""

import json
import typing
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Type, TypeVar, get_type_hints


class Param(NamedTuple):
    serial_port: str          # serial index ("0") or device path ("/dev/ttyUSB0")
    auto_zero_seconds: float
    io_num: int               # digital input polled as the stop signal


class Control(NamedTuple):
    control_space: bool       # False = Cartesian, True = joint space
    orientation_control: bool


class SensorOffset(NamedTuple):
    sensor_x: float           # sensor origin → TCP origin (metres),
    sensor_y: float           # measured in the SENSOR frame (along the
    sensor_z: float           # sensor's own axes)
    sensor_a: float           # rotation about X, Y, Z (degrees), sensor→TCP
    sensor_b: float
    sensor_c: float


class ForceToVelocity(NamedTuple):
    max_linear_vel: float
    max_angular_vel: float
    dead_zone_force: float
    dead_zone_torque: float
    gain_force: float
    gain_torque: float


class PayloadProfile(NamedTuple):
    mass_kg: float            # tool(+load) mass (kg)
    com_x: float              # centre of mass in the SENSOR frame (metres)
    com_y: float
    com_z: float
    identified_on: str        # ISO timestamp written by payload_calibration.py ("" = never)
    residual_force_rms: float   # quality record from the identification (N)
    residual_torque_rms: float  # (Nm)


class Payload(NamedTuple):
    enabled: bool             # master switch — true selects flange mode
    profile_tool: PayloadProfile        # bare tool on the sensor
    profile_tool_load: PayloadProfile   # tool + attached load (all-zero if unused)
    active_profile: str       # "tool" | "tool_load" — used when switch_io < 0
    switch_io: int            # DI pin: LOW = tool, HIGH = tool_load; -1 disables
    gravity_x: float          # gravity vector in the BASE frame (m/s²);
    gravity_y: float          # (0, 0, -9.81) for a floor mount — set for
    gravity_z: float          # wall/ceiling mounted robots
    startup_max_residual_force: float   # N  — post-tare verify threshold
    startup_max_residual_torque: float  # Nm
    max_operator_force: float           # N  — plausibility stop (collision guard)
    max_operator_torque: float          # Nm
    max_accel_linear: float             # m/s²  — command slew limit
    max_accel_angular: float            # rad/s²
    decel_multiplier: float             # decel rate = accel × this (≥ 1)
    inertial_feedforward: bool          # experimental m·a add-back; default false


class WrenchFilter(NamedTuple):
    filter_type: str          # "butter2" | "ema" | "none"
    cutoff_hz: float          # -3 dB cutoff (Hz)


class FTS_Free_DriveConfig(NamedTuple):
    param: Param
    control: Control
    sensor_offset: SensorOffset
    force_to_velocity: ForceToVelocity
    payload: Optional[Payload] = None
    wrench_filter: Optional[WrenchFilter] = None


T = TypeVar("T")


def load_config(path) -> FTS_Free_DriveConfig:
    """Read *path* and build the config tree.

    Raises FileNotFoundError / ValueError naming the file — and, for a missing
    or surplus entry, the section and key — rather than surfacing a bare
    KeyError from somewhere deep in init().
    """
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(f"FTS_Free_Drive config not found: {path}")

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as ex:
        raise ValueError(f"{path} is not valid JSON: {ex}") from ex

    return _build(FTS_Free_DriveConfig, data, path, section=None)


def _is_section(field_type) -> bool:
    """True for the NamedTuple classes above, i.e. a field that nests a section."""
    return isinstance(field_type, type) and issubclass(field_type, tuple) and hasattr(field_type, "_fields")


def _unwrap_optional(field_type):
    """Return T for Optional[T] (i.e. Union[T, None]); other types unchanged.

    The optional sections are annotated Optional[Payload] etc., which
    get_type_hints reports as a Union — unwrap it so _is_section() can
    recognise the nested NamedTuple.
    """
    if getattr(field_type, "__origin__", None) is typing.Union:
        args = [a for a in field_type.__args__ if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return field_type


def _build(cls: Type[T], data: Any, path: Path, section: str) -> T:
    """Instantiate NamedTuple *cls* from *data*, recursing into nested sections."""
    where = f"section '{section}'" if section else "top level"

    if not isinstance(data, dict):
        raise ValueError(f"{path}: {where} must be a JSON object, got {type(data).__name__}")

    # get_type_hints, not __annotations__: the latter is the annotation as
    # written, which is a string rather than the class on newer Pythons.
    hints = get_type_hints(cls)

    expected = set(cls._fields)
    surplus = set(data) - expected
    if surplus:
        raise ValueError(f"{path}: unknown key(s) in {where}: {', '.join(sorted(surplus))}")

    defaults = getattr(cls, "_field_defaults", {})

    kwargs: Dict[str, Any] = {}
    for name in cls._fields:
        if name not in data:
            # Fields with a NamedTuple default (the optional sections) fall
            # back to it; everything else stays strictly required.
            if name in defaults:
                kwargs[name] = defaults[name]
                continue
            raise ValueError(f"{path}: missing key '{name}' in {where}")

        value = data[name]
        field_type = _unwrap_optional(hints[name])
        if _is_section(field_type):
            kwargs[name] = _build(field_type, value, path, section=name)
        else:
            kwargs[name] = value

    return cls(**kwargs)
