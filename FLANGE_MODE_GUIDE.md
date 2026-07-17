# FTS Free Drive — Flange-Mount Calibration & Operation Guide

This guide covers the **flange mounting**: the force-torque sensor (M4313SFA9A)
is bolted directly to the robot flange, the tool is mounted on the sensor, and
the operator guides the robot by gripping the tool. The tool may carry a load
that is picked up / released during the application.

> The legacy **handle mounting** (sensor on a handle, no mass on its
> measurement side) still works unchanged: any config *without* a `payload`
> section — or with `"enabled": false` — runs exactly as before. Nothing in
> this guide is required for handle mode.

---

## 1. Why calibration is needed

With the tool hanging on the sensor, every reading contains the tool's
**weight**, and that weight vector rotates in the sensor frame whenever the
wrist rotates. To isolate the operator's push, the app needs to know the
payload's **mass** and **centre of mass** — that is what the calibration
identifies. You never type these numbers yourself: the calibration script
measures them and writes them into the config.

Because the mass changes when the load is attached, there are **two
profiles**, each calibrated separately:

| Profile | Physical state on the sensor | Config section it fills |
|---|---|---|
| `tool` | bare tool | `payload.profile_tool` |
| `tool_load` | tool + load attached | `payload.profile_tool_load` |

What the calibration writes automatically: `mass_kg`, `com_x/y/z`,
`identified_on`, `residual_force_rms`, `residual_torque_rms`.

What **you** set manually (decisions, not measurements): `sensor_offset`,
the dead-zones, `switch_io`, the safety thresholds, and finally
`payload.enabled`.

---

## 2. Prepare

### 2.1 Hardware
- Bolt the sensor to the flange, the tool to the sensor. Everything that
  hangs on the sensor's measurement side (tool, cables, covers) is part of
  the payload — route cables so they do not tug on the tool.
- Connect the sensor's serial line and note the port
  (`param.serial_port`: `"10"` → `COM10` on Windows, `/dev/ttyACM10` on Linux).
- Wire the **stop input** (`param.io_num`) — the app will not start without it.
- Optionally wire the **load-state input** (`payload.switch_io`):
  LOW = bare tool, HIGH = tool + load. Use `-1` if not wired.

### 2.2 Config
Start from `FTS_Free_Drive_config_flange_example.json`. Fill in:

- **`sensor_offset`** — the sensor→TCP transform:
  - `sensor_x/y/z`: offset **from the sensor origin to the TCP origin,
    measured along the sensor's own axes** (metres). Example: TCP 15 cm
    beyond the sensor along the sensor's +z → `sensor_z: 0.15`.
  - `sensor_a/b/c`: rotation about X, Y, Z (degrees) mapping sensor-frame
    vectors into the TCP frame.
  - If the TCP is re-taught on the pendant, update this section in lockstep.
- **`param.serial_port`**, **`param.io_num`**.
- Leave `payload.enabled: false` for now, and leave the profile values as
  zeros — the calibration fills them.

### 2.3 Teach calibration poses (recommended)
Teach **at least 4 poses** (e.g. `P1…P5`) on the pendant. Rules of thumb:

- Only **orientation** matters — position can be anywhere collision-free.
- The tool axis should point in clearly **different directions** across the
  poses: up, down, two sideways directions, one diagonal.
- Avoid a set where all poses only differ by a roll around a vertical tool
  axis — rolling about gravity does not change the measured weight direction,
  and the script's pre-flight check will reject such a set.
- The **last pose in the list** is used as the *holdout*: it is not part of
  the fit and only verifies that the result generalises. Teach one extra
  "odd" orientation for it.

If you do not want to teach poses, the script can build its own targets by
rotating wrist joints J4–J6 from the current pose (start with the tool axis
roughly horizontal), or you can jog manually between captures.

---

## 3. Run the calibration

One run per physical state. **With the bare tool mounted:**

```bash
python payload_calibration.py FTS_Free_Drive_config.json --profile tool --poses P1,P2,P3,P4,P5
```

**Then attach the load and run again:**

```bash
python payload_calibration.py FTS_Free_Drive_config.json --profile tool_load --poses P1,P2,P3,P4,P5
```

Useful variants:

| Command | What it does |
|---|---|
| `--dry-run` | Reads the poses back (`get_point`) and prints the predicted conditioning. **No motion, no write.** Do this first. |
| *(no `--poses`)* | Wrist-offset mode: 8 fit + 2 holdout poses built from the current pose by rotating J4–J6. |
| `--quick` | Wrist-offset mode with only 3 + 1 poses (time-constrained re-identification; wider uncertainty). |
| `--manual` | No script-commanded motion: you move the robot between captures (from the pendant); combinable with `--poses`. |
| `--no-write` | Identify and report only; config untouched. |
| `--speed 5` | Motion speed for `move_joint` (default 5). |

### What a run looks like
1. Script connects, reads the taught poses, prints the **pre-flight
   conditioning** — if the pose set is poorly conditioned it tells you
   before anything moves.
2. You confirm the motion (`y`), then each pose is visited with a per-pose
   Enter confirmation.
3. At each pose the robot must be still and **nobody may touch the tool** —
   the script averages ~2 s of readings and rejects noisy captures
   ("is someone touching the tool?").
4. After the last pose it solves the fit, checks it against the **holdout
   pose**, prints a quality report, and — on PASS — writes the profile into
   the config.

### Reading the report

```
mass          : 2.4013 kg
centre of mass: [+3.1, -11.8, +64.9] mm (sensor frame)
residual RMS  : 0.041 N / 0.0088 Nm
conditioning  : force 1.3, torque 1.5, spread 0.51
holdout 1     : 0.11 N / 0.021 Nm [ok]
status        : PASS
```

| Status | Meaning | Action |
|---|---|---|
| `PASS` | fit is clean, holdout verified | profile written — done |
| `WARN` | usable but degraded (conditioning, residuals, outlier pose) | script asks before writing; prefer fixing the cause and re-running |
| `REJECT` / holdout FAIL | implausible mass/COM, high residuals, or the model does not generalise | nothing written. Usual culprits: wrong `sensor_offset` rotation, cables tugging, someone touched the tool, TCP mismatch |

---

## 4. Enable flange mode

After **both** profiles are calibrated (or just `tool`, if you never carry a
load and keep `switch_io: -1`):

1. **Size the dead-zones.** The compensation is model-based, so its residual
   error plus the tool's inertial reaction must fit inside the dead-zone or
   the robot will drift/chatter. Rules of thumb (S = safety factor 1.5,
   m = heaviest profile mass):

   - orientation control **off** (translation-only guiding):
     `dead_zone_force ≥ 1.5 · (m · max_accel_linear + 0.2)` → ≈ **1.2 N** for 2.4 kg
   - orientation control **on** (wrist rotates away from the tare pose):
     ≈ **3.3 N / 0.8 Nm** for a 2.4 kg tool — the legacy 0.5 N / 0.1 Nm
     defaults are far too small here.
   - Also keep `max_accel_linear ≤ dead_zone_force / (3·m)`.

   The app checks this at startup and prints a WARNING if the configured
   dead-zones are below the computed floor.

2. Review the `payload` thresholds (defaults are sensible):
   `startup_max_residual_force/torque` (tare verification gate),
   `max_operator_force/torque` (implausible-wrench stop — also acts as a
   crude collision stop), `max_accel_linear/angular`, `decel_multiplier`.

3. Set `"enabled": true` in the `payload` section. (The calibration script
   deliberately never does this for you.)

4. Optionally keep the `wrench_filter` section (`butter2`, 10 Hz default).
   If you ever feel chatter with a heavy tool, lower `cutoff_hz` to 5 before
   touching any gains.

---

## 5. Run FTS_Free_Drive.py

```bash
python FTS_Free_Drive.py FTS_Free_Drive_config.json
```

### Startup sequence — what you will see
1. `FTS App initialized — … mounting: flange, profile: tool (m=2.40 kg, …)`
2. If `switch_io` is wired: `Profile from switch input DI 5: tool_load` —
   the pin decides the starting profile, so **the pin must match the real
   load state before you start**.
3. `Waiting for sensor zero calibration …` — the automatic tare.
   **Keep your hands off the tool and do not move the robot** for
   `auto_zero_seconds` (~1 s). The app aborts if the robot moved during
   the tare (`tare_motion`).
4. `Tare captured … verifying residual, keep hands off` — 0.5 s check that
   the compensated reading is near zero. If it is not (wrong profile on the
   pin, someone touching, dead sensor channel) → `tare_residual` stop and
   the robot never moves.
5. `Tare verified … hand guiding active` — **now you can guide the robot.**

### During operation
- Push the tool → the robot follows; release → it stops within ~100 ms.
- **Load release / pickup:** toggle the `switch_io` pin at (or right after)
  the physical event. The robot holds zero velocity, swaps the gravity
  model, and resumes once the reading settles — take your hands off for a
  moment so the verification gate can pass. If it cannot settle within 5 s
  (pin does not match the real load state) the app stops safely.
- The **stop input** (`param.io_num` HIGH) halts everything at any time.
- A rate-limited warning `compensated wrench is drifting toward the
  dead-zone` means the tare is degrading (temperature, bumped payload) —
  finish the task and restart the app to re-tare.

### Stop reasons (log line `FTS App finished (reason: …)`)

| Reason | Meaning / typical cause |
|---|---|
| `io_high`, `io_high_at_startup` | stop input asserted |
| `reader_thread_died`, `zero_calibration_timeout` | sensor unplugged / wrong serial port |
| `sensor_stale_…` | sensor frames stopped mid-run |
| `tare_motion` | robot moved during the tare window |
| `tare_residual_…` | reading not near zero after tare — wrong profile at startup, hands on tool, bad calibration |
| `pose_stale_…` | TCP pose updates stopped (gravity direction unknown) |
| `wrench_implausible_…` | compensated wrench above `max_operator_*` — collision or gross model error |
| `profile_switch_verify_failed` | switch pin does not match the physical load state |
| `profile_switch_during_startup` | load state changed while the tare was being taken |

---

## 6. When to re-calibrate

- The tool (or anything attached to it — grippers, cameras, cable dress) is
  changed, remounted, or modified → re-run **both** profiles.
- The sensor is re-mounted or `sensor_offset` changes → re-run both profiles.
- `tare_residual` stops or drift warnings become frequent → suspect the
  calibration; verify with `--no-write` and compare.
- The load itself changes (different part weight) → re-run `tool_load`.

A quick sanity check any time: run the calibration with `--no-write` and
confirm the reported mass/COM matches the stored profile within a few
percent, and the holdout passes.
