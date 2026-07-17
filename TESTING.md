# M4313SFA9A standalone test

`FTS_sensor_test.py` tests the sensor without connecting to the robot control box.
It verifies:

- RS-485 port access
- default serial format `460800 / 8E1`
- `AA 55` frame parsing and CRC-8 validation
- Auto Zero completion and calculated offsets
- live `Fx/Fy/Fz/Mx/My/Mz` values
- frame rate, stale-data age, and decode-error count
- optional CSV recording

It does **not** need Robot API communication, robot DI, power-on, collision-detection changes, or servo activation.

## 1. Install Python dependencies

On Windows, use the Python launcher:

```powershell
py -m pip install pyserial
```

If `py` is unavailable, install Python and enable **Add Python to PATH**.

## 2. Find the COM port

Open Windows Device Manager and check **Ports (COM & LPT)**.
For example, if the USB-RS485 adapter is `COM9`, either form is accepted:

```powershell
py FTS_sensor_test.py --port 9
```

```powershell
py FTS_sensor_test.py --port COM9
```

The default communication format is:

```text
460800 baud, 8 data bits, even parity, 1 stop bit
```

## 3. Auto Zero test

Keep the sensor, tool, and load completely still and untouched during the initial averaging window:

```powershell
py FTS_sensor_test.py --port COM9 --auto-zero 2
```

After Auto Zero, the six displayed values should remain near zero while untouched.
Apply force in one direction at a time and confirm the expected axis changes.
Press `Ctrl+C` to stop.

## 4. Fixed-duration test and CSV logging

Run for 30 seconds after Auto Zero and save values at 20 Hz:

```powershell
py FTS_sensor_test.py --port COM9 --auto-zero 2 --duration 30 --print-hz 20 --csv fts_log.csv
```

CSV columns include the six-axis values, frame rate, frame age, total frame count, and decode errors.

## 5. Change the serial format only when the sensor specification requires it

Example `115200 / 8N1`:

```powershell
py FTS_sensor_test.py --port COM9 --baudrate 115200 --bytesize 8 --parity N --stopbits 1
```

For the current M4313SFA9A helper, the expected default is `460800 / 8E1`.

## 6. Result interpretation

| Result | Meaning / action |
|---|---|
| Auto Zero completes and values change with force | Sensor communication and decoding are working |
| Reader stopped before Auto Zero | Check COM port, cable, RS-485 A/B polarity, power, and serial format |
| Auto Zero timeout with zero frames | No valid `AA 55` frames were received |
| Decode errors rise continuously | Wrong format, noisy wiring, reversed/poor RS-485 connection, or incompatible frame definition |
| Sensor data stale | Data stream stopped after communication had begun |
| Values drift while untouched | Repeat Auto Zero with no contact; inspect cable force and temperature drift |
| Large static offset before zero | Normal when a tool/load is mounted; Auto Zero removes only the current-pose offset |

## 7. Relationship to force-guiding operation

Passing this test confirms the sensor side only. Then test in this order:

1. sensor standalone test
2. robot network connection with `Robot()`
3. robot DI stop input
4. payload profile / gravity compensation when flange-mounted
5. low-speed force-guiding operation

Do not use Auto Zero as a replacement for payload calibration when the sensor is mounted on the robot flange and the wrist orientation changes.
