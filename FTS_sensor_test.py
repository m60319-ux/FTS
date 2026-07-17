"""Standalone diagnostic utility for the M4313SFA9A F/T sensor.

This program tests only the RS-485 sensor path. It does not connect to a robot,
read robot DI, or activate a servo interface.

Examples
--------
    python FTS_sensor_test.py --port 9
    python FTS_sensor_test.py --port COM9 --duration 30 --csv fts_log.csv

Expected default serial format: 460800 / 8E1.
"""

from __future__ import annotations

import argparse
import csv
import math
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, TextIO

try:
    import serial
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(
        "pyserial is not installed. Run: py -m pip install pyserial"
    ) from exc

try:
    from M4313SFA9A_helper import FTSSensorReader, resolve_serial_port
except ImportError as exc:
    raise SystemExit(
        "Cannot import M4313SFA9A_helper.py. Put it in the same folder as "
        "FTS_sensor_test.py."
    ) from exc


PARITY_MAP: Dict[str, str] = {
    "N": serial.PARITY_NONE,
    "E": serial.PARITY_EVEN,
    "O": serial.PARITY_ODD,
}

BYTE_SIZE_MAP: Dict[int, int] = {
    7: serial.SEVENBITS,
    8: serial.EIGHTBITS,
}

STOP_BITS_MAP: Dict[float, float] = {
    1.0: serial.STOPBITS_ONE,
    1.5: serial.STOPBITS_ONE_POINT_FIVE,
    2.0: serial.STOPBITS_TWO,
}


def normalize_port(value: str) -> str:
    """Accept 9, COM9, or a full /dev path."""
    port = str(value).strip()
    if not port:
        raise ValueError("Serial port cannot be empty")

    if port.upper().startswith("COM"):
        return port.upper()
    if port.startswith("/dev/"):
        return port
    return resolve_serial_port(port)


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test M4313SFA9A RS-485 communication, Auto Zero, frame rate, "
            "CRC decoding, and zeroed six-axis output without connecting a robot."
        )
    )
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port number or name, for example 9, COM9, or /dev/ttyACM0",
    )
    parser.add_argument("--baudrate", type=int, default=460800)
    parser.add_argument(
        "--parity",
        choices=tuple(PARITY_MAP),
        default="E",
        help="N, E, or O; default E",
    )
    parser.add_argument(
        "--bytesize",
        type=int,
        choices=tuple(BYTE_SIZE_MAP),
        default=8,
    )
    parser.add_argument(
        "--stopbits",
        type=float,
        choices=tuple(STOP_BITS_MAP),
        default=1.0,
    )
    parser.add_argument(
        "--auto-zero",
        type=positive_float,
        default=2.0,
        help="Auto Zero averaging time in seconds; keep the tool still and untouched",
    )
    parser.add_argument(
        "--duration",
        type=non_negative_float,
        default=0.0,
        help="Test duration after Auto Zero; 0 means run until Ctrl+C",
    )
    parser.add_argument(
        "--print-hz",
        type=positive_float,
        default=10.0,
        help="Console and CSV sample rate; default 10 Hz",
    )
    parser.add_argument(
        "--min-fps",
        type=non_negative_float,
        default=500.0,
        help="Warn when decoded sensor frame rate is below this value; 0 disables",
    )
    parser.add_argument(
        "--stale-ms",
        type=positive_float,
        default=100.0,
        help="Fail when no decoded frame arrives within this interval",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    return parser


def open_csv(path: Optional[Path]) -> tuple[Optional[TextIO], Optional[csv.writer]]:
    if path is None:
        return None, None

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    file_obj = path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(file_obj)
    writer.writerow(
        [
            "elapsed_s",
            "fx_N",
            "fy_N",
            "fz_N",
            "mx_Nm",
            "my_Nm",
            "mz_Nm",
            "fps",
            "age_ms",
            "frames",
            "decode_errors",
        ]
    )
    file_obj.flush()
    return file_obj, writer


def print_configuration(args: argparse.Namespace, port: str) -> None:
    print("=" * 72)
    print("M4313SFA9A standalone sensor test")
    print(f"Port          : {port}")
    print(
        "Serial format : "
        f"{args.baudrate} / {args.bytesize}{args.parity}{args.stopbits:g}"
    )
    print(f"Auto Zero     : {args.auto_zero:.2f} s")
    print(
        "Duration      : "
        + ("until Ctrl+C" if args.duration == 0 else f"{args.duration:.1f} s")
    )
    print("Keep the sensor/tool still and untouched during Auto Zero.")
    print("=" * 72)


def main() -> int:
    args = build_parser().parse_args()

    try:
        port = normalize_port(args.port)
    except ValueError as exc:
        print(f"Port error: {exc}", file=sys.stderr)
        return 2

    print_configuration(args, port)

    stop_requested = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    reader = FTSSensorReader(
        port=port,
        callback=lambda *_values: None,
        baudrate=args.baudrate,
        parity=PARITY_MAP[args.parity],
        stopbits=STOP_BITS_MAP[args.stopbits],
        bytesize=BYTE_SIZE_MAP[args.bytesize],
        auto_zero_seconds=args.auto_zero,
    )

    csv_file: Optional[TextIO] = None
    csv_writer: Optional[csv.writer] = None

    try:
        csv_file, csv_writer = open_csv(args.csv)

        reader.start()
        startup_deadline = time.perf_counter() + args.auto_zero + 5.0
        last_status = 0.0

        while not reader.zero_done and not stop_requested:
            now = time.perf_counter()
            if not reader.is_running:
                print(
                    "ERROR: Sensor reader stopped. Check the COM port, cable, "
                    "RS-485 polarity, and 460800/8E1 format.",
                    file=sys.stderr,
                )
                return 3
            if now >= startup_deadline:
                print(
                    "ERROR: Auto Zero timeout. Valid sensor frames were not received.",
                    file=sys.stderr,
                )
                return 4
            if now - last_status >= 0.5:
                print(
                    "Waiting for Auto Zero... "
                    f"frames={reader.frames_parsed}, "
                    f"fps={reader.fps:.1f}, "
                    f"crc/decode_errors={reader.decode_errors}",
                    end="\r",
                    flush=True,
                )
                last_status = now
            time.sleep(0.02)

        if stop_requested:
            print("\nStopped before Auto Zero completed.")
            return 0

        print("\nAuto Zero completed.")
        print(
            "Offsets        : "
            + ", ".join(f"{value:+.3f}" for value in reader.offsets)
            + "  [Fx,Fy,Fz N; Mx,My,Mz Nm]"
        )
        print("Live values:")

        start_t = time.perf_counter()
        next_sample = start_t
        sample_period = 1.0 / args.print_hz
        low_fps_warned = False

        while not stop_requested:
            now = time.perf_counter()
            elapsed = now - start_t
            if args.duration > 0 and elapsed >= args.duration:
                break

            if not reader.is_running:
                print("\nERROR: Sensor reader thread stopped.", file=sys.stderr)
                return 5

            if now < next_sample:
                time.sleep(min(next_sample - now, 0.01))
                continue

            values, age = reader.get_latest_with_age()
            age_ms = age * 1000.0
            if age_ms > args.stale_ms:
                print(
                    f"\nERROR: Sensor data is stale ({age_ms:.1f} ms).",
                    file=sys.stderr,
                )
                return 6

            fx, fy, fz, mx, my, mz = values
            print(
                f"t={elapsed:8.3f}s  "
                f"F=[{fx:+8.3f}, {fy:+8.3f}, {fz:+8.3f}] N  "
                f"M=[{mx:+8.3f}, {my:+8.3f}, {mz:+8.3f}] Nm  "
                f"fps={reader.fps:7.1f}  age={age_ms:6.2f}ms  "
                f"err={reader.decode_errors}"
            )

            if (
                args.min_fps > 0
                and elapsed > 2.0
                and reader.fps > 0
                and reader.fps < args.min_fps
                and not low_fps_warned
            ):
                print(
                    f"WARNING: Decoded frame rate {reader.fps:.1f} is below "
                    f"the configured minimum {args.min_fps:.1f} fps.",
                    file=sys.stderr,
                )
                low_fps_warned = True

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        f"{elapsed:.6f}",
                        f"{fx:.6f}",
                        f"{fy:.6f}",
                        f"{fz:.6f}",
                        f"{mx:.6f}",
                        f"{my:.6f}",
                        f"{mz:.6f}",
                        f"{reader.fps:.3f}",
                        f"{age_ms:.3f}",
                        reader.frames_parsed,
                        reader.decode_errors,
                    ]
                )
                csv_file.flush()

            next_sample += sample_period
            if next_sample < now - sample_period:
                next_sample = now + sample_period

        print("Test completed normally.")
        print(
            f"Final statistics: frames={reader.frames_parsed}, "
            f"fps={reader.fps:.1f}, decode_errors={reader.decode_errors}"
        )
        return 0

    except serial.SerialException as exc:
        print(f"Serial port error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        reader.stop()
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    raise SystemExit(main())
