import sys
import threading
import time
from typing import Callable, Optional, Tuple

import serial


def resolve_serial_port(port_number: str) -> str:
    """Convert a port number to a full serial port path based on OS.

    Parameters
    ----------
    port_number : str
        The numeric portion of the serial port (e.g. "9" for COM10 or "0" for /dev/ttyUSB0).

    Returns
    -------
    str
        Full serial port path: ``/dev/ttyUSB<n>`` on Linux, ``COM<n>`` on Windows.
    """
    num = str(port_number).strip()
    if sys.platform.startswith("linux"):
        return f"/dev/ttyACM{num}"
    return f"COM{num}"


FTCallback = Callable[[float, float, float, float, float, float], None]


class FTSSensorReader:
    FRAME_HEADER = b"\xAA\x55"
    FRAME_LEN = 14
    _CRC8_POLY = 0x8C  # CRC-8/MAXIM reflected polynomial

    def __init__(
        self,
        port: str,
        callback: FTCallback,
        baudrate: int = 460800,
        parity=serial.PARITY_EVEN,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout: float = 0.01,
        auto_zero_seconds: float = 1.0,
        auto_zero: bool = True,
        read_size: int = 65536,
        max_buffer_size: int = 262144,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.parity = parity
        self.stopbits = stopbits
        self.bytesize = bytesize
        self.timeout = timeout

        self.callback = callback
        self.auto_zero_seconds = auto_zero_seconds
        # auto_zero=False skips the startup tare entirely: zero_done is set
        # immediately, offsets stay zero, so get_latest*() and the callback
        # deliver RAW values.  Used by payload_calibration.py, which needs
        # un-tared readings.
        self.auto_zero = auto_zero
        self.read_size = read_size
        self.max_buffer_size = max_buffer_size

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running_lock = threading.Lock()

        self._ser: Optional[serial.Serial] = None

        # latest zeroed values (equal to raw when auto_zero is False)
        self._latest_lock = threading.Lock()
        self._latest = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._latest_t: float = 0.0

        # latest raw values — updated for every decoded frame, tare or not
        self._latest_raw = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._latest_raw_t: float = 0.0

        # Lock protecting stats and zero state read from the main thread
        self._stats_lock = threading.Lock()
        self._frames_parsed = 0
        self._decode_errors = 0
        self._fps = 0.0
        self._offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._zero_done = False

    @property
    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    @property
    def zero_done(self) -> bool:
        with self._stats_lock:
            return self._zero_done

    @property
    def offsets(self) -> Tuple[float, float, float, float, float, float]:
        with self._stats_lock:
            return self._offsets

    @property
    def fps(self) -> float:
        with self._stats_lock:
            return self._fps

    @property
    def frames_parsed(self) -> int:
        with self._stats_lock:
            return self._frames_parsed

    @property
    def decode_errors(self) -> int:
        with self._stats_lock:
            return self._decode_errors

    def get_latest(self) -> Tuple[float, float, float, float, float, float]:
        with self._latest_lock:
            return self._latest

    def get_latest_with_age(
        self,
    ) -> Tuple[Tuple[float, float, float, float, float, float], float]:
        """Return (latest_values, age_seconds) atomically.

        age_seconds = time.perf_counter() - timestamp of the last decoded frame.
        If no frame has arrived yet, age is very large — callers should gate on
        zero_done before using the values.
        """
        with self._latest_lock:
            values = self._latest
            t = self._latest_t
        return values, time.perf_counter() - t

    def get_latest_raw_with_age(
        self,
    ) -> Tuple[Tuple[float, float, float, float, float, float], float]:
        """Return (latest_RAW_values, age_seconds) atomically.

        Raw values are pre-tare and are updated for every decoded frame,
        including during the auto-zero window.  Used by the payload
        calibration, which must see the un-tared gravity wrench.
        """
        with self._latest_lock:
            values = self._latest_raw
            t = self._latest_raw_t
        return values, time.perf_counter() - t

    def start(self) -> None:
        with self._running_lock:
            if self.is_running:
                return

            self._stop_event.clear()
            with self._stats_lock:
                self._frames_parsed = 0
                self._decode_errors = 0
                self._fps = 0.0
                self._offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                # With auto_zero disabled the tare branch in _run() is never
                # entered: zero_done is already True and offsets stay zero,
                # so downstream consumers receive raw values immediately.
                self._zero_done = not self.auto_zero

            self._thread = threading.Thread(
                target=self._run,
                name="FTSSensorReaderThread",
                daemon=True,
            )
            self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        with self._running_lock:
            self._stop_event.set()

            # Do NOT close _ser here — _run()'s finally block handles it.
            # The reader thread will notice _stop_event within one serial
            # timeout period (default 10 ms) and exit cleanly.

            t = self._thread
            if t is not None:
                t.join(timeout=join_timeout)

            self._thread = None
            self._ser = None

    def _open_serial(self) -> serial.Serial:
        return serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=self.bytesize,
            parity=self.parity,
            stopbits=self.stopbits,
            timeout=0,  # non-blocking; reads are gated by in_waiting
        )

    @staticmethod
    def _crc8(buf: bytearray, start: int, length: int) -> int:
        """CRC-8/MAXIM (reflected poly 0x8C, init 0x00) over buf[start:start+length]."""
        crc = 0x00
        for i in range(start, start + length):
            crc ^= buf[i]
            for _ in range(8):
                if crc & 0x01:
                    crc = ((crc >> 1) ^ 0x8C) & 0xFF
                else:
                    crc = (crc >> 1) & 0xFF
        return crc

    @staticmethod
    def _decode_frame_from_buffer(buf: bytearray, start: int) -> Tuple[float, float, float, float, float, float]:
        b2 = buf[start + 2]
        b3 = buf[start + 3]
        b4 = buf[start + 4]
        b5 = buf[start + 5]
        b6 = buf[start + 6]
        b7 = buf[start + 7]
        b8 = buf[start + 8]
        b9 = buf[start + 9]
        b10 = buf[start + 10]
        b11 = buf[start + 11]
        b12 = buf[start + 12]

        fx0 = (b2 << 9) | (b3 << 1) | ((b4 & 0x80) >> 7)
        fy0 = ((b4 & 0x7F) << 10) | (b5 << 2) | ((b6 & 0xC0) >> 6)
        fz0 = ((b6 & 0x3F) << 11) | (b7 << 3) | ((b8 & 0xE0) >> 5)
        mx0 = ((b8 & 0x1F) << 7) | ((b9 & 0xFE) >> 1)
        my0 = ((b9 & 0x01) << 11) | (b10 << 3) | ((b11 & 0xE0) >> 5)
        mz0 = ((b11 & 0x1F) << 7) | ((b12 & 0xFE) >> 1)

        # C# sign-magnitude behavior
        fx0 = -(fx0 & 0xFFFF) if (fx0 >> 16) else (fx0 & 0xFFFF)
        fy0 = -(fy0 & 0xFFFF) if (fy0 >> 16) else (fy0 & 0xFFFF)
        fz0 = -(fz0 & 0xFFFF) if (fz0 >> 16) else (fz0 & 0xFFFF)

        mx0 = -(mx0 & 0x7FF) if (mx0 >> 11) else (mx0 & 0x7FF)
        my0 = -(my0 & 0x7FF) if (my0 >> 11) else (my0 & 0x7FF)
        mz0 = -(mz0 & 0x7FF) if (mz0 >> 11) else (mz0 & 0x7FF)

        return (
            fx0 * 0.1,
            fy0 * 0.1,
            fz0 * 0.1,
            mx0 * 0.1,
            my0 * 0.1,
            mz0 * 0.1,
        )

    def _run(self) -> None:
        buffer = bytearray()

        zero_sum_fx = 0.0
        zero_sum_fy = 0.0
        zero_sum_fz = 0.0
        zero_sum_mx = 0.0
        zero_sum_my = 0.0
        zero_sum_mz = 0.0
        zero_count = 0

        start_t = time.perf_counter()
        zero_start_t = start_t
        rate_t = start_t
        frames_since_rate = 0

        try:
            self._ser = self._open_serial()
            self._ser.reset_input_buffer()

            while not self._stop_event.is_set():
                try:
                    avail = self._ser.in_waiting
                    if not avail:
                        # Yield GIL while waiting for the next frame (~0.5 ms
                        # per frame at 2 kHz). Without this sleep the reader
                        # thread busy-spins and still loses GIL races with the
                        # control-loop RPCs; with it, the OS schedules the
                        # reader at least every 0.5 ms.
                        time.sleep(0.0005)
                        continue
                    chunk = self._ser.read(avail)
                except Exception:
                    break

                if chunk:
                    buffer.extend(chunk)

                if len(buffer) > self.max_buffer_size:
                    del buffer[:-8192]

                search_pos = 0
                consume_upto = 0

                while not self._stop_event.is_set():
                    start = buffer.find(self.FRAME_HEADER, search_pos)
                    if start < 0:
                        # No header found — discard all scanned bytes,
                        # but keep a trailing 0xAA that could start a header.
                        if len(buffer) > 0 and buffer[-1] == 0xAA:
                            consume_upto = len(buffer) - 1
                        else:
                            consume_upto = len(buffer)
                        break

                    if start + self.FRAME_LEN > len(buffer):
                        # Incomplete frame — keep from 'start' onwards
                        consume_upto = start
                        break

                    # Validate CRC-8 checksum (byte 13) over payload bytes 2–12
                    expected_crc = self._crc8(buffer, start + 2, 11)
                    if buffer[start + 13] != expected_crc:
                        with self._stats_lock:
                            self._decode_errors += 1
                        search_pos = start + 1
                        continue

                    try:
                        fx, fy, fz, mx, my, mz = self._decode_frame_from_buffer(buffer, start)
                    except Exception:
                        with self._stats_lock:
                            self._decode_errors += 1
                        search_pos = start + 1
                        continue

                    with self._stats_lock:
                        self._frames_parsed += 1
                    frames_since_rate += 1
                    consume_upto = start + self.FRAME_LEN
                    search_pos = consume_upto

                    now = time.perf_counter()

                    if now - rate_t >= 0.25:
                        dt = now - rate_t
                        if dt > 0:
                            with self._stats_lock:
                                self._fps = frames_since_rate / dt
                        frames_since_rate = 0
                        rate_t = now

                    # Raw values are published for every frame — including
                    # during the tare window — for consumers that need the
                    # un-tared reading (payload calibration).
                    with self._latest_lock:
                        self._latest_raw = (fx, fy, fz, mx, my, mz)
                        self._latest_raw_t = now

                    if not self._zero_done:
                        zero_sum_fx += fx
                        zero_sum_fy += fy
                        zero_sum_fz += fz
                        zero_sum_mx += mx
                        zero_sum_my += my
                        zero_sum_mz += mz
                        zero_count += 1

                        if now - zero_start_t >= self.auto_zero_seconds and zero_count > 0:
                            inv = 1.0 / zero_count
                            with self._stats_lock:
                                self._offsets = (
                                    zero_sum_fx * inv,
                                    zero_sum_fy * inv,
                                    zero_sum_fz * inv,
                                    zero_sum_mx * inv,
                                    zero_sum_my * inv,
                                    zero_sum_mz * inv,
                                )
                                self._zero_done = True
                        continue

                    off_fx, off_fy, off_fz, off_mx, off_my, off_mz = self._offsets

                    zfx = fx - off_fx
                    zfy = fy - off_fy
                    zfz = fz - off_fz
                    zmx = mx - off_mx
                    zmy = my - off_my
                    zmz = mz - off_mz

                    with self._latest_lock:
                        self._latest = (zfx, zfy, zfz, zmx, zmy, zmz)
                        self._latest_t = now

                    try:
                        self.callback(zfx, zfy, zfz, zmx, zmy, zmz)
                    except Exception:
                        # Keep reader alive even if user callback fails
                        pass

                if consume_upto > 0:
                    del buffer[:consume_upto]

        finally:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
            self._ser = None


# ---------------- Example usage ----------------

if __name__ == "__main__":
    def on_ft_data(fx: float, fy: float, fz: float, mx: float, my: float, mz: float) -> None:
        # Replace this with your robot jog logic
        print(f"Fx={fx:7.3f}, Fy={fy:7.3f}, Fz={fz:7.3f}, Mx={mx:7.3f}, My={my:7.3f}, Mz={mz:7.3f}")

    reader = FTSSensorReader(
        port=resolve_serial_port("9"),
        callback=on_ft_data,
        baudrate=460800,
        auto_zero_seconds=2.0,
    )

    try:
        reader.start()

        while True:
            time.sleep(1.0)
            print(
                f"running={reader.is_running}, "
                f"zero_done={reader.zero_done}, "
                f"fps={reader.fps:.1f}, "
                f"frames={reader.frames_parsed}"
            )
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
