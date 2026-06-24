# -*- coding: utf-8 -*-
"""Read temperature from SAX0R4S RS485 infrared thermometer via Modbus-RTU."""

from __future__ import annotations

import argparse
import struct
import sys
import time
from typing import Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:
    raise SystemExit("pyserial is required: pip install pyserial") from exc


DEFAULT_PORT = "COM6"
DEFAULT_BAUDRATE = 9600
DEFAULT_SLAVE_ID = 0x01
TEMPERATURE_REGISTER = 0x0000
READ_TIMEOUT_S = 1.0
DEFAULT_RETRIES = 2
POST_OPEN_DELAY_S = 0.1


def modbus_char_time_s(baudrate: int) -> float:
    """Time for one 8N1 character on the wire."""
    return 11.0 / baudrate


def modbus_inter_byte_timeout_s(baudrate: int) -> float:
    """Modbus RTU inter-character timeout (~3.5 character times)."""
    return modbus_char_time_s(baudrate) * 3.5


def list_serial_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def format_serial_error(exc: serial.SerialException, port: str) -> str:
    """Turn common serial failures into actionable messages."""
    if isinstance(exc, PermissionError) or "PermissionError" in str(exc) or "拒绝访问" in str(exc):
        others = [p for p in list_serial_ports() if p != port]
        hint = "Close any serial monitor/debugger using this port, then retry."
        if others:
            return f"Port {port} is busy or access denied. {hint} Other ports: {', '.join(others)}"
        return f"Port {port} is busy or access denied. {hint}"

    if isinstance(exc, FileNotFoundError) or "FileNotFoundError" in str(exc) or "找不到" in str(exc):
        available = list_serial_ports()
        if available:
            return f"Port {port} not found. Available ports: {', '.join(available)}"
        return f"Port {port} not found. No serial ports detected."

    return str(exc)


def crc16_modbus(data: bytes) -> int:
    """Compute Modbus RTU CRC16 (polynomial 0xA001, init 0xFFFF)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def build_read_temperature_request(slave_id: int = DEFAULT_SLAVE_ID) -> bytes:
    """Build Modbus-RTU frame: read 1 holding register at address 0x0000."""
    payload = struct.pack(">BBHH", slave_id, 0x03, TEMPERATURE_REGISTER, 0x0001)
    crc = crc16_modbus(payload)
    return payload + struct.pack("<H", crc)


def _verify_crc(frame: bytes) -> None:
    payload, recv_crc = frame[:-2], struct.unpack("<H", frame[-2:])[0]
    if crc16_modbus(payload) != recv_crc:
        raise ValueError(f"CRC check failed on frame: {frame.hex(' ')}")


def parse_temperature_response(response: bytes, slave_id: int = DEFAULT_SLAVE_ID) -> float:
    """
    Parse Modbus-RTU read response and return temperature in Celsius.

    Register value is signed int16, scaled by 0.1 (divide raw value by 10).
    """
    if len(response) < 5:
        raise ValueError(f"response too short: {len(response)} bytes, data={response.hex(' ')}")

    resp_slave, func_code = response[0], response[1]
    if resp_slave != slave_id:
        raise ValueError(
            f"unexpected slave id: 0x{resp_slave:02X}, expected 0x{slave_id:02X}, data={response.hex(' ')}"
        )

    if func_code & 0x80:
        if len(response) < 5:
            raise ValueError(f"incomplete exception response: {response.hex(' ')}")
        frame = response[:5]
        _verify_crc(frame)
        raise RuntimeError(f"modbus exception code: 0x{frame[2]:02X}")

    if func_code != 0x03:
        raise ValueError(f"unexpected function code: 0x{func_code:02X}, data={response.hex(' ')}")

    if len(response) < 3:
        raise ValueError(f"incomplete response header: {response.hex(' ')}")

    byte_count = response[2]
    expected_len = 3 + byte_count + 2
    if len(response) < expected_len:
        raise ValueError(
            f"incomplete response: got {len(response)} bytes, need {expected_len}, data={response.hex(' ')}"
        )
    if byte_count != 2:
        raise ValueError(f"unexpected byte count: {byte_count}")

    frame = response[:expected_len]
    _verify_crc(frame)

    raw = struct.unpack(">h", frame[3:5])[0]
    return raw / 10.0


def read_modbus_frame(
    ser: serial.Serial,
    slave_id: int,
    timeout: float,
) -> bytes:
    """
    Read one Modbus RTU response frame from the serial port.

    Uses inter-byte timeout to detect frame boundaries and scans for a valid
    slave/function header in case of leading noise.
    """
    deadline = time.monotonic() + timeout
    buffer = bytearray()

    while time.monotonic() < deadline:
        waiting = ser.in_waiting
        chunk = ser.read(waiting if waiting else 1)
        if not chunk:
            continue

        buffer.extend(chunk)

        for start in range(max(0, len(buffer) - 64), len(buffer) - 1):
            if buffer[start] != slave_id:
                continue

            func_code = buffer[start + 1]
            if func_code == 0x83:
                end = start + 5
                if len(buffer) >= end:
                    return bytes(buffer[start:end])
            elif func_code == 0x03:
                if len(buffer) < start + 3:
                    continue
                expected_len = 3 + buffer[start + 2] + 2
                end = start + expected_len
                if len(buffer) >= end:
                    return bytes(buffer[start:end])

    raise TimeoutError(
        f"no complete Modbus response within {timeout:.1f}s; received={bytes(buffer).hex(' ') or '(empty)'}"
    )


class Rs485Thermometer:
    """Keep the serial port open for repeated temperature reads."""

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        slave_id: int = DEFAULT_SLAVE_ID,
        timeout: float = READ_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.slave_id = slave_id
        self.timeout = timeout
        self.retries = max(0, retries)
        self._ser: Optional[serial.Serial] = None

    def __enter__(self) -> "Rs485Thermometer":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._ser and self._ser.is_open:
            return

        inter_byte_timeout = modbus_inter_byte_timeout_s(self.baudrate)
        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            inter_byte_timeout=inter_byte_timeout,
            write_timeout=self.timeout,
            rtscts=False,
            dsrdtr=False,
        )
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        time.sleep(POST_OPEN_DELAY_S)

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def read(self, debug: bool = False) -> float:
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("serial port is not open")

        request = build_read_temperature_request(self.slave_id)
        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                self._ser.reset_input_buffer()
                self._ser.write(request)
                self._ser.flush()

                response = read_modbus_frame(self._ser, self.slave_id, self.timeout)
                if debug:
                    print(f"TX: {request.hex(' ')}", file=sys.stderr)
                    print(f"RX: {response.hex(' ')}", file=sys.stderr)

                return parse_temperature_response(response, self.slave_id)
            except (TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(modbus_char_time_s(self.baudrate) * 4)

        assert last_error is not None
        raise last_error


def read_temperature(
    port: str = DEFAULT_PORT,
    baudrate: int = DEFAULT_BAUDRATE,
    slave_id: int = DEFAULT_SLAVE_ID,
    timeout: float = READ_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    debug: bool = False,
) -> float:
    """Open serial port, send read request, parse and return temperature."""
    with Rs485Thermometer(
        port=port,
        baudrate=baudrate,
        slave_id=slave_id,
        timeout=timeout,
        retries=retries,
    ) as sensor:
        return sensor.read(debug=debug)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read RS485 infrared thermometer temperature")
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="baud rate")
    parser.add_argument("--slave-id", type=lambda x: int(x, 0), default=DEFAULT_SLAVE_ID, help="Modbus slave address")
    parser.add_argument("--timeout", type=float, default=READ_TIMEOUT_S, help="read timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="retry count on timeout/CRC errors")
    parser.add_argument("--list-ports", action="store_true", help="list available serial ports and exit")
    parser.add_argument("--debug", action="store_true", help="print request/response hex frames")
    args = parser.parse_args()

    if args.list_ports:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports found.")
        else:
            for device in ports:
                info = next((p for p in list_ports.comports() if p.device == device), None)
                desc = info.description if info else ""
                print(f"{device}\t{desc}")
        return 0

    try:
        temp_c = read_temperature(
            port=args.port,
            baudrate=args.baudrate,
            slave_id=args.slave_id,
            timeout=args.timeout,
            retries=args.retries,
            debug=args.debug,
        )
    except serial.SerialException as exc:
        print(f"serial error: {format_serial_error(exc, args.port)}", file=sys.stderr)
        return 1
    except (TimeoutError, ValueError, RuntimeError) as exc:
        print(f"read failed: {exc}", file=sys.stderr)
        return 1

    print(f"Temperature: {temp_c:.1f} °C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
