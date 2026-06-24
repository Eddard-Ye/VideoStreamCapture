# -*- coding: utf-8 -*-
"""LD-BG laser displacement sensor on RS485 (Modbus RTU)."""

from __future__ import annotations

import argparse
import struct
import sys
import time
from typing import Optional

try:
    import serial
except ImportError as exc:
    raise SystemExit("pyserial is required: pip install pyserial") from exc


DEFAULT_PORT = "COM6"
DEFAULT_BAUDRATE = 115200
DEFAULT_SLAVE_ID = 0x01
READ_TIMEOUT_S = 1.0
DEFAULT_RETRIES = 2
POST_OPEN_DELAY_S = 0.1

BLG_DISTANCE_INPUT_REGISTER = 0x0000
BLG_THRESHOLD_REGISTER = 0x000C
UM_PER_MM = 1000.0


def modbus_char_time_s(baudrate: int) -> float:
    return 11.0 / baudrate


def modbus_inter_byte_timeout_s(baudrate: int) -> float:
    return modbus_char_time_s(baudrate) * 3.5


def crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def build_modbus_read_request(slave_id: int, register: int, count: int = 2) -> bytes:
    payload = struct.pack(">BBHH", slave_id, 0x04, register, count)
    return payload + struct.pack("<H", crc16_modbus(payload))


def build_modbus_write_uint32_request(slave_id: int, register: int, value: int) -> bytes:
    payload = struct.pack(">BBHHB", slave_id, 0x10, register, 2, 4) + struct.pack(">I", value & 0xFFFFFFFF)
    return payload + struct.pack("<H", crc16_modbus(payload))


def _verify_crc(frame: bytes) -> None:
    payload, recv_crc = frame[:-2], struct.unpack("<H", frame[-2:])[0]
    if crc16_modbus(payload) != recv_crc:
        raise ValueError(f"CRC check failed on frame: {frame.hex(' ')}")


def parse_modbus_read_response(response: bytes, slave_id: int) -> bytes:
    if len(response) < 5:
        raise ValueError(f"response too short: {len(response)} bytes")

    resp_slave, func_code = response[0], response[1]
    if resp_slave != slave_id:
        raise ValueError(f"unexpected slave id: 0x{resp_slave:02X}")

    if func_code & 0x80:
        frame = response[:5]
        _verify_crc(frame)
        raise RuntimeError(f"modbus exception code: 0x{frame[2]:02X}")

    if func_code != 0x04:
        raise ValueError(f"unexpected function code: 0x{func_code:02X}")

    byte_count = response[2]
    expected_len = 3 + byte_count + 2
    if len(response) < expected_len:
        raise ValueError(f"incomplete response: got {len(response)} bytes, need {expected_len}")

    frame = response[:expected_len]
    _verify_crc(frame)
    return frame[3:-2]


def parse_modbus_write_response(response: bytes, slave_id: int, register: int, count: int) -> None:
    if len(response) < 8:
        raise ValueError(f"response too short: {len(response)} bytes")

    resp_slave, func_code = response[0], response[1]
    if resp_slave != slave_id:
        raise ValueError(f"unexpected slave id: 0x{resp_slave:02X}")

    if func_code == 0x90:
        frame = response[:5]
        _verify_crc(frame)
        raise RuntimeError(f"modbus exception code: 0x{frame[2]:02X}")

    if func_code != 0x10:
        raise ValueError(f"unexpected function code: 0x{func_code:02X}")

    frame = response[:8]
    _verify_crc(frame)
    resp_reg, resp_count = struct.unpack(">HH", frame[2:6])
    if resp_reg != register or resp_count != count:
        raise ValueError(f"unexpected write ack: reg=0x{resp_reg:04X} count={resp_count}")


def read_modbus_frame(ser: serial.Serial, slave_id: int, timeout: float, function: int) -> bytes:
    deadline = time.monotonic() + timeout
    buffer = bytearray()
    exception_func = function | 0x80

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
            if func_code == exception_func:
                end = start + 5
                if len(buffer) >= end:
                    return bytes(buffer[start:end])
            elif func_code == function:
                if function in (0x06, 0x10):
                    end = start + 8
                    if len(buffer) >= end:
                        return bytes(buffer[start:end])
                elif len(buffer) >= start + 3:
                    expected_len = 3 + buffer[start + 2] + 2
                    end = start + expected_len
                    if len(buffer) >= end:
                        return bytes(buffer[start:end])

    raise TimeoutError(f"no complete Modbus response within {timeout:.1f}s")


def um_to_mm(raw_um: int) -> float:
    return raw_um / UM_PER_MM


def mm_to_um(distance_mm: float) -> int:
    return int(round(distance_mm * UM_PER_MM))


def parse_uint32_um_to_mm(data: bytes) -> float:
    if len(data) != 4:
        raise ValueError(f"expected 4 data bytes, got {len(data)}")
    return um_to_mm(struct.unpack(">I", data)[0])


class LaserDistanceSensor:
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

    def __enter__(self) -> "LaserDistanceSensor":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._ser and self._ser.is_open:
            return

        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            inter_byte_timeout=modbus_inter_byte_timeout_s(self.baudrate),
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

    def _modbus_read(self, register: int, count: int = 2) -> bytes:
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("serial port is not open")

        request = build_modbus_read_request(self.slave_id, register, count)
        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                self._ser.reset_input_buffer()
                self._ser.write(request)
                self._ser.flush()
                response = read_modbus_frame(self._ser, self.slave_id, self.timeout, 0x04)
                return parse_modbus_read_response(response, self.slave_id)
            except (TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(modbus_char_time_s(self.baudrate) * 4)

        assert last_error is not None
        raise last_error

    def _modbus_write(self, request: bytes, register: int, count: int) -> None:
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("serial port is not open")

        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                self._ser.reset_input_buffer()
                self._ser.write(request)
                self._ser.flush()
                response = read_modbus_frame(self._ser, self.slave_id, self.timeout, 0x10)
                parse_modbus_write_response(response, self.slave_id, register, count)
                return
            except (TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(modbus_char_time_s(self.baudrate) * 4)

        assert last_error is not None
        raise last_error

    def read_current_distance_mm(self) -> float:
        return parse_uint32_um_to_mm(self._modbus_read(BLG_DISTANCE_INPUT_REGISTER))

    def read_threshold_mm(self) -> float:
        return parse_uint32_um_to_mm(self._modbus_read(BLG_THRESHOLD_REGISTER))

    def write_threshold_mm(self, threshold_mm: float) -> None:
        raw_um = mm_to_um(threshold_mm)
        request = build_modbus_write_uint32_request(self.slave_id, BLG_THRESHOLD_REGISTER, raw_um)
        self._modbus_write(request, BLG_THRESHOLD_REGISTER, 2)

    def calibrate(self) -> float:
        distance_mm = self.read_current_distance_mm()
        self.write_threshold_mm(distance_mm)
        return distance_mm

    def read_actual_mm(self) -> float:
        threshold_mm = self.read_threshold_mm()
        distance_mm = self.read_current_distance_mm()
        return threshold_mm - distance_mm


def print_mm(value: float) -> None:
    print(f"{value:.3f}")


def add_serial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--slave-id", type=lambda x: int(x, 0), default=DEFAULT_SLAVE_ID)
    parser.add_argument("--timeout", type=float, default=READ_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)


def main() -> int:
    parser = argparse.ArgumentParser(description="LD-BG laser displacement sensor")
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser("calibrate", help="read current distance and store as threshold")
    add_serial_args(calibrate_parser)

    get_calibration_parser = subparsers.add_parser("get-calibration", help="read stored threshold")
    add_serial_args(get_calibration_parser)

    get_actual_parser = subparsers.add_parser("get-actual", help="threshold minus current distance")
    add_serial_args(get_actual_parser)

    args = parser.parse_args()

    try:
        with LaserDistanceSensor(
            port=args.port,
            baudrate=args.baudrate,
            slave_id=args.slave_id,
            timeout=args.timeout,
            retries=args.retries,
        ) as sensor:
            if args.command == "calibrate":
                print_mm(sensor.calibrate())
            elif args.command == "get-calibration":
                print_mm(sensor.read_threshold_mm())
            elif args.command == "get-actual":
                print_mm(sensor.read_actual_mm())
    except (serial.SerialException, TimeoutError, ValueError, RuntimeError):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
