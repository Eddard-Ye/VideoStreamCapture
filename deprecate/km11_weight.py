# -*- coding: utf-8 -*-
"""Read net weight, tare, and zero the Kunhong KM-11 transmitter via Modbus RTU."""

from __future__ import annotations

import argparse
import struct
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:
    raise SystemExit("pyserial is required: pip install pyserial") from exc


DEFAULT_PORT = "COM6"
DEFAULT_BAUDRATE = 38400
DEFAULT_SLAVE_ID = 0x01
NET_WEIGHT_REGISTER = 0x0000
TARE_WEIGHT_REGISTER = 0x0006
DECIMAL_REGISTER = 0x0003
STATUS_REGISTER = 0x0002
ZERO_COMMAND_REGISTER = 0x001E
ZERO_COMMAND_VALUE = 0x0001
READ_TIMEOUT_S = 2.0
DEFAULT_RETRIES = 2
POST_OPEN_DELAY_S = 0.1
POST_COMMAND_SETTLE_S = 0.35


@dataclass(frozen=True)
class WeightReading:
    raw: int
    decimal_places: int
    value: float
    status: int

    @property
    def value_kg(self) -> float:
        return self.value


@dataclass(frozen=True)
class TareResult:
    saved_net: WeightReading
    tare_raw: int
    tare_value: float
    net: WeightReading


def modbus_char_time_s(baudrate: int) -> float:
    return 11.0 / baudrate


def modbus_inter_byte_timeout_s(baudrate: int) -> float:
    return modbus_char_time_s(baudrate) * 3.5


def list_serial_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def format_serial_error(exc: serial.SerialException, port: str) -> str:
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
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def build_read_holding_request(slave_id: int, register: int, count: int) -> bytes:
    payload = struct.pack(">BBHH", slave_id, 0x03, register, count)
    return payload + struct.pack("<H", crc16_modbus(payload))


def build_write_single_request(slave_id: int, register: int, value: int) -> bytes:
    payload = struct.pack(">BBHH", slave_id, 0x06, register, value & 0xFFFF)
    return payload + struct.pack("<H", crc16_modbus(payload))


def build_write_word_swapped_u32_request(slave_id: int, register: int, value: int) -> bytes:
    encoded = value & 0xFFFFFFFF
    reg0 = encoded & 0xFFFF
    reg1 = (encoded >> 16) & 0xFFFF
    payload = struct.pack(">BBHHB", slave_id, 0x10, register, 2, 4) + struct.pack(">HH", reg0, reg1)
    return payload + struct.pack("<H", crc16_modbus(payload))


def _verify_crc(frame: bytes) -> None:
    payload, recv_crc = frame[:-2], struct.unpack("<H", frame[-2:])[0]
    if crc16_modbus(payload) != recv_crc:
        raise ValueError(f"CRC check failed on frame: {frame.hex(' ')}")


def parse_holding_read_response(response: bytes, slave_id: int) -> bytes:
    if len(response) < 5:
        raise ValueError(f"response too short: {len(response)} bytes, data={response.hex(' ')}")

    resp_slave, func_code = response[0], response[1]
    if resp_slave != slave_id:
        raise ValueError(
            f"unexpected slave id: 0x{resp_slave:02X}, expected 0x{slave_id:02X}, data={response.hex(' ')}"
        )

    if func_code & 0x80:
        frame = response[:5]
        _verify_crc(frame)
        raise RuntimeError(f"modbus exception code: 0x{frame[2]:02X}")

    if func_code != 0x03:
        raise ValueError(f"unexpected function code: 0x{func_code:02X}, data={response.hex(' ')}")

    byte_count = response[2]
    expected_len = 3 + byte_count + 2
    if len(response) < expected_len:
        raise ValueError(
            f"incomplete response: got {len(response)} bytes, need {expected_len}, data={response.hex(' ')}"
        )

    frame = response[:expected_len]
    _verify_crc(frame)
    return frame[3:-2]


def parse_write_single_response(response: bytes, slave_id: int, register: int, value: int) -> None:
    if len(response) < 8:
        raise ValueError(f"response too short: {len(response)} bytes, data={response.hex(' ')}")

    resp_slave, func_code = response[0], response[1]
    if resp_slave != slave_id:
        raise ValueError(
            f"unexpected slave id: 0x{resp_slave:02X}, expected 0x{slave_id:02X}, data={response.hex(' ')}"
        )

    if func_code == 0x86:
        frame = response[:5]
        _verify_crc(frame)
        raise RuntimeError(f"modbus exception code: 0x{frame[2]:02X}")

    if func_code != 0x06:
        raise ValueError(f"unexpected function code: 0x{func_code:02X}, data={response.hex(' ')}")

    frame = response[:8]
    _verify_crc(frame)
    resp_reg, resp_value = struct.unpack(">HH", frame[2:6])
    if resp_reg != register or resp_value != (value & 0xFFFF):
        raise ValueError(
            f"unexpected write ack: reg=0x{resp_reg:04X} value=0x{resp_value:04X}, "
            f"expected reg=0x{register:04X} value=0x{value & 0xFFFF:04X}"
        )


def parse_write_multiple_response(response: bytes, slave_id: int, register: int, count: int) -> None:
    if len(response) < 8:
        raise ValueError(f"response too short: {len(response)} bytes, data={response.hex(' ')}")

    resp_slave, func_code = response[0], response[1]
    if resp_slave != slave_id:
        raise ValueError(
            f"unexpected slave id: 0x{resp_slave:02X}, expected 0x{slave_id:02X}, data={response.hex(' ')}"
        )

    if func_code == 0x90:
        frame = response[:5]
        _verify_crc(frame)
        raise RuntimeError(f"modbus exception code: 0x{frame[2]:02X}")

    if func_code != 0x10:
        raise ValueError(f"unexpected function code: 0x{func_code:02X}, data={response.hex(' ')}")

    frame = response[:8]
    _verify_crc(frame)
    resp_reg, resp_count = struct.unpack(">HH", frame[2:6])
    if resp_reg != register or resp_count != count:
        raise ValueError(
            f"unexpected write ack: reg=0x{resp_reg:04X} count={resp_count}, "
            f"expected reg=0x{register:04X} count={count}"
        )


def parse_word_swapped_u32(data: bytes) -> int:
    if len(data) < 4:
        raise ValueError(f"expected 4 bytes, got {len(data)}")
    reg0, reg1 = struct.unpack(">HH", data[:4])
    return ((reg1 & 0xFFFF) << 16) | (reg0 & 0xFFFF)


def parse_word_swapped_s32(data: bytes) -> int:
    raw = parse_word_swapped_u32(data)
    if raw >= 0x80000000:
        raw -= 0x100000000
    return raw


def parse_u16(data: bytes) -> int:
    if len(data) < 2:
        raise ValueError(f"expected 2 bytes, got {len(data)}")
    return struct.unpack(">H", data[:2])[0]


def scale_weight(raw: int, decimal_places: int) -> float:
    if decimal_places <= 0:
        return float(raw)
    return raw / float(10**decimal_places)


def read_modbus_frame(
    ser: serial.Serial,
    slave_id: int,
    timeout: float,
    function: int = 0x03,
) -> bytes:
    deadline = time.monotonic() + timeout
    buffer = bytearray()
    exception_func = function | 0x80
    fixed_response_len = 8 if function in (0x06, 0x10) else None

    while time.monotonic() < deadline:
        chunk = ser.read(ser.in_waiting or 1)
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
                if fixed_response_len is not None:
                    end = start + fixed_response_len
                    if len(buffer) >= end:
                        return bytes(buffer[start:end])
                    continue

                if len(buffer) < start + 3:
                    continue
                expected_len = 3 + buffer[start + 2] + 2
                end = start + expected_len
                if len(buffer) >= end:
                    return bytes(buffer[start:end])

    raise TimeoutError(
        f"no complete Modbus response within {timeout:.1f}s; received={bytes(buffer).hex(' ') or '(empty)'}"
    )


class Km11WeightTransmitter:
    """KM-11 net-weight read, tare, and zero over a persistent serial connection."""

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

    def __enter__(self) -> "Km11WeightTransmitter":
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

    def _read_registers(self, register: int, count: int, debug: bool = False) -> bytes:
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("serial port is not open")

        request = build_read_holding_request(self.slave_id, register, count)
        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                self._ser.reset_input_buffer()
                self._ser.write(request)
                self._ser.flush()

                response = read_modbus_frame(self._ser, self.slave_id, self.timeout, 0x03)
                if debug:
                    print(f"TX: {request.hex(' ')}", file=sys.stderr)
                    print(f"RX: {response.hex(' ')}", file=sys.stderr)

                return parse_holding_read_response(response, self.slave_id)
            except (TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(modbus_char_time_s(self.baudrate) * 4)

        assert last_error is not None
        raise last_error

    def _write_register(self, register: int, value: int, debug: bool = False) -> None:
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("serial port is not open")

        request = build_write_single_request(self.slave_id, register, value)
        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                self._ser.reset_input_buffer()
                self._ser.write(request)
                self._ser.flush()

                response = read_modbus_frame(self._ser, self.slave_id, self.timeout, 0x06)
                if debug:
                    print(f"TX: {request.hex(' ')}", file=sys.stderr)
                    print(f"RX: {response.hex(' ')}", file=sys.stderr)

                parse_write_single_response(response, self.slave_id, register, value)
                return
            except (TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(modbus_char_time_s(self.baudrate) * 4)

        assert last_error is not None
        raise last_error

    def _write_word_swapped_u32(self, register: int, value: int, debug: bool = False) -> None:
        if not self._ser or not self._ser.is_open:
            raise RuntimeError("serial port is not open")

        request = build_write_word_swapped_u32_request(self.slave_id, register, value)
        last_error: Optional[Exception] = None

        for attempt in range(self.retries + 1):
            try:
                self._ser.reset_input_buffer()
                self._ser.write(request)
                self._ser.flush()

                response = read_modbus_frame(self._ser, self.slave_id, self.timeout, 0x10)
                if debug:
                    print(f"TX: {request.hex(' ')}", file=sys.stderr)
                    print(f"RX: {response.hex(' ')}", file=sys.stderr)

                parse_write_multiple_response(response, self.slave_id, register, 2)
                return
            except (TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(modbus_char_time_s(self.baudrate) * 4)

        assert last_error is not None
        raise last_error

    def _read_tare_raw(self, debug: bool = False) -> int:
        return parse_word_swapped_s32(self._read_registers(TARE_WEIGHT_REGISTER, 2, debug=debug))

    def read_net(self, debug: bool = False) -> WeightReading:
        """Read net weight from 40001/40002 (gross minus stored tare)."""
        net_data = self._read_registers(NET_WEIGHT_REGISTER, 2, debug=debug)
        decimal_data = self._read_registers(DECIMAL_REGISTER, 1, debug=debug)
        status_data = self._read_registers(STATUS_REGISTER, 1, debug=debug)

        raw = parse_word_swapped_s32(net_data)
        decimal_places = parse_u16(decimal_data)
        status = parse_u16(status_data)

        return WeightReading(
            raw=raw,
            decimal_places=decimal_places,
            value=scale_weight(raw, decimal_places),
            status=status,
        )

    def read(self, debug: bool = False) -> WeightReading:
        return self.read_net(debug=debug)

    def tare(self, debug: bool = False) -> TareResult:
        """
        Execute tare (去皮).

        Net weight from read is the current displayed weight (gross minus tare).
        Tare saves that reading into the tare-weight register (40007/40008):
        new_tare = old_tare + current_net, so net weight becomes 0 afterward.
        """
        saved_net = self.read_net(debug=debug)
        old_tare_raw = self._read_tare_raw(debug=debug)
        new_tare_raw = old_tare_raw + saved_net.raw

        self._write_word_swapped_u32(TARE_WEIGHT_REGISTER, new_tare_raw, debug=debug)
        time.sleep(POST_COMMAND_SETTLE_S)

        net_after = self.read_net(debug=debug)
        dp = saved_net.decimal_places
        return TareResult(
            saved_net=saved_net,
            tare_raw=new_tare_raw,
            tare_value=scale_weight(new_tare_raw, dp),
            net=net_after,
        )

    def zero(self, debug: bool = False) -> WeightReading:
        """Zero the current reading on register 40031."""
        self._write_register(ZERO_COMMAND_REGISTER, ZERO_COMMAND_VALUE, debug=debug)
        time.sleep(POST_COMMAND_SETTLE_S)
        return self.read_net(debug=debug)


def read_weight(
    port: str = DEFAULT_PORT,
    baudrate: int = DEFAULT_BAUDRATE,
    slave_id: int = DEFAULT_SLAVE_ID,
    timeout: float = READ_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    debug: bool = False,
) -> WeightReading:
    with Km11WeightTransmitter(
        port=port,
        baudrate=baudrate,
        slave_id=slave_id,
        timeout=timeout,
        retries=retries,
    ) as sensor:
        return sensor.read_net(debug=debug)


def tare_weight(
    port: str = DEFAULT_PORT,
    baudrate: int = DEFAULT_BAUDRATE,
    slave_id: int = DEFAULT_SLAVE_ID,
    timeout: float = READ_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    debug: bool = False,
) -> TareResult:
    with Km11WeightTransmitter(
        port=port,
        baudrate=baudrate,
        slave_id=slave_id,
        timeout=timeout,
        retries=retries,
    ) as sensor:
        return sensor.tare(debug=debug)


def zero_weight(
    port: str = DEFAULT_PORT,
    baudrate: int = DEFAULT_BAUDRATE,
    slave_id: int = DEFAULT_SLAVE_ID,
    timeout: float = READ_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    debug: bool = False,
) -> WeightReading:
    with Km11WeightTransmitter(
        port=port,
        baudrate=baudrate,
        slave_id=slave_id,
        timeout=timeout,
        retries=retries,
    ) as sensor:
        return sensor.zero(debug=debug)


def add_serial_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--port", default=DEFAULT_PORT, help=f"serial port (default: {DEFAULT_PORT})")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE, help="baud rate")
    parser.add_argument(
        "--slave-id",
        type=lambda x: int(x, 0),
        default=DEFAULT_SLAVE_ID,
        help="Modbus slave address (default: 0x01 / 1)",
    )
    parser.add_argument("--timeout", type=float, default=READ_TIMEOUT_S, help="read timeout in seconds")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="retry count on timeout/CRC errors")
    parser.add_argument("--debug", action="store_true", help="print request/response hex frames")


def print_net_weight(reading: WeightReading) -> None:
    dp = reading.decimal_places
    print(
        f"Net weight: {reading.value:.{dp}f} "
        f"(raw={reading.raw}, decimal={dp}, status=0x{reading.status:04X})"
    )


def print_tare_result(result: TareResult) -> None:
    dp = result.saved_net.decimal_places
    print(
        f"Saved net weight as tare: {result.saved_net.value:.{dp}f} "
        f"(raw={result.saved_net.raw})"
    )
    print(
        f"Total tare weight: {result.tare_value:.{dp}f} "
        f"(raw={result.tare_raw}, decimal={dp})"
    )
    print_net_weight(result.net)


def main() -> int:
    argv = sys.argv[1:]
    if "--list-ports" in argv:
        ports = list_serial_ports()
        if not ports:
            print("No serial ports found.")
        else:
            for device in ports:
                info = next((p for p in list_ports.comports() if p.device == device), None)
                desc = info.description if info else ""
                print(f"{device}\t{desc}")
        return 0

    if not argv or argv[0] not in {"read", "tare", "zero"}:
        argv = ["read", *argv]

    parser = argparse.ArgumentParser(description="KM-11 weighing transmitter over Modbus RTU")
    subparsers = parser.add_subparsers(dest="command", required=True)

    read_parser = subparsers.add_parser("read", help="read net weight from 40001/40002 (default)")
    add_serial_args(read_parser)

    tare_parser = subparsers.add_parser(
        "tare",
        help="read net weight, then save it into tare-weight register 40007/40008",
    )
    add_serial_args(tare_parser)

    zero_parser = subparsers.add_parser("zero", help="zero current reading on register 40031")
    add_serial_args(zero_parser)

    args = parser.parse_args(argv)

    try:
        with Km11WeightTransmitter(
            port=args.port,
            baudrate=args.baudrate,
            slave_id=args.slave_id,
            timeout=args.timeout,
            retries=args.retries,
        ) as sensor:
            if args.command == "tare":
                print_tare_result(sensor.tare(debug=args.debug))
            elif args.command == "zero":
                print("Zero command sent.")
                print_net_weight(sensor.zero(debug=args.debug))
            else:
                print_net_weight(sensor.read_net(debug=args.debug))
    except serial.SerialException as exc:
        print(f"serial error: {format_serial_error(exc, args.port)}", file=sys.stderr)
        return 1
    except (TimeoutError, ValueError, RuntimeError) as exc:
        print(f"command failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
