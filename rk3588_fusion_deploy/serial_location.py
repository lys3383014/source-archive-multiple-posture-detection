#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
import time

import serial


DEFAULT_PORT = "/dev/ttyS9"
DEFAULT_BAUDRATE = 115200
CURRENT_ZERO_FRAME = bytes.fromhex("01 93 88 01 6B")
HOME_FRAME = bytes.fromhex("01 9A 00 00 6B")


def timestamp_ms() -> str:
    now = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"


def log_serial_send(meaning: str, frame: bytes, port: str | None = None, written: int | None = None) -> None:
    port_text = f" port={port}" if port else ""
    written_text = f" bytes={written}" if written is not None else ""
    print(
        f"[serial][{timestamp_ms()}]{port_text} meaning={meaning}{written_text} "
        f"hex={frame.hex(' ').upper()}",
        flush=True,
    )


def write_frame(ser: serial.Serial, frame: bytes, meaning: str) -> None:
    written = ser.write(frame)
    log_serial_send(meaning, frame, getattr(ser, "port", None), written)


def open_serial(port: str = DEFAULT_PORT, baudrate: int = DEFAULT_BAUDRATE, timeout: float = 1.0) -> serial.Serial:
    return serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=timeout,
    )


def build_location_cmd(angle: float, relative: bool = False) -> bytes:
    mm = 0x01 if angle > 0 else 0x00
    int_val = round((abs(float(angle)) / 225.0) * 32000.0)
    data_bytes = struct.pack(">I", int_val)
    mode_byte = 0x00 if relative else 0x01
    return bytes([0x01, 0xFD, mm, 0x02, 0xDC, 0x00]) + data_bytes + bytes([mode_byte, 0x00, 0x6B])


def set_current_zero(ser: serial.Serial, delay_sec: float = 0.02) -> None:
    write_frame(ser, CURRENT_ZERO_FRAME, "set current motor position as zero")
    if delay_sec > 0:
        time.sleep(delay_sec)


def move_to_angle(ser: serial.Serial, angle: float, settle_sec: float = 2.0) -> None:
    write_frame(ser, HOME_FRAME, "home motor before target angle")
    if settle_sec > 0:
        time.sleep(settle_sec)
    write_frame(
        ser,
        build_location_cmd(angle, relative=True),
        f"move motor from zero to target angle {float(angle):.2f} deg",
    )


def move_relative_angle(ser: serial.Serial, angle: float) -> None:
    write_frame(ser, build_location_cmd(angle, relative=True), f"move motor relative {float(angle):+.2f} deg")


def main() -> int:
    parser = argparse.ArgumentParser(description="Move the motor to a target angle.")
    parser.add_argument("angle", nargs="?", type=float, default=0.0)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--relative", action="store_true")
    parser.add_argument("--set-zero", action="store_true")
    args = parser.parse_args()

    try:
        ser = open_serial(args.port, args.baudrate)
    except serial.SerialException:
        return 1

    try:
        if args.set_zero:
            set_current_zero(ser)
        elif args.relative:
            move_relative_angle(ser, args.angle)
        else:
            move_to_angle(ser, args.angle, args.settle_sec)
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
