# -*- coding: utf-8 -*-

import serial
import time


PORT = "/dev/ttyS9"
BAUDRATE = 115200
MAX_SPEED = 60


def timestamp_ms() -> str:
    now = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)) + f".{int((now % 1) * 1000):03d}"


def speed_meaning(speed) -> str:
    speed_int = round(float(speed))
    if speed_int == 0:
        return "stop motor"
    if speed_int > 0:
        return f"set motor speed +{speed_int} turn right"
    return f"set motor speed {speed_int} turn left"


def log_serial_send(meaning: str, frame: bytes, port=None, written=None) -> None:
    port_text = f" port={port}" if port else ""
    written_text = f" bytes={written}" if written is not None else ""
    print(
        f"[serial][{timestamp_ms()}]{port_text} meaning={meaning}{written_text} "
        f"hex={frame.hex(' ').upper()}",
        flush=True,
    )


def build_speed_cmd(speed) -> bytes:
    """Build the motor speed command.

    Positive speed turns right, negative speed turns left, and 0 stops.
    The absolute speed must be no greater than MAX_SPEED.
    """
    speed = round(float(speed))

    if abs(speed) > MAX_SPEED:
        raise ValueError(f"speed absolute value must be <= {MAX_SPEED}")

    if speed == 0:
        return bytes.fromhex("01 FE 98 00 6B")

    direction = 0x01 if speed > 0 else 0x00
    speed_abs = abs(speed)
    speed_high = (speed_abs >> 8) & 0xFF
    speed_low = speed_abs & 0xFF

    return bytes([
        0x01,
        0xF6,
        direction,
        speed_high,
        speed_low,
        0x0A,
        0x00,
        0x6B,
    ])


def send_speed(speed, port=PORT, baudrate=BAUDRATE, recv=False):
    """Send one motor speed command."""
    cmd = build_speed_cmd(speed)

    ser = serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=0.5,
    )

    try:
        written = ser.write(cmd)
        log_serial_send(speed_meaning(speed), cmd, port, written)
        print("input speed:", speed)
        print("actual integer speed:", round(float(speed)))
        print("send HEX:", cmd.hex(" ").upper())

        if recv:
            time.sleep(0.1)

            if ser.in_waiting > 0:
                data = ser.read(ser.in_waiting)
                print("recv HEX:", data.hex(" ").upper())
                return data

            print("no response")
            return None

        return None
    finally:
        ser.close()


if __name__ == "__main__":
    send_speed(10.6)
