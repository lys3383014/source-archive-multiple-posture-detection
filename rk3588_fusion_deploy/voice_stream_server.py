from __future__ import annotations

import argparse
import json
import socket
import socketserver
import subprocess
import threading
import time
from typing import BinaryIO

from audio_priority import (
    DEFAULT_AUDIO_CONTROL_FILE,
    is_audio_owner_active,
    mark_audio_owner,
)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8891
DEFAULT_CAPTURE_DEVICE = "plughw:rockchipnau8822,0"
DEFAULT_PLAY_DEVICE = "plughw:1,0"
DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2
DEFAULT_FRAME_MS = 40
DEFAULT_AUDIO_GAIN = 8.0
HEADER_LIMIT = 2048


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def amplify_pcm_frames(frames: bytes, sample_width: int, gain: float) -> bytes:
    if gain == 1.0:
        return frames

    if sample_width == 1:
        return bytes(clamp_int((sample - 128) * gain + 128, 0, 255) for sample in frames)

    if sample_width == 2:
        out = bytearray(len(frames))
        for idx in range(0, len(frames) - 1, 2):
            sample = int.from_bytes(frames[idx:idx + 2], "little", signed=True)
            value = clamp_int(sample * gain, -32768, 32767)
            out[idx:idx + 2] = value.to_bytes(2, "little", signed=True)
        return bytes(out)

    raise ValueError(f"unsupported sample width: {sample_width}")


def safe_close_proc(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()


def read_header(conn: socket.socket) -> bytes:
    conn.settimeout(5.0)
    data = bytearray()
    try:
        while len(data) < HEADER_LIMIT:
            chunk = conn.recv(1)
            if not chunk:
                break
            data.extend(chunk)
            if b"\n\n" in data or b"\r\n\r\n" in data:
                break
    finally:
        conn.settimeout(None)
    return bytes(data)


def parse_header(data: bytes) -> dict[str, str]:
    text = data.decode("utf-8", errors="ignore")
    parsed: dict[str, str] = {}
    for raw_line in text.replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def send_http_json(conn: socket.socket, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    reason = "OK" if status == 200 else "ERROR"
    header = (
        f"HTTP/1.1 {status} {reason}\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Cache-Control: no-store\r\n"
        "Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "\r\n"
    ).encode("utf-8")
    conn.sendall(header + body)


class VoiceStreamTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, handler_class, args: argparse.Namespace):
        super().__init__(server_address, handler_class)
        self.args = args
        self.stop_event = threading.Event()


class VoiceStreamHandler(socketserver.BaseRequestHandler):
    server: VoiceStreamTCPServer

    def handle(self) -> None:
        conn: socket.socket = self.request
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        header = read_header(conn)
        if header.startswith(b"GET ") or header.startswith(b"OPTIONS "):
            send_http_json(conn, 200, {
                "ok": True,
                "service": "voice_stream",
                "port": self.server.args.port,
                "rate": self.server.args.rate,
                "channels": self.server.args.channels,
                "format": self.server.args.audio_format,
            })
            return

        fields = parse_header(header)
        role = fields.get("role", "").lower()
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        print(f"[voice-stream] {peer} role={role or '-'} connected", flush=True)
        try:
            if role == "downlink":
                self.handle_downlink(conn)
            elif role == "uplink":
                self.handle_uplink(conn)
            else:
                print(f"[voice-stream] {peer} bad role header={header!r}", flush=True)
        except Exception as exc:
            print(f"[voice-stream] {peer} failed: {exc}", flush=True)
        finally:
            print(f"[voice-stream] {peer} disconnected", flush=True)

    def frame_bytes(self) -> int:
        args = self.server.args
        size = int(args.rate * args.channels * args.sample_width * args.frame_ms / 1000.0)
        size -= size % max(1, args.channels * args.sample_width)
        return max(args.channels * args.sample_width, size)

    def capture_cmd(self) -> list[str]:
        args = self.server.args
        return [
            args.arecord_bin,
            "-D",
            args.capture_device,
            "-f",
            args.audio_format,
            "-r",
            str(args.rate),
            "-c",
            str(args.channels),
            "-t",
            "raw",
            "-",
        ]

    def play_cmd(self) -> list[str]:
        args = self.server.args
        return [
            args.aplay_bin,
            "-D",
            args.play_device,
            "-f",
            args.audio_format,
            "-r",
            str(args.rate),
            "-c",
            str(args.channels),
            "-t",
            "raw",
            "-",
        ]

    def handle_downlink(self, conn: socket.socket) -> None:
        args = self.server.args
        frame_bytes = self.frame_bytes()
        silence = b"\x00" * frame_bytes
        proc: subprocess.Popen | None = None
        stream: BinaryIO | None = None
        next_retry = 0.0

        try:
            while not self.server.stop_event.is_set():
                if is_audio_owner_active(args.audio_control_file, "ai_record"):
                    safe_close_proc(proc)
                    proc = None
                    stream = None
                    conn.sendall(silence)
                    time.sleep(args.frame_ms / 1000.0)
                    continue

                if proc is None or proc.poll() is not None or stream is None:
                    now = time.time()
                    if now < next_retry:
                        conn.sendall(silence)
                        time.sleep(args.frame_ms / 1000.0)
                        continue
                    proc = subprocess.Popen(
                        self.capture_cmd(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=0,
                    )
                    stream = proc.stdout
                    next_retry = now + args.capture_restart_sec
                    print("[voice-stream] capture started", flush=True)

                data = stream.read(frame_bytes)
                if not data:
                    code = proc.poll()
                    safe_close_proc(proc)
                    stderr = b""
                    if proc.stderr is not None:
                        try:
                            stderr = proc.stderr.read(300)
                        except Exception:
                            stderr = b""
                    print(
                        f"[voice-stream] capture stopped code={code} "
                        f"stderr={stderr.decode('utf-8', errors='ignore').strip()}",
                        flush=True,
                    )
                    proc = None
                    stream = None
                    continue

                if args.input_gain != 1.0:
                    data = amplify_pcm_frames(data, args.sample_width, args.input_gain)
                conn.sendall(data)
        finally:
            safe_close_proc(proc)

    def handle_uplink(self, conn: socket.socket) -> None:
        args = self.server.args
        frame_bytes = self.frame_bytes()
        proc: subprocess.Popen | None = None
        try:
            proc = subprocess.Popen(
                self.play_cmd(),
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            if proc.stdin is None:
                raise RuntimeError("aplay stdin is not available")
            print("[voice-stream] playback started", flush=True)

            while not self.server.stop_event.is_set():
                data = conn.recv(frame_bytes * 4)
                if not data:
                    break
                mark_audio_owner(
                    args.audio_control_file,
                    "phone_speaker",
                    "phone realtime audio",
                    args.phone_audio_hold_sec,
                )
                if args.output_gain != 1.0:
                    data = amplify_pcm_frames(data, args.sample_width, args.output_gain)
                proc.stdin.write(data)
        finally:
            if proc is not None and proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            safe_close_proc(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency RK3588 native Android voice stream server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--capture-device", default=DEFAULT_CAPTURE_DEVICE)
    parser.add_argument("--play-device", default=DEFAULT_PLAY_DEVICE)
    parser.add_argument("--audio-format", default="S16_LE")
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--sample-width", type=int, default=DEFAULT_SAMPLE_WIDTH)
    parser.add_argument("--frame-ms", type=int, default=DEFAULT_FRAME_MS)
    parser.add_argument("--audio-gain", type=float, default=None,
                        help="Set both input and output gain.")
    parser.add_argument("--input-gain", type=float, default=DEFAULT_AUDIO_GAIN)
    parser.add_argument("--output-gain", type=float, default=DEFAULT_AUDIO_GAIN)
    parser.add_argument("--audio-control-file", default=DEFAULT_AUDIO_CONTROL_FILE)
    parser.add_argument("--phone-audio-hold-sec", type=float, default=1.0)
    parser.add_argument("--capture-restart-sec", type=float, default=0.5)
    parser.add_argument("--arecord-bin", default="arecord")
    parser.add_argument("--aplay-bin", default="aplay")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.audio_gain is not None:
        args.input_gain = args.audio_gain
        args.output_gain = args.audio_gain
    args.port = max(1, min(65535, int(args.port)))
    args.rate = max(8000, int(args.rate))
    args.channels = max(1, int(args.channels))
    args.sample_width = max(1, min(2, int(args.sample_width)))
    args.frame_ms = max(10, min(200, int(args.frame_ms)))
    args.input_gain = max(0.0, float(args.input_gain))
    args.output_gain = max(0.0, float(args.output_gain))
    args.phone_audio_hold_sec = max(0.1, float(args.phone_audio_hold_sec))

    server = VoiceStreamTCPServer((args.host, args.port), VoiceStreamHandler, args)
    print(
        f"[voice-stream] listening on tcp://{args.host}:{args.port}, "
        f"mic={args.capture_device}, speaker={args.play_device}, "
        f"rate={args.rate}, channels={args.channels}, frame_ms={args.frame_ms}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop_event.set()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
