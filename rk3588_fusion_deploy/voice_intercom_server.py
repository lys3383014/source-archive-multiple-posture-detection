from __future__ import annotations

import argparse
import io
import json
import queue
import subprocess
import tempfile
import threading
import time
import wave
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from audio_priority import (
    DEFAULT_AUDIO_CONTROL_FILE,
    clear_audio_owner,
    is_audio_owner_active,
    mark_audio_owner,
)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8890
DEFAULT_CAPTURE_DEVICE = "hw:rockchipnau8822,0"
DEFAULT_PLAY_DEVICE = "plughw:1,0"
DEFAULT_CHUNK_SECONDS = 1
DEFAULT_AUDIO_GAIN = 8.0
MAX_CHUNK_SECONDS = 5
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
CAPTURE_RESTART_DELAY = 1.0


class VoiceState:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.capture = ContinuousCapture(args)
        self.play_queue: queue.Queue[Path | None] = queue.Queue(maxsize=args.play_queue_size)
        self.current_play: subprocess.Popen | None = None
        self.play_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self.play_worker, daemon=True)
        self.worker.start()

    def enqueue_play(self, path: Path) -> bool:
        try:
            self.play_queue.put_nowait(path)
            return True
        except queue.Full:
            try:
                old_path = self.play_queue.get_nowait()
                if old_path is not None:
                    old_path.unlink(missing_ok=True)
            except queue.Empty:
                pass
            try:
                self.play_queue.put_nowait(path)
                return True
            except queue.Full:
                path.unlink(missing_ok=True)
                return False

    def clear_play_queue(self) -> None:
        while True:
            try:
                path = self.play_queue.get_nowait()
            except queue.Empty:
                break
            if path is not None:
                path.unlink(missing_ok=True)

    def stop_playback(self) -> None:
        self.clear_play_queue()
        with self.play_lock:
            if self.current_play and self.current_play.poll() is None:
                self.current_play.terminate()

    def play_worker(self) -> None:
        while not self.stop_event.is_set():
            path = self.play_queue.get()
            if path is None:
                return

            try:
                cmd = [
                    self.args.gst_launch_bin,
                    "-q",
                    "filesrc",
                    f"location={path}",
                    "!",
                    "decodebin",
                    "!",
                    "audioconvert",
                    "!",
                    "audioresample",
                    "!",
                    "volume",
                    f"volume={self.args.output_gain}",
                    "!",
                    "alsasink",
                    f"device={self.args.play_device}",
                ]
                with self.play_lock:
                    self.current_play = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    proc = self.current_play
                start_time = time.time()
                while proc.poll() is None:
                    mark_audio_owner(
                        self.args.audio_control_file,
                        "phone_speaker",
                        "phone audio playing",
                        1.0,
                    )
                    if time.time() - start_time > self.args.play_timeout:
                        raise subprocess.TimeoutExpired(cmd, self.args.play_timeout)
                    time.sleep(0.1)
            except subprocess.TimeoutExpired:
                with self.play_lock:
                    if self.current_play and self.current_play.poll() is None:
                        self.current_play.kill()
            except Exception as exc:
                print(f"[voice] play failed: {exc}", flush=True)
            finally:
                with self.play_lock:
                    self.current_play = None
                path.unlink(missing_ok=True)

    def close(self) -> None:
        self.stop_event.set()
        self.capture.close()
        self.stop_playback()
        try:
            self.play_queue.put_nowait(None)
        except queue.Full:
            pass


def json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class ContinuousCapture:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.frame_size = int(args.capture_channels) * int(args.capture_sample_width)
        self.bytes_per_second = int(args.capture_rate) * self.frame_size
        self.max_buffer_bytes = max(self.bytes_per_second, int(self.bytes_per_second * args.capture_buffer_sec))
        self.buffer = bytearray()
        self.start_seq = 0
        self.write_seq = 0
        self.last_error: str | None = None
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.cond = threading.Condition()

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        block_bytes = max(self.frame_size, int(self.bytes_per_second * self.args.capture_block_ms / 1000.0))
        block_bytes -= block_bytes % self.frame_size
        if block_bytes <= 0:
            block_bytes = self.frame_size

        while not self.stop_event.is_set():
            if is_audio_owner_active(self.args.audio_control_file, "ai_record"):
                with self.cond:
                    self.last_error = "capture paused for AI recording"
                    self.cond.notify_all()
                time.sleep(0.1)
                continue

            cmd = [
                self.args.arecord_bin,
                "-D",
                self.args.capture_device,
                "-f",
                self.args.capture_format,
                "-r",
                str(self.args.capture_rate),
                "-c",
                str(self.args.capture_channels),
                "-t",
                "raw",
                "-",
            ]
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                with self.cond:
                    self.last_error = None
                    self.cond.notify_all()

                assert self.proc.stdout is not None
                while not self.stop_event.is_set():
                    if is_audio_owner_active(self.args.audio_control_file, "ai_record"):
                        print("[voice] capture paused for AI recording", flush=True)
                        self.proc.terminate()
                        break
                    chunk = self.proc.stdout.read(block_bytes)
                    if not chunk:
                        break
                    self._append(chunk)

                stderr = b""
                if self.proc.stderr is not None:
                    try:
                        stderr = self.proc.stderr.read(500)
                    except Exception:
                        stderr = b""
                code = self.proc.poll()
                if not self.stop_event.is_set():
                    message = stderr.decode("utf-8", errors="ignore").strip()
                    if is_audio_owner_active(self.args.audio_control_file, "ai_record"):
                        message = "capture paused for AI recording"
                    with self.cond:
                        self.last_error = message or f"arecord stopped: {code}"
                        self.cond.notify_all()
                    print(f"[voice] capture stopped: {self.last_error}", flush=True)
            except Exception as exc:
                with self.cond:
                    self.last_error = str(exc)
                    self.cond.notify_all()
                print(f"[voice] capture failed: {exc}", flush=True)
            finally:
                if self.proc and self.proc.poll() is None:
                    self.proc.terminate()
                self.proc = None

            if not self.stop_event.is_set():
                time.sleep(CAPTURE_RESTART_DELAY)

    def _append(self, chunk: bytes) -> None:
        with self.cond:
            self.buffer.extend(chunk)
            self.write_seq += len(chunk)
            overflow = len(self.buffer) - self.max_buffer_bytes
            if overflow > 0:
                overflow -= overflow % self.frame_size
                if overflow > 0:
                    del self.buffer[:overflow]
                    self.start_seq += overflow
            self.cond.notify_all()

    def read_chunk(self, duration: float, cursor: int | None, timeout: float) -> tuple[bytes, int, bool, str | None]:
        chunk_bytes = int(duration * self.bytes_per_second)
        chunk_bytes -= chunk_bytes % self.frame_size
        chunk_bytes = max(self.frame_size, chunk_bytes)
        deadline = time.time() + timeout
        reset = False

        with self.cond:
            if cursor is None:
                while self.write_seq - self.start_seq < chunk_bytes and time.time() < deadline:
                    self.cond.wait(timeout=max(0.05, deadline - time.time()))
                start = max(self.start_seq, self.write_seq - chunk_bytes)
            else:
                start = int(cursor)
                if start < self.start_seq:
                    start = self.start_seq
                    reset = True
                if start > self.write_seq:
                    start = self.write_seq
                    reset = True

                while self.write_seq < start + chunk_bytes and time.time() < deadline:
                    self.cond.wait(timeout=max(0.05, deadline - time.time()))

            end = min(start + chunk_bytes, self.write_seq)
            if end <= start:
                return b"", start, reset, self.last_error or "no audio captured"

            offset_start = max(0, start - self.start_seq)
            offset_end = max(offset_start, end - self.start_seq)
            return bytes(self.buffer[offset_start:offset_end]), end, reset, self.last_error

    def close(self) -> None:
        self.stop_event.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        with self.cond:
            self.cond.notify_all()


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def amplify_pcm_frames(frames: bytes, sample_width: int, gain: float) -> bytes:
    if gain == 1.0:
        return frames

    if sample_width == 1:
        return bytes(clamp_int((sample - 128) * gain + 128, 0, 255) for sample in frames)

    if sample_width == 2:
        out = bytearray(len(frames))
        for idx in range(0, len(frames), 2):
            sample = int.from_bytes(frames[idx:idx + 2], "little", signed=True)
            value = clamp_int(sample * gain, -32768, 32767)
            out[idx:idx + 2] = value.to_bytes(2, "little", signed=True)
        return bytes(out)

    if sample_width == 3:
        out = bytearray()
        for idx in range(0, len(frames), 3):
            chunk = frames[idx:idx + 3]
            if len(chunk) < 3:
                break
            sign = b"\xff" if chunk[2] & 0x80 else b"\x00"
            sample = int.from_bytes(chunk + sign, "little", signed=True)
            value = clamp_int(sample * gain, -8388608, 8388607)
            out.extend(value.to_bytes(4, "little", signed=True)[:3])
        return bytes(out)

    if sample_width == 4:
        out = bytearray(len(frames))
        for idx in range(0, len(frames), 4):
            sample = int.from_bytes(frames[idx:idx + 4], "little", signed=True)
            value = clamp_int(sample * gain, -2147483648, 2147483647)
            out[idx:idx + 4] = value.to_bytes(4, "little", signed=True)
        return bytes(out)

    raise ValueError(f"unsupported sample width: {sample_width}")


def amplify_wav_bytes(wav_data: bytes, gain: float) -> bytes:
    if gain == 1.0:
        return wav_data

    src = io.BytesIO(wav_data)
    dst = io.BytesIO()
    try:
        with wave.open(src, "rb") as reader:
            params = reader.getparams()
            frames = reader.readframes(reader.getnframes())
            amplified = amplify_pcm_frames(frames, reader.getsampwidth(), gain)

        with wave.open(dst, "wb") as writer:
            writer.setparams(params)
            writer.writeframes(amplified)
        return dst.getvalue()
    except Exception as exc:
        print(f"[voice] input gain failed, sending raw audio: {exc}", flush=True)
        return wav_data


def make_wav_bytes(frames: bytes, channels: int, sample_width: int, rate: int, gain: float) -> bytes:
    amplified = amplify_pcm_frames(frames, sample_width, gain)
    dst = io.BytesIO()
    with wave.open(dst, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(rate)
        writer.writeframes(amplified)
    return dst.getvalue()


def guess_upload_ext(content_type: str) -> str:
    lowered = content_type.lower()
    if "wav" in lowered:
        return ".wav"
    if "webm" in lowered:
        return ".webm"
    if "ogg" in lowered or "opus" in lowered:
        return ".ogg"
    if "aac" in lowered:
        return ".aac"
    if "m4a" in lowered:
        return ".m4a"
    return ".mp3"


def extract_multipart_file(body: bytes, content_type: str) -> bytes:
    marker = "boundary="
    if marker not in content_type:
        return body

    boundary = content_type.split(marker, 1)[1].split(";", 1)[0].strip().strip('"')
    if not boundary:
        return body

    delimiter = b"--" + boundary.encode("utf-8")
    for part in body.split(delimiter):
        if b"\r\n\r\n" not in part:
            continue
        headers, data = part.split(b"\r\n\r\n", 1)
        if b'name="file"' not in headers and b"filename=" not in headers:
            continue
        data = data.rstrip(b"\r\n")
        if data.endswith(b"--"):
            data = data[:-2].rstrip(b"\r\n")
        return data

    return body


def make_handler(state: VoiceState):
    class VoiceHandler(BaseHTTPRequestHandler):
        server_version = "RKVoiceIntercom/1.0"

        def log_message(self, fmt: str, *args) -> None:
            print(f"[voice] {self.address_string()} - {fmt % args}", flush=True)

        def send_common_headers(
            self,
            status: int,
            content_type: str,
            length: int | None = None,
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Expose-Headers", "X-Next-Cursor, X-Audio-Reset")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", content_type)
            if length is not None:
                self.send_header("Content-Length", str(length))
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def send_json(self, status: int, payload: dict) -> None:
            data = json_bytes(payload)
            self.send_common_headers(status, "application/json; charset=utf-8", len(data))
            self.wfile.write(data)

        def do_OPTIONS(self) -> None:
            self.send_common_headers(204, "text/plain", 0)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_json(200, {
                    "ok": True,
                    "time": time.time(),
                    "input_gain": state.args.input_gain,
                    "output_gain": state.args.output_gain,
                })
                return

            if parsed.path != "/mic-chunk":
                self.send_json(404, {"ok": False, "error": "not found"})
                return

            query = parse_qs(parsed.query)
            try:
                duration = float(query.get("duration", [state.args.chunk_seconds])[0])
            except ValueError:
                duration = state.args.chunk_seconds
            duration = max(0.2, min(MAX_CHUNK_SECONDS, duration))
            cursor_text = query.get("cursor", [""])[0]
            cursor = None
            try:
                if cursor_text:
                    cursor = int(cursor_text)
            except ValueError:
                cursor = None

            if is_audio_owner_active(state.args.audio_control_file, "ai_record"):
                chunk_bytes = int(duration * state.capture.bytes_per_second)
                chunk_bytes -= chunk_bytes % state.capture.frame_size
                chunk_bytes = max(state.capture.frame_size, chunk_bytes)
                audio = make_wav_bytes(
                    b"\x00" * chunk_bytes,
                    channels=state.args.capture_channels,
                    sample_width=state.args.capture_sample_width,
                    rate=state.args.capture_rate,
                    gain=1.0,
                )
                self.send_common_headers(
                    200,
                    "audio/wav",
                    len(audio),
                    {
                        "X-Next-Cursor": str(cursor if cursor is not None else state.capture.write_seq),
                        "X-Audio-Reset": "0",
                        "X-Audio-Paused": "1",
                    },
                )
                self.wfile.write(audio)
                return

            state.capture.start()
            frames, next_cursor, reset, error = state.capture.read_chunk(
                duration=duration,
                cursor=cursor,
                timeout=duration + state.args.capture_timeout_extra,
            )
            if not frames:
                self.send_json(500, {"ok": False, "error": error or "no audio captured"})
                return

            audio = make_wav_bytes(
                frames,
                channels=state.args.capture_channels,
                sample_width=state.args.capture_sample_width,
                rate=state.args.capture_rate,
                gain=state.args.input_gain,
            )
            self.send_common_headers(
                200,
                "audio/wav",
                len(audio),
                {
                    "X-Next-Cursor": str(next_cursor),
                    "X-Audio-Reset": "1" if reset else "0",
                },
            )
            self.wfile.write(audio)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/speaker-stop":
                state.stop_playback()
                clear_audio_owner(state.args.audio_control_file, "phone_speaker")
                self.send_json(200, {"ok": True})
                return

            if parsed.path != "/speaker-chunk":
                self.send_json(404, {"ok": False, "error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json(400, {"ok": False, "error": "bad content length"})
                return

            if length <= 0 or length > MAX_UPLOAD_BYTES:
                self.send_json(413, {"ok": False, "error": "upload too large or empty"})
                return

            body = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            audio_data = extract_multipart_file(body, content_type)
            if not audio_data:
                self.send_json(400, {"ok": False, "error": "empty audio"})
                return

            mark_audio_owner(
                state.args.audio_control_file,
                "phone_speaker",
                "phone audio uploaded",
                state.args.phone_audio_hold_sec,
            )

            ext = guess_upload_ext(content_type)
            upload_dir = Path(state.args.upload_dir)
            upload_dir.mkdir(parents=True, exist_ok=True)
            path = upload_dir / f"phone_{int(time.time() * 1000)}_{uuid.uuid4().hex}{ext}"
            path.write_bytes(audio_data)

            if not state.enqueue_play(path):
                self.send_json(503, {"ok": False, "error": "play queue full"})
                return

            self.send_json(200, {"ok": True, "bytes": len(audio_data)})

    return VoiceHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3588 phone-style voice intercom server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--capture-device", default=DEFAULT_CAPTURE_DEVICE)
    parser.add_argument("--play-device", default=DEFAULT_PLAY_DEVICE)
    parser.add_argument("--capture-format", default="cd")
    parser.add_argument("--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS)
    parser.add_argument("--capture-rate", type=int, default=44100)
    parser.add_argument("--capture-channels", type=int, default=2)
    parser.add_argument("--capture-sample-width", type=int, default=2,
                        help="Bytes per audio sample. cd/S16_LE should use 2.")
    parser.add_argument("--capture-buffer-sec", type=float, default=12.0)
    parser.add_argument("--capture-block-ms", type=float, default=100.0)
    parser.add_argument("--capture-timeout-extra", type=float, default=3.0)
    parser.add_argument("--play-timeout", type=float, default=10.0)
    parser.add_argument("--play-queue-size", type=int, default=3)
    parser.add_argument("--audio-gain", type=float, default=None,
                        help="Set both input and output gain. Defaults to 8x when input/output gains are not set.")
    parser.add_argument("--input-gain", type=float, default=DEFAULT_AUDIO_GAIN,
                        help="Gain for RK3588 microphone audio sent to the app.")
    parser.add_argument("--output-gain", type=float, default=DEFAULT_AUDIO_GAIN,
                        help="Gain for phone/app audio played on RK3588.")
    parser.add_argument("--audio-control-file", default=DEFAULT_AUDIO_CONTROL_FILE,
                        help="Shared audio priority state file used by AI assistant and voice intercom.")
    parser.add_argument("--phone-audio-hold-sec", type=float, default=3.0,
                        help="Seconds to reserve speaker priority immediately after a phone audio upload.")
    parser.add_argument("--upload-dir", default=str(Path(tempfile.gettempdir()) / "rk_voice_intercom"))
    parser.add_argument("--arecord-bin", default="arecord")
    parser.add_argument("--gst-launch-bin", default="gst-launch-1.0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.chunk_seconds = max(1, min(MAX_CHUNK_SECONDS, int(args.chunk_seconds)))
    if args.audio_gain is not None:
        args.input_gain = args.audio_gain
        args.output_gain = args.audio_gain
    args.input_gain = max(0.0, float(args.input_gain))
    args.output_gain = max(0.0, float(args.output_gain))
    args.phone_audio_hold_sec = max(0.1, float(args.phone_audio_hold_sec))

    state = VoiceState(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(
        f"[voice] listening on http://{args.host}:{args.port}, "
        f"mic={args.capture_device}, speaker={args.play_device}, "
        f"input_gain={args.input_gain}, output_gain={args.output_gain}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
