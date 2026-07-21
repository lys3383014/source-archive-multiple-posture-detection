from __future__ import annotations

import argparse
import base64
import json
import math
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audio_priority import DEFAULT_AUDIO_CONTROL_FILE

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    raise SystemExit("Missing RKNN runtime. Install rknn-toolkit-lite2 on RK3588.") from exc

from get_datamatrix import stream_datamatrix


BASE_DIR = Path(__file__).resolve().parent

FUSED_CLASSES = ["stand", "lying", "sit", "fall", "walk", "bend"]
STATIC_DISPLAY_CLASSES = {"stand", "lying", "sit"}
STATIC_DISPLAY_LABEL = "static"
CLASS_CN = {
    "stand": "stand",
    "lying": "lying",
    "sit": "sit",
    "static": "static",
    "fall": "fall",
    "walk": "walk",
    "bend": "bend",
}
FUSED_COLORS = {
    "stand": (0, 200, 0),
    "lying": (120, 90, 240),
    "sit": (0, 165, 255),
    "fall": (0, 0, 220),
    "walk": (200, 0, 0),
    "bend": (0, 180, 180),
}

VISION_CLASSES = ["stand", "walk", "bendover", "lying", "sit"]
VISION_TO_FUSED = {"stand": "stand", "walk": "walk", "bendover": "bend", "lying": "lying", "sit": "sit"}
VISION_MISSING_LABEL = "__missing__"
RADAR_IDX_TO_CLASS = {0: "static", 1: "walk", 2: "bend", 3: "fall"}

YOLO_IMG_SIZE = (640, 640)
YOLO_OBJ_THRESH = 0.65
YOLO_NMS_THRESH = 0.45
YOLO_PERSON_CLS = 0

CLS_IMG_SIZE = 224
CLS_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
CLS_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SPEED_WINDOW = 5
SPEED_THRESHOLD = 0.008
SPEED_MOTOR_COMP_K = 0.32
IOU_MATCH_THRESHOLD = 0.30
VISION_FILTER_WINDOW = 5
VISION_LYING_STABLE_FRAMES = 5
VISION_LYING_FALL_HOLD_SEC = 2.0
VISION_TEMPERATURE = 1.6
RADAR_TEMPERATURE = 1.0
DYNAMIC_RADAR_WEIGHT = 0.4
STATIC_RADAR_WEIGHT = 0.3
STATUS_POST_SEC = 0.5
STATUS_IMAGE_SEC = 10.0
STATUS_IMAGE_WIDTH = 480
STATUS_JPEG_QUALITY = 65
REMINDER_POPUP_SEC = 5.0
REMINDER_DUE_GRACE_SEC = 60.0
ALARM_URL = ""
APP_STATUS_PORT = 8889
MOTOR_PORT = "/dev/ttyS9"
MOTOR_BAUDRATE = 115200
MOTOR_ANGLE_MIN = -90.0
MOTOR_ANGLE_MAX = 90.0
MOTOR_LOCATION_SETTLE_SEC = 2.0
MOTOR_INIT_ZERO_DELAY_SEC = 0.02
TRACK_KP = 200.0
TRACK_KI = 20.0
TRACK_KD = 0.8
TRACK_MAX_SPEED = 60.0
TRACK_DEADBAND_PX = 10.0
TRACK_SEND_INTERVAL = 0.08
WINDOW_NAME = "Fusion Posture Recognition"
AI_BUTTON_RECT = (1110, 520, 150, 72)
AI_TAKEOVER_CANCEL_RECT = (1110, 602, 150, 42)
BOARD_IP_REFRESH_SEC = 10.0
NETWORK_ERROR_LOG_SEC = 5.0
_PIL_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}
_NETWORK_ERROR_LOG_TIMES: dict[tuple[str, str, str], float] = {}
_NETWORK_ERROR_LOG_LOCK = threading.Lock()

RADAR_IP = "192.168.1.100"
SCAN_START_PS = 5000
SCAN_STOP_PS = 50000
INTERVAL_US = 20000
WINDOW_SCANS = 200
DISTANCE_MIN_M = 1.0
DISTANCE_MAX_M = 7.0
CABLE_COMPENSATION_M = 0.20
DEFAULT_MTI_ALPHA = 0.95
RADAR_MODEL_SIZE = 224
RADAR_IMAGE_MEAN = 0.5
RADAR_IMAGE_STD = 0.5

LPF_B = np.asarray(
    [0.09311643508372727, 0.27934930525118185, 0.27934930525118185, 0.09311643508372727],
    dtype=np.float32,
)
LPF_A = np.asarray(
    [1.0, -0.6320445529936096, 0.43946018469982556, -0.06248415103639782],
    dtype=np.float32,
)
VIRIDIS_GRAY_LUT = np.asarray(
    [
        30, 31, 32, 34, 34, 36, 37, 38, 39, 40, 41, 42, 43, 45, 45, 46,
        47, 48, 49, 51, 51, 52, 53, 54, 55, 55, 56, 57, 58, 58, 60, 60,
        61, 61, 63, 63, 64, 64, 66, 66, 67, 67, 69, 69, 70, 70, 71, 71,
        72, 73, 73, 74, 74, 75, 76, 76, 77, 77, 78, 78, 79, 79, 80, 81,
        81, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 86, 87, 87, 88, 88,
        89, 89, 89, 90, 91, 91, 91, 92, 92, 93, 93, 94, 94, 95, 95, 96,
        96, 97, 97, 97, 98, 99, 99, 99, 100, 100, 100, 101, 101, 102, 102,
        103, 103, 104, 104, 104, 105, 105, 106, 106, 107, 107, 107, 108,
        108, 109, 109, 109, 110, 111, 111, 111, 112, 113, 113, 113, 114,
        115, 115, 115, 116, 116, 117, 117, 118, 118, 119, 120, 120, 121,
        122, 122, 123, 123, 124, 124, 125, 126, 127, 128, 129, 129, 130,
        131, 131, 132, 133, 134, 135, 136, 137, 138, 139, 139, 140, 141,
        142, 143, 144, 145, 145, 146, 148, 149, 149, 150, 151, 152, 153,
        154, 155, 156, 157, 158, 159, 160, 161, 162, 162, 164, 164, 165,
        166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 176, 177,
        178, 179, 180, 181, 181, 183, 183, 184, 185, 186, 187, 188, 189,
        189, 190, 191, 192, 193, 194, 195, 195, 196, 197, 198, 198, 200,
        201, 201, 203, 203, 204, 206, 206, 207, 209, 210, 211, 212, 213,
        214, 215,
    ],
    dtype=np.uint8,
)


@dataclass
class FusionState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    vision_prob: np.ndarray | None = None
    vision_label: str | None = None
    vision_raw_label: str | None = None
    vision_time: float = 0.0
    vision_frame: np.ndarray | None = None
    radar_prob: np.ndarray | None = None
    radar_label: str | None = None
    radar_time: float = 0.0
    radar_image: np.ndarray | None = None
    mode: str = "vision"
    detection_enabled: bool = True
    tracking_enabled: bool = False
    debug_vision_label: str | None = None
    debug_radar_label: str | None = None
    control_request_id: int = 0
    tracking_error_px: float = 0.0
    tracking_motor_speed: float = 0.0
    tracking_target_x: float | None = None
    motor_angle: float | None = None
    motor_command: str = "absolute"
    motor_angle_request_id: int = 0
    motor_zero_request_id: int = 0
    motor_relative_angle: float | None = None
    motor_relative_request_id: int = 0
    alarm_ack_timestamp: int = 0
    reminders: list[dict] = field(default_factory=list)
    reminder_request_id: int = 0
    reminder_notification: dict | None = None
    system_command_request_id: int = 0
    board_message_request_id: int = 0
    board_message_notification: dict | None = None
    vision_error: str | None = None
    radar_error: str | None = None
    control_error: str | None = None
    tracking_error: str | None = None


class ReminderManager:
    def __init__(self, path: Path, due_grace_sec: float = REMINDER_DUE_GRACE_SEC):
        self.path = Path(path)
        self.due_grace_sec = max(1.0, float(due_grace_sec))
        self.lock = threading.Lock()
        self.reminders: list[dict] = []
        self.load()

    def load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.reminders = [self.normalize(item) for item in data if isinstance(item, dict)]
        except Exception as exc:
            print(f"[reminder] load failed: {exc}", flush=True)
            self.reminders = []

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.reminders, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"[reminder] save failed: {exc}", flush=True)

    def snapshot(self) -> list[dict]:
        with self.lock:
            now = datetime.now()
            return [self.with_next_text(item, now) for item in self.reminders]

    def with_next_text(self, item: dict, now: datetime) -> dict:
        out = dict(item)
        out["nextText"] = self.next_text(item, now)
        return out

    def normalize(self, raw: dict) -> dict:
        now = datetime.now()
        label = str(raw.get("label") or raw.get("title") or "提醒").strip() or "提醒"
        repeat = str(raw.get("repeat") or raw.get("type") or "once").strip().lower()
        weekdays = self.parse_weekdays(raw.get("weekdays"))
        if repeat not in ("once", "weekly"):
            repeat = "weekly" if weekdays else "once"

        date_text = str(raw.get("date") or "").strip()
        time_text = str(raw.get("time") or "").strip()
        scheduled_at = str(raw.get("datetime") or raw.get("scheduledAt") or raw.get("remindAt") or "").strip()
        if scheduled_at:
            parsed = self.parse_datetime(scheduled_at)
            if parsed is not None:
                date_text = parsed.strftime("%Y-%m-%d")
                time_text = parsed.strftime("%H:%M")

        if not time_text:
            time_text = now.strftime("%H:%M")
        time_text = self.normalize_time(time_text) or now.strftime("%H:%M")

        if repeat == "once":
            if not date_text:
                hour, minute = [int(part) for part in time_text.split(":", 1)]
                candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if candidate <= now:
                    candidate += timedelta(days=1)
                date_text = candidate.strftime("%Y-%m-%d")
            weekdays = []
        else:
            date_text = ""
            if not weekdays:
                weekdays = [now.isoweekday()]

        return {
            "id": str(raw.get("id") or uuid4().hex[:8]),
            "label": label,
            "repeat": repeat,
            "date": date_text,
            "time": time_text,
            "weekdays": weekdays,
            "enabled": bool(raw.get("enabled", True)),
            "done": bool(raw.get("done", False)),
            "lastTriggerKey": str(raw.get("lastTriggerKey") or ""),
            "createdAt": int(raw.get("createdAt") or int(time.time() * 1000)),
            "updatedAt": int(time.time() * 1000),
        }

    def parse_weekdays(self, value) -> list[int]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = re_split_weekdays(value)
        elif isinstance(value, (list, tuple, set)):
            parts = list(value)
        else:
            parts = [value]

        result: list[int] = []
        for part in parts:
            try:
                number = int(part)
            except (TypeError, ValueError):
                continue
            if 0 <= number <= 6:
                number = 7 if number == 0 else number
            if 1 <= number <= 7 and number not in result:
                result.append(number)
        result.sort()
        return result

    def normalize_time(self, value: str) -> str | None:
        text = str(value or "").strip()
        match = re_search_time(text)
        if not match:
            return None
        hour = int(match[0])
        minute = int(match[1])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
        return None

    def parse_datetime(self, value: str) -> datetime | None:
        text = str(value or "").strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
        return None

    def find_index(self, reminder_id: str = "", target_label: str = "") -> int | None:
        reminder_id = str(reminder_id or "").strip()
        target_label = str(target_label or "").strip()
        for idx, item in enumerate(self.reminders):
            if reminder_id and str(item.get("id")) == reminder_id:
                return idx
        if target_label:
            for idx, item in enumerate(self.reminders):
                if str(item.get("label") or "") == target_label:
                    return idx
            for idx, item in enumerate(self.reminders):
                if target_label in str(item.get("label") or ""):
                    return idx
        return None

    def apply_command(self, command: dict) -> tuple[bool, str]:
        action = str(command.get("action") or "").strip()
        with self.lock:
            if action == "reminder_add":
                reminder = self.normalize(command.get("reminder") or {})
                self.reminders.append(reminder)
                self.save()
                return True, f"added {reminder['label']}"

            if action == "reminder_update":
                idx = self.find_index(command.get("reminderId"), command.get("targetLabel"))
                if idx is None:
                    return False, "reminder not found"
                original = dict(self.reminders[idx])
                patch = dict(command.get("reminder") or {})
                patch["id"] = original["id"]
                if "label" not in patch:
                    patch["label"] = original.get("label")
                if "repeat" not in patch:
                    patch["repeat"] = original.get("repeat")
                if "date" not in patch:
                    patch["date"] = original.get("date")
                if "time" not in patch:
                    patch["time"] = original.get("time")
                if "weekdays" not in patch:
                    patch["weekdays"] = original.get("weekdays")
                patch["createdAt"] = original.get("createdAt")
                self.reminders[idx] = self.normalize(patch)
                self.save()
                return True, f"updated {self.reminders[idx]['label']}"

            if action == "reminder_delete":
                idx = self.find_index(command.get("reminderId"), command.get("targetLabel"))
                if idx is None:
                    return False, "reminder not found"
                removed = self.reminders.pop(idx)
                self.save()
                return True, f"deleted {removed.get('label')}"

        return False, f"unsupported reminder action: {action}"

    def check_due(self, now: datetime | None = None) -> list[dict]:
        now = now or datetime.now()
        due: list[dict] = []
        changed = False
        with self.lock:
            for item in self.reminders:
                event = self.due_event(item, now)
                if event is not None:
                    due.append(event)
                    changed = True
            if changed:
                self.save()
        return due

    def due_event(self, item: dict, now: datetime) -> dict | None:
        if not item.get("enabled", True):
            return None
        label = str(item.get("label") or "提醒")
        repeat = str(item.get("repeat") or "once")
        time_text = self.normalize_time(str(item.get("time") or "")) or "00:00"
        hour, minute = [int(part) for part in time_text.split(":", 1)]

        if repeat == "weekly":
            weekdays = self.parse_weekdays(item.get("weekdays"))
            if now.isoweekday() not in weekdays:
                return None
            due_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            diff = (now - due_time).total_seconds()
            key = f"{now.strftime('%Y-%m-%d')} {time_text}"
            if 0 <= diff <= self.due_grace_sec and item.get("lastTriggerKey") != key:
                item["lastTriggerKey"] = key
                item["updatedAt"] = int(time.time() * 1000)
                return self.make_event(item, label, due_time)
            return None

        if item.get("done", False):
            return None
        date_text = str(item.get("date") or now.strftime("%Y-%m-%d"))
        due_time = self.parse_datetime(f"{date_text} {time_text}")
        if due_time is None:
            return None
        diff = (now - due_time).total_seconds()
        if 0 <= diff <= self.due_grace_sec:
            item["done"] = True
            item["enabled"] = False
            item["lastTriggerKey"] = f"{date_text} {time_text}"
            item["updatedAt"] = int(time.time() * 1000)
            return self.make_event(item, label, due_time)
        return None

    def make_event(self, item: dict, label: str, due_time: datetime) -> dict:
        return {
            "id": str(item.get("id") or ""),
            "label": label,
            "message": f"提醒：{label}",
            "scheduledText": self.schedule_text(item),
            "scheduledAt": due_time.strftime("%Y-%m-%d %H:%M"),
            "triggeredAt": int(time.time() * 1000),
        }

    def schedule_text(self, item: dict) -> str:
        repeat = str(item.get("repeat") or "once")
        if repeat == "weekly":
            names = ["一", "二", "三", "四", "五", "六", "日"]
            days = [names[idx - 1] for idx in self.parse_weekdays(item.get("weekdays")) if 1 <= idx <= 7]
            return f"每周{''.join(days)} {item.get('time')}"
        return f"{item.get('date')} {item.get('time')}"

    def next_text(self, item: dict, now: datetime) -> str:
        if not item.get("enabled", True):
            return "已关闭"
        if item.get("done", False):
            return "已完成"
        return self.schedule_text(item)


def re_search_time(text: str) -> tuple[str, str] | None:
    import re

    match = re.search(r"(\d{1,2})[:：点时](\d{1,2})?", text)
    if match:
        return match.group(1), match.group(2) or "0"
    match = re.search(r"\b(\d{1,2})\b", text)
    if match:
        return match.group(1), "0"
    return None


def re_split_weekdays(value: str) -> list[str]:
    import re

    mapping = {
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "日": "7",
        "天": "7",
    }
    out: list[str] = []
    for ch in str(value):
        if ch in mapping:
            out.append(mapping[ch])
    out.extend(re.findall(r"\d", str(value)))
    return out


class SpeedTracker:
    def __init__(self, track_id: int, window_size: int = SPEED_WINDOW):
        self.track_id = track_id
        self.history = deque(maxlen=window_size)
        self.last_bbox = None
        self.frames_since_update = 0

    def update(self, bbox: np.ndarray, frame_idx: int, motor_speed: float = 0.0) -> None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        h = max(y2 - y1, 1.0)
        self.history.append({"frame": frame_idx, "cx": cx, "h": h, "motor_speed": float(motor_speed)})
        self.last_bbox = bbox.copy()
        self.frames_since_update = 0

    def mark_missed(self) -> None:
        self.frames_since_update += 1

    def is_stale(self, max_missed: int = 30) -> bool:
        return self.frames_since_update > max_missed

    def compute_speed(self, motor_comp_k: float = 0.0) -> float:
        if len(self.history) < 2:
            return 0.0
        frames = np.array([d["frame"] for d in self.history], dtype=np.float64)
        cxs = np.array([d["cx"] for d in self.history], dtype=np.float64)
        hs = np.array([d["h"] for d in self.history], dtype=np.float64)
        motor_speeds = np.array([d.get("motor_speed", 0.0) for d in self.history], dtype=np.float64)
        t = frames - frames[0]
        h_mean = float(hs.mean())
        if h_mean < 1.0:
            return 0.0
        k_h = regression_slope(t, cxs)
        k_v = regression_slope(t, hs)
        norm_h = float(k_h) / h_mean
        if motor_comp_k != 0.0:
            norm_h -= float(motor_comp_k) * float(motor_speeds.mean()) / h_mean
        norm_v = float(k_v) / h_mean
        return math.sqrt(norm_h ** 2 + norm_v ** 2)


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, output_limit: float, integral_limit: float = 2.0):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.output_limit = abs(float(output_limit))
        self.integral_limit = abs(float(integral_limit))
        self.integral = 0.0
        self.last_error: float | None = None
        self.last_time: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.last_error = None
        self.last_time = None

    def update(self, error: float, now: float | None = None) -> float:
        now = time.time() if now is None else now
        if self.last_time is None:
            dt = 0.0
        else:
            dt = max(1e-3, now - self.last_time)

        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        derivative = 0.0 if self.last_error is None or dt <= 0 else (error - self.last_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.last_error = error
        self.last_time = now
        return max(-self.output_limit, min(self.output_limit, output))


class MotorController:
    def __init__(self, args: argparse.Namespace):
        self.port = args.motor_port
        self.baudrate = args.motor_baudrate
        self.max_speed = abs(float(args.track_max_speed))
        self.send_interval = max(0.0, float(args.track_send_interval))
        self.location_settle_sec = max(0.0, float(args.motor_location_settle_sec))
        self.ser = None
        self.serial_speed = None
        self.serial_location = None
        self.io_lock = threading.Lock()
        self.position_lock = threading.Lock()
        self.position_pending: tuple[str, float | None] | None = None
        self.position_thread: threading.Thread | None = None
        self.position_busy = False
        self.last_position_angle: float | None = None
        self.last_sent: int | None = None
        self.last_send_time = 0.0
        self.last_error: str | None = None
        self.closing = False

    def _open(self) -> None:
        if self.ser is not None and getattr(self.ser, "is_open", True):
            return
        import serial_speed

        self.serial_speed = serial_speed
        self.ser = serial_speed.serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial_speed.serial.EIGHTBITS,
            parity=serial_speed.serial.PARITY_NONE,
            stopbits=serial_speed.serial.STOPBITS_ONE,
            timeout=0.02,
        )

    def set_speed(self, speed: float, force: bool = False) -> float:
        if self.closing:
            return 0.0
        if self.max_speed <= 0:
            speed_int = 0
        else:
            speed = max(-self.max_speed, min(self.max_speed, float(speed)))
            speed_int = int(round(speed))

        now = time.time()
        if not force and self.last_sent == speed_int:
            return float(speed_int)

        if speed_int == 0 and self.ser is None and self.last_sent in (None, 0):
            self.last_sent = 0
            self.last_send_time = now
            self.last_error = None
            return 0.0

        try:
            with self.io_lock:
                self._open()
                cmd = self.serial_speed.build_speed_cmd(speed_int)
                written = self.ser.write(cmd)
                self.serial_speed.log_serial_send(
                    self.serial_speed.speed_meaning(speed_int),
                    cmd,
                    getattr(self.ser, "port", self.port),
                    written,
                )
            self.last_sent = speed_int
            self.last_send_time = now
            self.last_error = None
            return float(speed_int)
        except Exception as exc:
            self.last_error = str(exc)
            self.close(send_stop=False, mark_closing=False)
            return 0.0

    def stop(self) -> float:
        return self.set_speed(0.0, force=True)

    def request_position(self, angle: float) -> None:
        self._request_manual_command("absolute", float(angle))

    def request_current_zero(self) -> None:
        self._request_manual_command("zero", None)

    def request_relative_position(self, angle: float) -> None:
        self._request_manual_command("relative", float(angle))

    def _request_manual_command(self, command: str, angle: float | None) -> None:
        if self.closing:
            return
        with self.position_lock:
            self.position_pending = (command, angle)
            if self.position_thread is not None and self.position_thread.is_alive():
                return
            self.position_thread = threading.Thread(target=self._position_worker, daemon=True)
            self.position_thread.start()

    def is_position_busy(self) -> bool:
        with self.position_lock:
            return self.position_busy or self.position_pending is not None

    def _position_worker(self) -> None:
        while not self.closing:
            with self.position_lock:
                pending = self.position_pending
                self.position_pending = None
                if pending is None:
                    self.position_busy = False
                    return
                self.position_busy = True
                command, angle = pending

            try:
                import serial_location

                self.serial_location = serial_location
                with self.io_lock:
                    self._open()
                    if command == "zero":
                        serial_location.set_current_zero(self.ser)
                    elif command == "relative":
                        serial_location.move_relative_angle(self.ser, float(angle or 0.0))
                    else:
                        serial_location.move_to_angle(self.ser, float(angle or 0.0), self.location_settle_sec)
                if command == "zero":
                    self.last_position_angle = 0.0
                    print("[motor] current position set as zero by app", flush=True)
                elif command == "relative":
                    print(f"[motor] moved relative angle={float(angle or 0.0):+.2f}", flush=True)
                else:
                    self.last_position_angle = float(angle or 0.0)
                    print(f"[motor] moved to angle={float(angle or 0.0):.2f}", flush=True)
                self.last_error = None
            except Exception as exc:
                self.last_error = str(exc)
                print(f"[motor] manual motor command failed: {exc}", flush=True)
                self.close(send_stop=False, mark_closing=False)

    def close(self, send_stop: bool = True, mark_closing: bool = True) -> None:
        if send_stop:
            try:
                self.stop()
            except Exception:
                pass
        if mark_closing:
            self.closing = True
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None


def regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    den = float(((x - mx) ** 2).sum())
    if den <= 1e-8:
        return 0.0
    return float(((x - mx) * (y - my)).sum()) / den


def class_index(name: str) -> int:
    return FUSED_CLASSES.index(name)


def normalize_debug_label(value: object) -> str | None:
    label = str(value or "").strip().lower()
    return label if label in FUSED_CLASSES else None


def clear_detection_results_locked(state: FusionState) -> None:
    state.vision_prob = None
    state.vision_label = None
    state.vision_raw_label = None
    state.vision_time = 0.0
    state.radar_prob = None
    state.radar_label = None
    state.radar_time = 0.0


def display_label_for_result(
    fused_label: str | None,
    vision_label: str | None,
    vision_w: float,
    mode: str,
) -> str | None:
    vision_available = mode != "radar" and vision_label is not None and float(vision_w) > 0.0
    if fused_label in STATIC_DISPLAY_CLASSES and not vision_available:
        return STATIC_DISPLAY_LABEL
    return fused_label


def normalize_vector(prob: np.ndarray, fallback_len: int | None = None) -> np.ndarray:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    total = float(prob.sum())
    if total <= 1e-8:
        length = int(fallback_len or prob.size or 1)
        return np.ones(length, dtype=np.float32) / max(length, 1)
    return prob / total


def normalize_prob(prob: np.ndarray) -> np.ndarray:
    return normalize_vector(prob, len(FUSED_CLASSES))


def soften_prob(prob: np.ndarray, temperature: float) -> np.ndarray:
    prob = normalize_vector(prob)
    if temperature <= 1.0:
        return prob
    softened = np.power(np.clip(prob, 1e-8, 1.0), 1.0 / temperature)
    return normalize_vector(softened)


def one_hot(label: str, conf: float) -> np.ndarray:
    prob = np.zeros(len(FUSED_CLASSES), dtype=np.float32)
    prob[class_index(label)] = float(conf)
    rest = max(0.0, 1.0 - float(conf)) / (len(FUSED_CLASSES) - 1)
    for i in range(len(FUSED_CLASSES)):
        if i != class_index(label):
            prob[i] = rest
    return prob


def random_debug_prob(label: str, low: float = 0.95, high: float = 0.99) -> np.ndarray:
    selected_idx = class_index(label)
    selected_conf = float(np.random.uniform(low, high))
    rest_total = max(0.0, 1.0 - selected_conf)
    rest_weights = np.random.random(len(FUSED_CLASSES) - 1).astype(np.float32)
    weight_sum = float(rest_weights.sum())
    if weight_sum <= 1e-8:
        rest_weights = np.ones(len(FUSED_CLASSES) - 1, dtype=np.float32)
        weight_sum = float(rest_weights.sum())

    prob = np.zeros(len(FUSED_CLASSES), dtype=np.float32)
    prob[selected_idx] = selected_conf
    rest_index = 0
    for i in range(len(FUSED_CLASSES)):
        if i == selected_idx:
            continue
        prob[i] = rest_total * float(rest_weights[rest_index]) / weight_sum
        rest_index += 1
    return prob


def random_debug_radar_static_prob(low: float = 0.95, high: float = 0.99) -> np.ndarray:
    static_conf = float(np.random.uniform(low, high))
    rest_total = max(0.0, 1.0 - static_conf)
    rest_labels = ("fall", "walk", "bend")
    rest_weights = np.random.random(len(rest_labels)).astype(np.float32)
    weight_sum = float(rest_weights.sum())
    if weight_sum <= 1e-8:
        rest_weights = np.ones(len(rest_labels), dtype=np.float32)
        weight_sum = float(rest_weights.sum())

    prob = np.zeros(len(FUSED_CLASSES), dtype=np.float32)
    for label in STATIC_DISPLAY_CLASSES:
        prob[class_index(label)] = static_conf
    for idx, label in enumerate(rest_labels):
        prob[class_index(label)] = rest_total * float(rest_weights[idx]) / weight_sum
    return prob


def core_mask_from_arg(core: str):
    masks = {
        "0": getattr(RKNNLite, "NPU_CORE_0", None),
        "1": getattr(RKNNLite, "NPU_CORE_1", None),
        "2": getattr(RKNNLite, "NPU_CORE_2", None),
        "all": getattr(RKNNLite, "NPU_CORE_0_1_2", None),
    }
    return masks.get(str(core).lower())


def load_rknn_model(model_path: Path, npu_core: str):
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {model_path}")
    core_mask = core_mask_from_arg(npu_core)
    ret = rknn.init_runtime(core_mask=core_mask) if core_mask is not None else rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {model_path}")
    return rknn


def softmax_axis(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_1d(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32).reshape(-1)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def match_trackers(
    bboxes: list[np.ndarray], trackers: list[SpeedTracker]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not trackers:
        return [], list(range(len(bboxes))), []

    iou_matrix = np.zeros((len(bboxes), len(trackers)), dtype=np.float32)
    for i, box in enumerate(bboxes):
        for j, tracker in enumerate(trackers):
            if tracker.last_bbox is not None:
                iou_matrix[i, j] = bbox_iou(box, tracker.last_bbox)

    matches = []
    used_box = set()
    used_tracker = set()
    for _ in range(min(len(bboxes), len(trackers))):
        best_val = IOU_MATCH_THRESHOLD
        best_pair = None
        for i in range(len(bboxes)):
            if i in used_box:
                continue
            for j in range(len(trackers)):
                if j in used_tracker:
                    continue
                if iou_matrix[i, j] > best_val:
                    best_val = float(iou_matrix[i, j])
                    best_pair = (i, j)
        if best_pair is None:
            break
        matches.append(best_pair)
        used_box.add(best_pair[0])
        used_tracker.add(best_pair[1])

    unmatched_boxes = [i for i in range(len(bboxes)) if i not in used_box]
    unmatched_trackers = [j for j in range(len(trackers)) if j not in used_tracker]
    return matches, unmatched_boxes, unmatched_trackers


def update_speed(
    valid_boxes: list[np.ndarray],
    trackers: list[SpeedTracker],
    frame_idx: int,
    window_size: int,
    motor_speed: float,
    motor_comp_k: float,
) -> tuple[list[SpeedTracker], dict[int, float]]:
    matches, unmatched_box_idx, unmatched_tracker_idx = match_trackers(valid_boxes, trackers)
    matched_trackers = {}

    for box_idx, tracker_idx in matches:
        trackers[tracker_idx].update(valid_boxes[box_idx], frame_idx, motor_speed)
        matched_trackers[box_idx] = trackers[tracker_idx]
    for tracker_idx in unmatched_tracker_idx:
        trackers[tracker_idx].mark_missed()
    for box_idx in unmatched_box_idx:
        new_id = max((t.track_id for t in trackers), default=-1) + 1
        tracker = SpeedTracker(track_id=new_id, window_size=window_size)
        tracker.update(valid_boxes[box_idx], frame_idx, motor_speed)
        trackers.append(tracker)

    trackers = [t for t in trackers if not t.is_stale()]
    speed_by_box = {box_idx: tracker.compute_speed(motor_comp_k) for box_idx, tracker in matched_trackers.items()}
    for box_idx in unmatched_box_idx:
        for tracker in trackers:
            if tracker.frames_since_update == 0 and np.array_equal(tracker.last_bbox, valid_boxes[box_idx]):
                speed_by_box[box_idx] = tracker.compute_speed(motor_comp_k)
                break
    return trackers, speed_by_box


def letterbox(image: np.ndarray, new_shape: tuple[int, int] = YOLO_IMG_SIZE, color=(0, 0, 0)):
    src_h, src_w = image.shape[:2]
    dst_w, dst_h = new_shape
    scale = min(dst_w / src_w, dst_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    pad_w = dst_w - resized_w
    pad_h = dst_h - resized_h
    left = pad_w // 2
    top = pad_h // 2
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, scale, left, top


def yolo_preprocess(frame_bgr: np.ndarray):
    padded, scale, pad_x, pad_y = letterbox(frame_bgr, YOLO_IMG_SIZE, color=(0, 0, 0))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(np.expand_dims(rgb, axis=0)), scale, pad_x, pad_y


def map_boxes_to_original(boxes: np.ndarray, scale: float, pad_x: int, pad_y: int, shape: tuple[int, int]):
    if boxes is None or len(boxes) == 0:
        return boxes
    h, w = shape
    mapped = boxes.copy().astype(np.float32)
    mapped[:, [0, 2]] = (mapped[:, [0, 2]] - pad_x) / scale
    mapped[:, [1, 3]] = (mapped[:, [1, 3]] - pad_y) / scale
    mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0, w - 1)
    mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0, h - 1)
    return mapped


def dfl(position: np.ndarray) -> np.ndarray:
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax_axis(y, axis=2)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def yolo_box_process(position: np.ndarray) -> np.ndarray:
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([YOLO_IMG_SIZE[1] // grid_h, YOLO_IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def flatten_output(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[None, :, :, :]
    ch = x.shape[1]
    x = x.transpose(0, 2, 3, 1)
    return x.reshape(-1, ch)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, nms_thresh: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-8)
        order = order[np.where(iou <= nms_thresh)[0] + 1]
    return np.array(keep, dtype=np.int64)


def yolo_post_process(outputs: list[np.ndarray], obj_thresh: float, nms_thresh: float):
    boxes = []
    class_confs = []
    branch_num = 3
    pair_per_branch = len(outputs) // branch_num
    for i in range(branch_num):
        boxes.append(yolo_box_process(np.asarray(outputs[pair_per_branch * i], dtype=np.float32)))
        class_confs.append(np.asarray(outputs[pair_per_branch * i + 1], dtype=np.float32))

    boxes = np.concatenate([flatten_output(x) for x in boxes], axis=0)
    class_confs = np.concatenate([flatten_output(x) for x in class_confs], axis=0)
    person_scores = class_confs[:, YOLO_PERSON_CLS]
    keep = np.where(person_scores >= obj_thresh)[0]
    if keep.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
    boxes = boxes[keep]
    scores = person_scores[keep]
    nms_keep = nms_boxes(boxes, scores, nms_thresh)
    return boxes[nms_keep], scores[nms_keep]


def detect_persons(yolo_rknn, frame_bgr: np.ndarray, obj_thresh: float, nms_thresh: float):
    yolo_input, scale, pad_x, pad_y = yolo_preprocess(frame_bgr)
    outputs = yolo_rknn.inference(inputs=[yolo_input], data_format="nhwc")
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("YOLO RKNN inference returned no output")
    boxes, scores = yolo_post_process(outputs, obj_thresh, nms_thresh)
    boxes = map_boxes_to_original(boxes, scale, pad_x, pad_y, frame_bgr.shape[:2])
    return boxes, scores


def crop_person(frame_bgr: np.ndarray, box: np.ndarray, padding: float = 0.15) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(np.float32)
    dw = (x2 - x1) * padding
    dh = (y2 - y1) * padding
    x1 = max(0, int(x1 - dw))
    y1 = max(0, int(y1 - dh))
    x2 = min(w, int(x2 + dw))
    y2 = min(h, int(y2 + dh))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def posture_preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = crop_rgb.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    padded[y0 : y0 + h, x0 : x0 + w] = crop_rgb
    resized = cv2.resize(padded, (CLS_IMG_SIZE, CLS_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - CLS_NORM_MEAN) / CLS_NORM_STD
    return np.ascontiguousarray(np.expand_dims(tensor, axis=0), dtype=np.float32)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def vision_prob_from_crop(classifier_rknn, crop_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    inp = posture_preprocess(crop_bgr)
    outputs = classifier_rknn.inference(inputs=[inp], data_type="float32", data_format="nhwc")
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("Posture RKNN inference returned no output")
    logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    evidence = softplus(logits)
    alpha = evidence + 1.0
    total = float(alpha.sum())
    vision_prob = alpha / max(total, 1e-8)
    uncertainty = len(VISION_CLASSES) / max(total, 1e-8)
    return normalize_vector(vision_prob, len(VISION_CLASSES)), float(uncertainty)


def apply_speed_override(prob: np.ndarray, speed: float, threshold: float) -> tuple[np.ndarray, str]:
    prob = normalize_vector(prob.copy(), len(VISION_CLASSES))
    top = VISION_CLASSES[int(np.argmax(prob))]
    if top not in ("stand", "walk"):
        return prob, top
    speed_label = "walk" if speed > threshold else "stand"
    stand_idx = VISION_CLASSES.index("stand")
    walk_idx = VISION_CLASSES.index("walk")
    sw_mass = float(prob[stand_idx] + prob[walk_idx])
    prob[stand_idx] = 0.0
    prob[walk_idx] = 0.0
    prob[VISION_CLASSES.index(speed_label)] = sw_mass
    return normalize_vector(prob, len(VISION_CLASSES)), speed_label


def vision_prob_to_fused(vision_prob: np.ndarray, label: str) -> np.ndarray:
    vision_prob = normalize_vector(vision_prob, len(VISION_CLASSES))
    fused = np.zeros(len(FUSED_CLASSES), dtype=np.float32)
    for i, cls in enumerate(VISION_CLASSES):
        if cls == "lying":
            target = "fall" if label == "fall" else "lying"
        else:
            target = VISION_TO_FUSED[cls]
        fused[class_index(target)] += float(vision_prob[i])
    return fused


def choose_recent_mode(entries: deque[tuple[str, np.ndarray]]) -> str:
    labels = [label for label, _ in entries]
    counts = Counter(labels)
    best_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == best_count}
    for label in reversed(labels):
        if label in tied:
            return label
    return labels[-1]


def filtered_vision(entries: deque[tuple[str, np.ndarray]]) -> tuple[str, np.ndarray]:
    label = choose_recent_mode(entries)
    prob = normalize_prob(np.mean([p for _, p in entries], axis=0))
    idx = class_index(label)
    top_idx = int(np.argmax(prob))
    if idx != top_idx:
        prob[idx] = float(prob[top_idx]) + 1e-4
        prob = normalize_prob(prob)
    return label, prob


class VisionLyingFallState:
    def __init__(self, stable_frames: int, hold_sec: float):
        self.stable_frames = max(1, int(stable_frames))
        self.hold_sec = max(0.0, float(hold_sec))
        self.candidate_label: str | None = None
        self.candidate_count = 0
        self.candidate_probs: deque[np.ndarray] = deque(maxlen=self.stable_frames)
        self.stable_raw_label: str | None = None
        self.stable_prob: np.ndarray | None = None
        self.output_label: str | None = None
        self.fall_hold_until = 0.0
        self.missing_count = 0

    def reset(self) -> None:
        self.candidate_label = None
        self.candidate_count = 0
        self.candidate_probs.clear()
        self.stable_raw_label = None
        self.stable_prob = None
        self.output_label = None
        self.fall_hold_until = 0.0
        self.missing_count = 0

    def is_lying_active(self) -> bool:
        return self.stable_raw_label == "lying" or self.output_label in ("lying", "fall")

    def mark_missing_stable(self) -> None:
        self.candidate_label = None
        self.candidate_count = 0
        self.candidate_probs.clear()
        self.stable_raw_label = VISION_MISSING_LABEL
        self.stable_prob = None
        self.output_label = None
        self.fall_hold_until = 0.0
        self.missing_count = self.stable_frames

    def update_missing_pose(self) -> bool:
        self.missing_count += 1
        self.candidate_label = None
        self.candidate_count = 0
        self.candidate_probs.clear()
        if self.missing_count >= self.stable_frames and self.stable_raw_label != VISION_MISSING_LABEL:
            self.mark_missing_stable()
            return True
        return False

    def update(self, raw_label: str, vision_prob: np.ndarray, now: float) -> tuple[str | None, np.ndarray | None]:
        vision_prob = normalize_vector(vision_prob, len(VISION_CLASSES))
        self.missing_count = 0
        if raw_label != self.candidate_label:
            self.candidate_label = raw_label
            self.candidate_count = 0
            self.candidate_probs.clear()
        self.candidate_count += 1
        self.candidate_probs.append(vision_prob.copy())

        if self.candidate_count >= self.stable_frames:
            rolling_prob = normalize_vector(np.mean(list(self.candidate_probs), axis=0), len(VISION_CLASSES))
            if self.stable_raw_label != raw_label:
                previous = self.stable_raw_label
                self.stable_raw_label = raw_label
                self.stable_prob = rolling_prob
                if raw_label == "lying":
                    if previous is not None and previous != "lying":
                        self.output_label = "fall"
                        self.fall_hold_until = now + self.hold_sec
                    else:
                        self.output_label = "lying"
                        self.fall_hold_until = 0.0
                else:
                    self.output_label = VISION_TO_FUSED[raw_label]
                    self.fall_hold_until = 0.0
            elif self.stable_raw_label == raw_label:
                self.stable_prob = rolling_prob

        if self.output_label == "fall" and self.stable_raw_label == "lying" and now >= self.fall_hold_until:
            self.output_label = "lying"
            self.fall_hold_until = 0.0

        if self.output_label is None:
            if self.stable_raw_label == VISION_MISSING_LABEL and raw_label == "lying":
                return None, None
            self.output_label = VISION_TO_FUSED[raw_label]

        output_prob = self.stable_prob if self.stable_prob is not None else vision_prob
        return self.output_label, vision_prob_to_fused(output_prob, self.output_label)


def draw_vision(frame: np.ndarray, box: np.ndarray | None, label: str | None, prob: np.ndarray | None) -> np.ndarray:
    canvas = frame.copy()
    if box is not None and label is not None:
        x1, y1, x2, y2 = map(int, box)
        color = FUSED_COLORS.get(label, (255, 255, 255))
        conf = float(prob[class_index(label)]) if prob is not None else 0.0
        text = f"vision: {label} {conf:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, text, (max(5, x1), max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return canvas


def closest_center_box_index(boxes: list[np.ndarray], frame_width: int) -> int:
    frame_center_x = float(frame_width) / 2.0
    distances = [abs(float((box[0] + box[2]) / 2.0) - frame_center_x) for box in boxes]
    return int(np.argmin(distances))


def update_camera_tracking(
    state: FusionState,
    motor: MotorController,
    pid: PIDController,
    frame: np.ndarray | None,
    target_box: np.ndarray | None,
    tracking_enabled: bool,
    deadband_px: float,
) -> None:
    speed = 0.0
    error_px = 0.0
    target_x = None

    if tracking_enabled and frame is not None and target_box is not None:
        frame_w = max(1.0, float(frame.shape[1]))
        target_x = float((target_box[0] + target_box[2]) / 2.0)
        frame_center_x = frame_w / 2.0
        error_px = target_x - frame_center_x
        if abs(error_px) <= deadband_px:
            pid.reset()
        else:
            normalized_error = error_px / max(frame_center_x, 1.0)
            speed = pid.update(normalized_error)
    else:
        pid.reset()

    position_busy = motor.is_position_busy()
    if position_busy:
        pid.reset()
        speed = 0.0
        actual_speed = 0.0
    else:
        actual_speed = motor.set_speed(speed)
    print(
        "[tracking] "
        f"enabled={tracking_enabled} "
        f"error_px={error_px:+.1f} "
        f"target_x={'-' if target_x is None else f'{target_x:.1f}'} "
        f"calc_speed={speed:+.3f} "
        f"actual_speed={actual_speed:+.0f} "
        f"position_busy={position_busy} "
        f"max_speed={motor.max_speed:.1f} "
        f"motor_error={motor.last_error or '-'}",
        flush=True,
    )
    with state.lock:
        state.tracking_error_px = float(error_px)
        state.tracking_target_x = target_x
        state.tracking_motor_speed = float(actual_speed)
        state.tracking_error = "motor position command running" if position_busy else motor.last_error


def make_black_frame(text: str = "PRIVACY MODE") -> np.ndarray:
    frame = np.zeros((720, 860, 3), dtype=np.uint8)
    cv2.putText(frame, text, (230, 340), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (160, 160, 160), 2)
    cv2.putText(frame, "Camera view disabled", (250, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (110, 110, 110), 1)
    return frame


def vision_thread(state: FusionState, args: argparse.Namespace) -> None:
    yolo_rknn = None
    cls_rknn = None
    cap = None
    motor = None
    try:
        yolo_rknn = load_rknn_model(Path(args.yolo_model), args.yolo_npu_core)
        cls_rknn = load_rknn_model(Path(args.vision_model), args.vision_npu_core)
        motor = MotorController(args)
        pid = PIDController(args.track_kp, args.track_ki, args.track_kd, args.track_max_speed)

        trackers: list[SpeedTracker] = []
        vision_state = VisionLyingFallState(args.vision_lying_stable_frames, args.vision_lying_fall_hold_sec)
        frame_idx = 0
        last_motor_angle_request_id = 0

        def handle_missing_pose(frame_to_show: np.ndarray) -> None:
            cleared = vision_state.update_missing_pose()
            if cleared:
                print(
                    f"[vision] stable missing pose after {vision_state.stable_frames} missing frames",
                    flush=True,
                )
            with state.lock:
                state.vision_frame = frame_to_show
                if cleared:
                    state.vision_prob = None
                    state.vision_label = None
                    state.vision_raw_label = None
                    state.vision_time = 0.0
                    state.vision_error = None

        while not state.stop_event.is_set():
            with state.lock:
                current_mode = state.mode
                enabled = state.detection_enabled
                tracking_enabled = state.tracking_enabled
                tracking_motor_speed = state.tracking_motor_speed
                motor_angle = state.motor_angle
                motor_command = state.motor_command
                motor_angle_request_id = state.motor_angle_request_id
                debug_vision_label = state.debug_vision_label

            if motor_angle is not None and motor_angle_request_id != last_motor_angle_request_id:
                if tracking_enabled:
                    print(
                        f"[motor] manual request ignored while tracking is enabled "
                        f"(command={motor_command}, angle={motor_angle:.2f}, request_id={motor_angle_request_id})",
                        flush=True,
                    )
                else:
                    print(
                        f"[motor] executing manual request "
                        f"command={motor_command} angle={motor_angle:.2f} request_id={motor_angle_request_id}",
                        flush=True,
                    )
                    if motor_command == "zero":
                        motor.request_current_zero()
                    elif motor_command == "relative":
                        motor.request_relative_position(motor_angle)
                    else:
                        motor.request_position(motor_angle)
                last_motor_angle_request_id = motor_angle_request_id

            if current_mode == "radar":
                if cap is not None:
                    cap.release()
                    cap = None
                update_camera_tracking(state, motor, pid, None, None, False, args.track_deadband_px)
                vision_state.reset()
                trackers.clear()
                with state.lock:
                    state.vision_prob = None
                    state.vision_label = None
                    state.vision_raw_label = None
                    state.vision_time = 0.0
                    state.vision_frame = make_black_frame("PRIVACY MODE")
                    state.vision_error = None
                time.sleep(0.1)
                continue

            if cap is None:
                cap = cv2.VideoCapture(args.camera)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open camera: {args.camera}")

            ok, frame = cap.read()
            if not ok:
                update_camera_tracking(state, motor, pid, None, None, tracking_enabled, args.track_deadband_px)
                time.sleep(0.02)
                continue

            if not enabled:
                update_camera_tracking(state, motor, pid, None, None, False, args.track_deadband_px)
                vision_state.reset()
                trackers.clear()
                with state.lock:
                    state.vision_prob = None
                    state.vision_label = None
                    state.vision_raw_label = None
                    state.vision_time = 0.0
                    state.vision_frame = frame
                    state.vision_error = None
                time.sleep(0.02)
                continue

            boxes, det_scores = detect_persons(yolo_rknn, frame, args.yolo_conf, args.yolo_nms)
            valid_boxes = []
            crops = []
            for box in boxes:
                crop = crop_person(frame, box)
                if crop is not None:
                    valid_boxes.append(box)
                    crops.append(crop)

            if valid_boxes and crops:
                trackers, speed_by_box = update_speed(
                    valid_boxes,
                    trackers,
                    frame_idx,
                    args.speed_window,
                    tracking_motor_speed if tracking_enabled else 0.0,
                    args.speed_motor_comp_k,
                )
                target_idx = closest_center_box_index(valid_boxes, frame.shape[1])
                update_camera_tracking(state, motor, pid, frame, valid_boxes[target_idx], tracking_enabled, args.track_deadband_px)
                prob, uncertainty = vision_prob_from_crop(cls_rknn, crops[target_idx])
                if debug_vision_label:
                    vision_state.reset()
                    raw_label = debug_vision_label
                    filt_label = debug_vision_label
                    filt_prob = random_debug_prob(debug_vision_label)
                    drawn = draw_vision(frame, valid_boxes[target_idx], filt_label, filt_prob)
                    with state.lock:
                        state.vision_prob = filt_prob
                        state.vision_label = filt_label
                        state.vision_raw_label = raw_label
                        state.vision_time = time.time()
                        state.vision_frame = drawn
                        state.vision_error = None
                elif uncertainty <= args.uncertainty_threshold:
                    speed = speed_by_box.get(target_idx, 0.0)
                    if not args.no_speed:
                        prob, raw_label = apply_speed_override(prob, speed, args.speed_threshold)
                    else:
                        raw_label = VISION_CLASSES[int(np.argmax(prob))]
                    prob = soften_prob(prob, args.vision_temperature)

                    if prob[VISION_CLASSES.index(raw_label)] >= args.vision_conf:
                        filt_label, filt_prob = vision_state.update(raw_label, prob, time.time())
                        if filt_label is None or filt_prob is None:
                            with state.lock:
                                state.vision_prob = None
                                state.vision_label = None
                                state.vision_raw_label = raw_label
                                state.vision_time = 0.0
                                state.vision_frame = frame
                                state.vision_error = None
                        else:
                            drawn = draw_vision(frame, valid_boxes[target_idx], filt_label, filt_prob)
                            with state.lock:
                                state.vision_prob = filt_prob
                                state.vision_label = filt_label
                                state.vision_raw_label = raw_label
                                state.vision_time = time.time()
                                state.vision_frame = drawn
                                state.vision_error = None
                    else:
                        handle_missing_pose(frame)
                else:
                    handle_missing_pose(frame)
            else:
                for tracker in trackers:
                    tracker.mark_missed()
                trackers = [t for t in trackers if not t.is_stale()]
                update_camera_tracking(state, motor, pid, frame, None, tracking_enabled, args.track_deadband_px)
                handle_missing_pose(frame)

            frame_idx += 1
    except Exception as exc:
        with state.lock:
            state.vision_error = str(exc)
        state.stop_event.set()
    finally:
        if motor is not None:
            motor.close(send_stop=True)
        if cap is not None:
            cap.release()
        if yolo_rknn is not None:
            yolo_rknn.release()
        if cls_rknn is not None:
            cls_rknn.release()


def ps_to_distance_m(time_ps: np.ndarray) -> np.ndarray:
    return 0.5 * 299_792_458.0 * (time_ps * 1e-12)


def build_range_axis_m(num_bins: int, scan_start_ps: float, scan_stop_ps: float) -> np.ndarray:
    time_axis_ps = np.linspace(scan_start_ps, scan_stop_ps, num_bins, dtype=np.float64)
    return ps_to_distance_m(time_axis_ps).astype(np.float32)


def mti_alpha_filter(dc_removed: np.ndarray, alpha: float) -> np.ndarray:
    bg = np.empty_like(dc_removed, dtype=np.float32)
    prev = np.zeros(dc_removed.shape[1], dtype=np.float32)
    one_minus = np.float32(1.0 - alpha)
    alpha32 = np.float32(alpha)
    for i in range(dc_removed.shape[0]):
        prev = one_minus * dc_removed[i] + alpha32 * prev
        bg[i] = prev
    return dc_removed - bg


def iir_lfilter_axis1(x: np.ndarray, b: np.ndarray = LPF_B, a: np.ndarray = LPF_A) -> np.ndarray:
    y = np.empty_like(x, dtype=np.float32)
    x1 = np.zeros(x.shape[0], dtype=np.float32)
    x2 = np.zeros(x.shape[0], dtype=np.float32)
    x3 = np.zeros(x.shape[0], dtype=np.float32)
    y1 = np.zeros(x.shape[0], dtype=np.float32)
    y2 = np.zeros(x.shape[0], dtype=np.float32)
    y3 = np.zeros(x.shape[0], dtype=np.float32)
    for n in range(x.shape[1]):
        xn = x[:, n]
        yn = b[0] * xn + b[1] * x1 + b[2] * x2 + b[3] * x3 - a[1] * y1 - a[2] * y2 - a[3] * y3
        y[:, n] = yn
        x3, x2, x1 = x2, x1, xn
        y3, y2, y1 = y2, y1, yn
    return y


def build_tr_from_scan_matrix(scan_matrix: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    raw_range_m = build_range_axis_m(scan_matrix.shape[1], args.scan_start_ps, args.scan_stop_ps)
    corrected_range_m = raw_range_m - args.cable_compensation
    keep = (corrected_range_m >= args.distance_min) & (corrected_range_m <= args.distance_max)
    selected = scan_matrix[:, keep].astype(np.float32)
    dc_removed = selected - np.mean(selected, axis=0, keepdims=True)
    mti = mti_alpha_filter(dc_removed, args.mti_alpha)
    filtered = iir_lfilter_axis1(np.abs(mti))
    tr1 = np.maximum(filtered, 0.0)
    x_min = float(np.min(tr1))
    x_max = float(np.max(tr1))
    norm = (tr1 - x_min) / (x_max - x_min) if x_max - x_min > 1e-8 else np.zeros_like(tr1, dtype=np.float32)
    arr = np.clip(norm.T * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((RADAR_MODEL_SIZE, RADAR_MODEL_SIZE), resample=Image.Resampling.BILINEAR)
    gray_idx = np.asarray(pil, dtype=np.uint8).T
    tr_gray = VIRIDIS_GRAY_LUT[gray_idx].astype(np.float32) / 255.0
    try:
        tr_color = cv2.applyColorMap(gray_idx, cv2.COLORMAP_VIRIDIS)
    except Exception:
        tr_color = cv2.cvtColor(gray_idx, cv2.COLOR_GRAY2BGR)
    return tr_gray, tr_color


def radar_preprocess(tr_gray: np.ndarray, input_format: str) -> np.ndarray:
    x = (tr_gray.astype(np.float32) - RADAR_IMAGE_MEAN) / RADAR_IMAGE_STD
    if input_format == "nchw":
        return np.ascontiguousarray(x[None, None, :, :], dtype=np.float32)
    return np.ascontiguousarray(x[None, :, :, None], dtype=np.float32)


def radar_thread(state: FusionState, args: argparse.Namespace) -> None:
    radar_rknn = None
    try:
        radar_rknn = load_rknn_model(Path(args.radar_model), args.radar_npu_core)
        for scan_matrix in stream_datamatrix(
            ip=args.radar_ip,
            start_ps=args.scan_start_ps,
            stop_ps=args.scan_stop_ps,
            interval_us=args.interval_us,
            chunk_size=args.window_scans,
            program_start_time=time.time(),
        ):
            if state.stop_event.is_set():
                break
            tr_gray, tr_color = build_tr_from_scan_matrix(scan_matrix, args)
            with state.lock:
                detection_enabled = state.detection_enabled
                debug_radar_label = state.debug_radar_label
            if not detection_enabled:
                with state.lock:
                    state.radar_prob = None
                    state.radar_label = None
                    state.radar_time = 0.0
                    state.radar_image = tr_color
                    state.radar_error = None
                continue
            inp = radar_preprocess(tr_gray, args.radar_input_format)
            try:
                outputs = radar_rknn.inference(inputs=[inp], data_type="float32", data_format=args.radar_input_format)
            except TypeError:
                outputs = radar_rknn.inference(inputs=[inp])
            if outputs is None or len(outputs) == 0:
                raise RuntimeError("Radar RKNN inference returned no output")
            radar_prob = soften_prob(softmax_1d(np.asarray(outputs[0])), args.radar_temperature)
            label = RADAR_IDX_TO_CLASS[int(np.argmax(radar_prob))]
            fused_prob = radar_prob_to_fused(radar_prob)
            if debug_radar_label:
                if debug_radar_label in STATIC_DISPLAY_CLASSES:
                    label = STATIC_DISPLAY_LABEL
                    fused_prob = random_debug_radar_static_prob()
                else:
                    label = debug_radar_label
                    fused_prob = random_debug_prob(debug_radar_label)
            with state.lock:
                state.radar_prob = fused_prob
                state.radar_label = label
                state.radar_time = time.time()
                state.radar_image = tr_color
                state.radar_error = None
    except Exception as exc:
        with state.lock:
            state.radar_error = str(exc)
        state.stop_event.set()
    finally:
        if radar_rknn is not None:
            radar_rknn.release()


def radar_prob_to_fused(radar_prob: np.ndarray) -> np.ndarray:
    radar_prob = normalize_vector(radar_prob, len(RADAR_IDX_TO_CLASS))
    fused = np.zeros(len(FUSED_CLASSES), dtype=np.float32)
    static_conf = float(radar_prob[0])
    fused[class_index("stand")] = static_conf
    fused[class_index("lying")] = static_conf
    fused[class_index("sit")] = static_conf
    fused[class_index("walk")] = float(radar_prob[1])
    fused[class_index("bend")] = float(radar_prob[2])
    fused[class_index("fall")] = float(radar_prob[3])
    return fused


def fusion_weights_for_class(label: str, dynamic_radar_weight: float, static_radar_weight: float) -> tuple[float, float]:
    if label in ("stand", "lying", "sit"):
        radar_w = float(static_radar_weight)
    else:
        radar_w = float(dynamic_radar_weight)
    radar_w = max(0.0, min(1.0, radar_w))
    return 1.0 - radar_w, radar_w


def fuse_probs(
    vision_prob: np.ndarray | None,
    vision_time: float,
    radar_prob: np.ndarray | None,
    radar_time: float,
    now: float,
    radar_stale_sec: float,
    vision_stale_sec: float,
    dynamic_radar_weight: float,
    static_radar_weight: float,
) -> tuple[np.ndarray | None, str | None, float, float]:
    has_vision = vision_prob is not None and (now - vision_time) <= vision_stale_sec
    has_radar = radar_prob is not None and (now - radar_time) <= radar_stale_sec
    if not has_vision and not has_radar:
        return None, None, 0.0, 0.0
    if not has_vision:
        fused = np.asarray(radar_prob, dtype=np.float32).reshape(-1)
        label = FUSED_CLASSES[int(np.argmax(fused))]
        return fused, label, 0.0, 1.0

    if has_radar:
        vision = np.asarray(vision_prob, dtype=np.float32).reshape(-1)
        radar = np.asarray(radar_prob, dtype=np.float32).reshape(-1)
        fused = np.zeros(len(FUSED_CLASSES), dtype=np.float32)
        for i, cls in enumerate(FUSED_CLASSES):
            vw, rw = fusion_weights_for_class(cls, dynamic_radar_weight, static_radar_weight)
            fused[i] = vw * float(vision[i]) + rw * float(radar[i])
    else:
        fused = np.asarray(vision_prob, dtype=np.float32).reshape(-1)
    label = FUSED_CLASSES[int(np.argmax(fused))]
    vw, rw = fusion_weights_for_class(label, dynamic_radar_weight, static_radar_weight) if has_radar else (1.0, 0.0)
    return fused, label, vw, rw


def draw_bar(panel: np.ndarray, x: int, y: int, w: int, h: int, value: float, color: tuple[int, int, int], text: str) -> None:
    cv2.rectangle(panel, (x, y), (x + w, y + h), (80, 80, 80), 1)
    fill = int(w * max(0.0, min(1.0, value)))
    cv2.rectangle(panel, (x, y), (x + fill, y + h), color, -1)
    cv2.putText(panel, text, (x + 8, y + h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def get_display_font(size: int) -> ImageFont.ImageFont:
    cached = _PIL_FONT_CACHE.get(size)
    if cached is not None:
        return cached

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for font_path in candidates:
        try:
            font = ImageFont.truetype(font_path, size)
            _PIL_FONT_CACHE[size] = font
            return font
        except Exception:
            pass

    font = ImageFont.load_default()
    _PIL_FONT_CACHE[size] = font
    return font


def draw_unicode_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    size: int = 18,
    color_bgr: tuple[int, int, int] = (230, 230, 230),
) -> None:
    if not text:
        return
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((x, y), text, font=get_display_font(size), fill=color_rgb)
    img[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def compact_text(text: str, max_chars: int = 22) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def get_board_ip() -> str:
    ips: list[str] = []

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ips.append(sock.getsockname()[0])
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            ips.append(addr)
    except OSError:
        pass

    for ip in ips:
        if ip and not ip.startswith("127.") and ip != "0.0.0.0":
            return ip
    return "-"


def draw_board_ip(panel: np.ndarray, board_ip: str) -> None:
    text = f"IP: {compact_text(board_ip or '-', 16)}"
    x, y, w, h = 230, 650, 150, 26
    cv2.rectangle(panel, (x, y), (x + w, y + h), (24, 24, 24), -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (85, 85, 85), 1)
    cv2.putText(panel, text, (x + 7, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (225, 225, 225), 1)


def parse_url_endpoint(url: str) -> tuple[str, str, int | None, str]:
    try:
        parsed = urllib.parse.urlparse(url)
    except ValueError as exc:
        return "", "", None, f"parse_error={type(exc).__name__}:{exc}"
    host = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is None and parsed.scheme == "http":
        port = 80
    elif port is None and parsed.scheme == "https":
        port = 443
    return parsed.scheme or "", host, port, ""


def resolve_host_summary(host: str, port: int | None) -> str:
    if not host:
        return "host-empty"
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET)
        addrs = sorted({info[4][0] for info in infos})
        return ",".join(addrs[:4]) if addrs else "no-ipv4-result"
    except OSError as exc:
        return f"resolve_error={type(exc).__name__}:{exc}"


def exception_detail(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    errno = getattr(reason, "errno", getattr(exc, "errno", None))
    reason_text = f" reason_type={type(reason).__name__} reason={reason}" if reason is not None else ""
    errno_text = f" errno={errno}" if errno is not None else ""
    return f"type={type(exc).__name__}{errno_text} error={exc}{reason_text}"


def network_log_key(kind: str, url: str, exc: BaseException) -> tuple[str, str, str]:
    reason = getattr(exc, "reason", None)
    reason_text = str(reason if reason is not None else exc)
    return kind, url, f"{type(exc).__name__}:{reason_text[:120]}"


def log_network_error(kind: str, url: str, exc: BaseException) -> None:
    key = network_log_key(kind, url, exc)
    now = time.time()
    with _NETWORK_ERROR_LOG_LOCK:
        last = _NETWORK_ERROR_LOG_TIMES.get(key, 0.0)
        if now - last < NETWORK_ERROR_LOG_SEC:
            return
        _NETWORK_ERROR_LOG_TIMES[key] = now

    scheme, host, port, parse_error = parse_url_endpoint(url)
    resolve_text = parse_error or resolve_host_summary(host, port)
    print(
        f"[{kind}] request failed "
        f"url={url or '-'} scheme={scheme or '-'} host={host or '-'} port={port if port is not None else '-'} "
        f"board_ip={get_board_ip()} resolve={resolve_text} {exception_detail(exc)}",
        flush=True,
    )


def log_url_diagnostic(name: str, url: str) -> None:
    if not url:
        print(f"[network] {name}=disabled", flush=True)
        return
    scheme, host, port, parse_error = parse_url_endpoint(url)
    resolve_text = parse_error or resolve_host_summary(host, port)
    print(
        f"[network] {name}={url} scheme={scheme or '-'} host={host or '-'} "
        f"port={port if port is not None else '-'} board_ip={get_board_ip()} resolve={resolve_text}",
        flush=True,
    )


def build_status_url(value: str, default_port: int | str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""

    if "://" not in raw:
        if "/" in raw or ":" in raw:
            raw = f"http://{raw}"
        else:
            return f"http://{raw}:{default_port}/status"

    try:
        parsed = urllib.parse.urlparse(raw)
    except ValueError:
        return raw

    if not parsed.netloc:
        return raw

    path = parsed.path or "/status"
    if path == "/":
        path = "/status"

    return urllib.parse.urlunparse((parsed.scheme or "http", parsed.netloc, path, "", "", ""))


def draw_ai_button(panel: np.ndarray, ai_status: dict | None, panel_x: int) -> None:
    if not ai_status:
        return

    if ai_status.get("fall_takeover_active"):
        x, y, w, h = AI_TAKEOVER_CANCEL_RECT
        x -= panel_x
        cv2.rectangle(panel, (x, y), (x + w, y + h), (30, 80, 210), -1)
        cv2.rectangle(panel, (x, y), (x + w, y + h), (230, 230, 230), 1)
        cv2.putText(panel, "EXIT", (x + 13, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(panel, "AI takeover", (x - 4, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)

    x, y, w, h = AI_BUTTON_RECT
    x -= panel_x
    state = ai_status.get("state", "idle")
    enabled = bool(ai_status.get("enabled", True))
    label_by_state = {
        "idle": "AI",
        "recording": "AI",
        "processing": "AI",
        "speaking": "AI",
        "error": "AI",
    }
    label = label_by_state.get(state, "AI")
    if state == "recording":
        color = (0, 80, 220)
    elif enabled:
        color = (50, 145, 70)
    else:
        color = (90, 90, 90)

    cv2.rectangle(panel, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (210, 210, 210), 1)
    cv2.putText(panel, label, (x + 23, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (255, 255, 255), 2)

    status = str(ai_status.get("status") or state)
    cv2.putText(panel, f"AI:{status[:8]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)
    error = str(ai_status.get("last_error") or "")
    if state == "error" and error:
        cv2.putText(panel, error[:12], (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)

    last_asr = str(ai_status.get("last_asr") or "")
    if last_asr:
        cv2.rectangle(panel, (212, 568), (396, 600), (28, 28, 28), -1)
        cv2.rectangle(panel, (212, 568), (396, 600), (80, 80, 80), 1)
        draw_unicode_text(panel, f"识别：{compact_text(last_asr, 8)}", 220, 574, 15, (235, 235, 235))

    last_reply = str(ai_status.get("last_reply") or "")
    if last_reply:
        cv2.rectangle(panel, (212, 530), (396, 562), (28, 28, 28), -1)
        cv2.rectangle(panel, (212, 530), (396, 562), (80, 80, 80), 1)
        draw_unicode_text(panel, f"回复：{compact_text(last_reply, 8)}", 220, 536, 15, (210, 245, 210))

def draw_ai_button_v2(panel: np.ndarray, ai_status: dict | None, panel_x: int) -> None:
    if not ai_status:
        return

    if ai_status.get("fall_takeover_active"):
        x, y, w, h = AI_TAKEOVER_CANCEL_RECT
        x -= panel_x
        cv2.rectangle(panel, (x, y), (x + w, y + h), (30, 80, 210), -1)
        cv2.rectangle(panel, (x, y), (x + w, y + h), (230, 230, 230), 1)
        cv2.putText(panel, "EXIT", (x + 44, y + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

    x, y, w, h = AI_BUTTON_RECT
    x -= panel_x
    state = ai_status.get("state", "idle")
    enabled = bool(ai_status.get("enabled", True))
    if state == "recording":
        color = (55, 170, 65)
    elif enabled:
        color = (0, 55, 210)
    else:
        color = (75, 75, 95)

    cv2.rectangle(panel, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(panel, (x, y), (x + w, y + h), (230, 230, 230), 1)
    draw_unicode_text(panel, "按下说话", x + 18, y + 20, 26, (255, 255, 255))

    status = str(ai_status.get("status") or state)
    cv2.putText(panel, f"AI:{status[:14]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 210), 1)
    error = str(ai_status.get("last_error") or "")
    if error:
        cv2.putText(panel, error[:18], (x, y + h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)

    last_reply = str(ai_status.get("last_reply") or "")
    if last_reply:
        cv2.rectangle(panel, (230, 470), (396, 506), (28, 28, 28), -1)
        cv2.rectangle(panel, (230, 470), (396, 506), (80, 80, 80), 1)
        draw_unicode_text(panel, f"回复：{compact_text(last_reply, 5)}", 238, 478, 15, (210, 245, 210))

    last_asr = str(ai_status.get("last_asr") or "")
    if last_asr:
        cv2.rectangle(panel, (20, 680), (396, 714), (28, 28, 28), -1)
        cv2.rectangle(panel, (20, 680), (396, 714), (80, 80, 80), 1)
        draw_unicode_text(panel, f"识别：{compact_text(last_asr, 18)}", 28, 686, 16, (235, 235, 235))


def post_alarm_async(url: str, payload: dict, timeout: float) -> None:
    if not url:
        return

    def worker() -> None:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            print(
                f"[alarm] sent alarm={payload.get('alarm')} "
                f"confidence={payload.get('confidence')} "
                f"label={payload.get('label')} "
                f"timestamp={payload.get('timestamp')} "
                f"url={url}",
                flush=True,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log_network_error("alarm", url, exc)

    threading.Thread(target=worker, daemon=True).start()


def post_status_async(url: str, payload: dict, timeout: float) -> None:
    if not url:
        return

    def worker() -> None:
        start_perf = time.perf_counter()
        body = json.dumps(payload).encode("utf-8")
        preview_image = payload.get("previewImage") or ""
        preview_ts = payload.get("previewImageTimestamp") or payload.get("timestamp")
        preview_seq = payload.get("previewImageSeq")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp.read()
            if preview_image:
                http_ms = (time.perf_counter() - start_perf) * 1000.0
                try:
                    source_age_ms = int(time.time() * 1000) - int(preview_ts)
                except (TypeError, ValueError):
                    source_age_ms = -1
                print(
                    f"[status] preview sent "
                    f"seq={preview_seq} timestamp={preview_ts} "
                    f"source_age_ms={source_age_ms} http_ms={http_ms:.1f} "
                    f"chars={len(preview_image)} body_bytes={len(body)} url={url}",
                    flush=True,
                )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if preview_image:
                http_ms = (time.perf_counter() - start_perf) * 1000.0
                print(
                    f"[status] preview send failed "
                    f"seq={preview_seq} timestamp={preview_ts} http_ms={http_ms:.1f} "
                    f"chars={len(preview_image)} body_bytes={len(body)}",
                    flush=True,
                )
            log_network_error("status", url, exc)

    threading.Thread(target=worker, daemon=True).start()


def encode_alarm_image(frame: np.ndarray | None, max_width: int, jpeg_quality: int) -> str | None:
    if frame is None:
        return None
    image = frame
    h, w = image.shape[:2]
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        image = cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def make_alarm_payload(alarm: bool, confidence: float, label: str | None, image: str | None = None) -> dict:
    payload = {
        "enabled": True,
        "mode": "fusion",
        "alarm": bool(alarm),
        "confidence": float(confidence),
        "label": label or "",
        "timestamp": int(time.time() * 1000),
    }
    if image:
        payload["image"] = image
    return payload


def make_status_payload(
    label: str,
    confidence: float,
    preview_image: str | None = None,
    reminders: list[dict] | None = None,
    reminder_notification: dict | None = None,
    reminder_handled_request_id: int | None = None,
    system_command_handled_request_id: int | None = None,
    board_message_handled_request_id: int | None = None,
    board_message_notification: dict | None = None,
    control_handled_request_id: int | None = None,
    enabled: bool | None = None,
    mode: str | None = None,
    debug_vision_label: str | None = None,
    debug_radar_label: str | None = None,
    preview_image_seq: int | None = None,
    preview_encode_ms: float | None = None,
    status_post_elapsed: float | None = None,
    status_image_elapsed: float | None = None,
) -> dict:
    payload = {
        "label": label,
        "confidence": float(confidence),
        "timestamp": int(time.time() * 1000),
    }
    if enabled is not None:
        payload["enabled"] = bool(enabled)
    if mode:
        payload["mode"] = mode
    payload["debugVisionLabel"] = debug_vision_label or ""
    payload["debugRadarLabel"] = debug_radar_label or ""
    if preview_image:
        payload["previewImage"] = preview_image
        payload["previewImageTimestamp"] = payload["timestamp"]
        if preview_image_seq is not None:
            payload["previewImageSeq"] = int(preview_image_seq)
        if preview_encode_ms is not None:
            payload["previewEncodeMs"] = round(float(preview_encode_ms), 3)
        if status_post_elapsed is not None:
            payload["statusPostElapsedSec"] = round(float(status_post_elapsed), 3)
        if status_image_elapsed is not None:
            payload["statusImageElapsedSec"] = round(float(status_image_elapsed), 3)
    if reminders is not None:
        payload["reminders"] = reminders
    payload["reminderNotification"] = reminder_notification
    if reminder_handled_request_id is not None:
        payload["reminderHandledRequestId"] = int(reminder_handled_request_id)
    if system_command_handled_request_id is not None:
        payload["systemCommandHandledRequestId"] = int(system_command_handled_request_id)
    if board_message_handled_request_id is not None:
        payload["boardMessageHandledRequestId"] = int(board_message_handled_request_id)
    payload["boardMessageNotification"] = board_message_notification
    if control_handled_request_id is not None:
        payload["controlHandledRequestId"] = int(control_handled_request_id)
    return payload


def get_control_status(url: str, timeout: float) -> dict | None:
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except json.JSONDecodeError as exc:
        log_network_error("control-json", url, exc)
        return {"_error": f"invalid control JSON: {exc}"}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        log_network_error("control", url, exc)
        return {"_error": str(exc)}


def clamp_motor_angle(angle: float, args: argparse.Namespace) -> float:
    return max(float(args.motor_angle_min), min(float(args.motor_angle_max), float(angle)))


def initialize_motor_zero(args: argparse.Namespace) -> None:
    if not args.motor_init_zero:
        return
    try:
        import serial_location

        ser = serial_location.open_serial(args.motor_port, args.motor_baudrate, timeout=1.0)
        try:
            serial_location.set_current_zero(ser, args.motor_init_zero_delay)
        finally:
            ser.close()
        print("[motor] current position set as zero", flush=True)
    except Exception as exc:
        print(f"[motor] init zero failed: {exc}", flush=True)


def execute_system_command(command: str) -> None:
    command = str(command or "").strip().lower()
    if command not in ("poweroff", "reboot"):
        return
    print(f"[system] executing {command}", flush=True)
    subprocess.Popen([command])


def normalize_message_text(value: str, max_len: int = 500) -> str:
    return str(value or "").strip()[:max_len]


def update_debug_override_state(state: FusionState, command: dict, data: dict) -> None:
    if "debugVisionLabel" in command:
        new_label = normalize_debug_label(command.get("debugVisionLabel"))
        if new_label != state.debug_vision_label:
            print(f"[debug] vision_override={new_label or '-'}", flush=True)
        state.debug_vision_label = new_label
    elif "debugVisionLabel" in data:
        new_label = normalize_debug_label(data.get("debugVisionLabel"))
        if new_label != state.debug_vision_label:
            print(f"[debug] vision_override={new_label or '-'}", flush=True)
        state.debug_vision_label = new_label

    if "debugRadarLabel" in command:
        new_label = normalize_debug_label(command.get("debugRadarLabel"))
        if new_label != state.debug_radar_label:
            print(f"[debug] radar_override={new_label or '-'}", flush=True)
        state.debug_radar_label = new_label
    elif "debugRadarLabel" in data:
        new_label = normalize_debug_label(data.get("debugRadarLabel"))
        if new_label != state.debug_radar_label:
            print(f"[debug] radar_override={new_label or '-'}", flush=True)
        state.debug_radar_label = new_label


def control_thread(state: FusionState, args: argparse.Namespace, reminder_manager: ReminderManager | None = None) -> None:
    last_error: str | None = None
    while not state.stop_event.is_set():
        data = get_control_status(args.control_url, args.control_timeout)
        system_command_to_execute = ""
        if data:
            with state.lock:
                if "_error" in data:
                    last_error = data["_error"]
                    state.control_error = last_error
                else:
                    if last_error:
                        print("[control] GET recovered", flush=True)
                        last_error = None
                    handled_control_request = False
                    if data.get("controlRequestId") is not None:
                        try:
                            control_request_id = int(data["controlRequestId"])
                        except (TypeError, ValueError):
                            control_request_id = state.control_request_id
                        if control_request_id > 0:
                            handled_control_request = True
                            if control_request_id != state.control_request_id:
                                command = data.get("controlCommand") or {}
                                if not isinstance(command, dict):
                                    command = {}
                                new_mode = command.get("mode")
                                if new_mode not in ("radar", "vision"):
                                    new_mode = data.get("mode")
                                if new_mode in ("radar", "vision"):
                                    if new_mode != state.mode:
                                        print(f"[control] mode={new_mode}", flush=True)
                                    state.mode = new_mode
                                new_enabled = command.get("enabled")
                                if not isinstance(new_enabled, bool):
                                    new_enabled = data.get("enabled")
                                if isinstance(new_enabled, bool):
                                    if new_enabled != state.detection_enabled:
                                        print(f"[control] detection_enabled={new_enabled}", flush=True)
                                    state.detection_enabled = new_enabled
                                    if not new_enabled:
                                        clear_detection_results_locked(state)
                                new_tracking = command.get("tracking")
                                if not isinstance(new_tracking, bool):
                                    new_tracking = data.get("tracking")
                                if isinstance(new_tracking, bool):
                                    if new_tracking != state.tracking_enabled:
                                        print(f"[control] tracking_enabled={new_tracking}", flush=True)
                                    state.tracking_enabled = new_tracking
                                update_debug_override_state(state, command, data)
                                state.control_request_id = control_request_id
                                print(
                                    "[control] state request handled "
                                    f"request_id={state.control_request_id} "
                                    f"enabled={state.detection_enabled} "
                                    f"mode={state.mode} "
                                    f"tracking={state.tracking_enabled} "
                                    f"debug_vision={state.debug_vision_label or '-'} "
                                    f"debug_radar={state.debug_radar_label or '-'}",
                                    flush=True,
                                )
                    if not handled_control_request:
                        if data.get("mode") in ("radar", "vision"):
                            new_mode = data["mode"]
                            if new_mode != state.mode:
                                print(f"[control] mode={new_mode}", flush=True)
                            state.mode = new_mode
                        if isinstance(data.get("enabled"), bool):
                            new_enabled = data["enabled"]
                            if new_enabled != state.detection_enabled:
                                print(f"[control] detection_enabled={new_enabled}", flush=True)
                            state.detection_enabled = new_enabled
                            if not new_enabled:
                                clear_detection_results_locked(state)
                        if isinstance(data.get("tracking"), bool):
                            new_tracking = data["tracking"]
                            if new_tracking != state.tracking_enabled:
                                print(f"[control] tracking_enabled={new_tracking}", flush=True)
                            state.tracking_enabled = new_tracking
                        update_debug_override_state(state, {}, data)
                    if data.get("alarmAckTimestamp") is not None:
                        try:
                            state.alarm_ack_timestamp = max(
                                state.alarm_ack_timestamp,
                                int(data["alarmAckTimestamp"]),
                            )
                        except (TypeError, ValueError):
                            pass
                    if data.get("motorAngle") is not None and data.get("motorAngleRequestId") is not None:
                        try:
                            request_id = int(data["motorAngleRequestId"])
                            if request_id != state.motor_angle_request_id:
                                command = data.get("motorCommand")
                                if command not in ("absolute", "zero", "relative"):
                                    command = "absolute"
                                command_angle = data.get("motorCommandAngle")
                                if command_angle is None:
                                    command_angle = data.get("motorAngle")
                                state.motor_command = command
                                state.motor_angle = clamp_motor_angle(float(command_angle), args)
                                state.motor_angle_request_id = request_id
                                print(
                                    "[motor] control received "
                                    f"command={state.motor_command} "
                                    f"angle={state.motor_angle:.2f} "
                                    f"request_id={state.motor_angle_request_id} "
                                    f"tracking={state.tracking_enabled}",
                                    flush=True,
                                )
                        except (TypeError, ValueError):
                            print(
                                f"[motor] invalid control payload: "
                                f"motorAngle={data.get('motorAngle')} "
                                f"motorAngleRequestId={data.get('motorAngleRequestId')}",
                                flush=True,
                            )
                    if data.get("reminderRequestId") is not None:
                        try:
                            request_id = int(data["reminderRequestId"])
                        except (TypeError, ValueError):
                            request_id = state.reminder_request_id
                        if reminder_manager is not None and request_id != state.reminder_request_id:
                            command = data.get("reminderCommand") or {}
                            ok, message = reminder_manager.apply_command(command)
                            state.reminder_request_id = request_id
                            state.reminders = reminder_manager.snapshot()
                            print(
                                f"[reminder] command request_id={request_id} ok={ok} message={message}",
                                flush=True,
                            )
                    if data.get("systemCommandRequestId") is not None:
                        try:
                            request_id = int(data["systemCommandRequestId"])
                        except (TypeError, ValueError):
                            request_id = state.system_command_request_id
                        if request_id != state.system_command_request_id:
                            command = data.get("systemCommand") or {}
                            command_name = str(command.get("command") or "").strip().lower()
                            if command_name in ("poweroff", "reboot"):
                                state.system_command_request_id = request_id
                                system_command_to_execute = command_name
                                print(
                                    f"[system] command request_id={request_id} command={command_name}",
                                    flush=True,
                                )
                    if data.get("boardMessageRequestId") is not None:
                        try:
                            request_id = int(data["boardMessageRequestId"])
                        except (TypeError, ValueError):
                            request_id = state.board_message_request_id
                        if request_id != state.board_message_request_id:
                            command = data.get("boardMessageCommand") or {}
                            text = normalize_message_text(command.get("text") or command.get("message") or "")
                            if text:
                                event = {
                                    "id": str(command.get("id") or request_id),
                                    "text": text,
                                    "source": str(command.get("source") or "app"),
                                    "timestamp": int(command.get("timestamp") or int(time.time() * 1000)),
                                }
                                state.board_message_request_id = request_id
                                state.board_message_notification = event
                                print(
                                    f"[message] command request_id={request_id} text={text}",
                                    flush=True,
                                )
                    state.control_error = None
        if system_command_to_execute:
            execute_system_command(system_command_to_execute)
        state.stop_event.wait(args.control_poll_sec)


def draw_reminder_popup(canvas: np.ndarray, notification: dict | None) -> None:
    if not notification:
        return
    label = str(notification.get("label") or "提醒")
    scheduled_text = str(notification.get("scheduledText") or "")
    text = f"提醒：{label}"
    if scheduled_text:
        text = f"{text}  {scheduled_text}"
    x, y, w, h = 130, 18, 760, 58
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (25, 105, 230), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (245, 245, 245), 2)
    draw_unicode_text(canvas, compact_text(text, 30), x + 22, y + 14, 25, (255, 255, 255))


def draw_board_message_popup(canvas: np.ndarray, notification: dict | None, y: int = 84) -> None:
    if not notification:
        return
    text = normalize_message_text(notification.get("text") or "")
    if not text:
        return
    title = f"来自App的信息：{text}"
    x, w, h = 130, 760, 58
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 125, 80), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (245, 245, 245), 2)
    draw_unicode_text(canvas, compact_text(title, 30), x + 22, y + 14, 25, (255, 255, 255))


def make_display(
    frame: np.ndarray | None,
    radar_img: np.ndarray | None,
    fused_prob: np.ndarray | None,
    fused_label: str | None,
    vision_label: str | None,
    radar_label: str | None,
    vision_w: float,
    radar_w: float,
    vision_age: float,
    radar_age: float,
    tracking_enabled: bool,
    tracking_motor_speed: float,
    tracking_error_px: float,
    tracking_target_x: float | None,
    errors: tuple[str | None, str | None, str | None, str | None],
    mode: str,
    detection_enabled: bool,
    board_ip: str,
    ai_status: dict | None = None,
    reminder_notification: dict | None = None,
    board_message_notification: dict | None = None,
) -> np.ndarray:
    width, height = 1280, 720
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if frame is not None:
        fh, fw = frame.shape[:2]
        scale = min(860 / fw, height / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        y0 = (height - new_h) // 2
        canvas[y0 : y0 + new_h, 0:new_w] = resized

    panel_x = 880
    panel = canvas[:, panel_x:]
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), (60, 60, 60), 1)
    draw_ai_button_v2(panel, ai_status, panel_x)
    mode_text = "privacy/radar" if mode == "radar" else "daily/fusion"
    cv2.putText(panel, f"mode: {mode_text}  enabled: {detection_enabled}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1)
    tracking_text = "on" if tracking_enabled else "off"
    target_text = "-" if tracking_target_x is None else f"{tracking_target_x:.0f}px"
    cv2.putText(
        panel,
        f"tracking: {tracking_text}  motor={tracking_motor_speed:+.0f}  err={tracking_error_px:+.0f}px  x={target_text}",
        (20, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (190, 190, 190),
        1,
    )

    if not detection_enabled:
        cv2.putText(panel, "Detection disabled", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    elif fused_label is None or fused_prob is None:
        cv2.putText(panel, "Waiting for modalities...", (20, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
    else:
        display_label = display_label_for_result(fused_label, vision_label, vision_w, mode) or fused_label
        color = FUSED_COLORS.get(fused_label, (255, 255, 255))
        cv2.putText(panel, "FUSED RESULT", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        cv2.putText(panel, f"{CLASS_CN.get(display_label, display_label)}  {fused_prob[class_index(fused_label)]:.2%}", (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(panel, f"vision: {vision_label or '-'}", (20, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 230, 230), 2)
        cv2.putText(panel, f"w={vision_w:.2f} age={vision_age:.1f}s", (210, 158), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        cv2.putText(panel, f"radar : {radar_label or '-'}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (230, 230, 230), 2)
        cv2.putText(panel, f"w={radar_w:.2f} age={radar_age:.1f}s", (210, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        y = 225
        for i, cls in enumerate(FUSED_CLASSES):
            draw_bar(panel, 20, y + i * 34, 330, 24, float(fused_prob[i]), FUSED_COLORS[cls], f"{cls:6s} {fused_prob[i]:.2%}")

    if radar_img is not None:
        rimg = cv2.resize(radar_img, (190, 190), interpolation=cv2.INTER_NEAREST)
        y0 = 465
        x0 = 20
        panel[y0 : y0 + 190, x0 : x0 + 190] = rimg
        cv2.putText(panel, "radar TR", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    draw_board_ip(panel, board_ip)

    vision_error, radar_error, control_error, tracking_error = errors
    if vision_error:
        cv2.putText(panel, f"vision error: {vision_error[:42]}", (20, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if radar_error:
        cv2.putText(panel, f"radar error: {radar_error[:42]}", (20, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if control_error:
        cv2.putText(panel, f"control error: {control_error[:42]}", (20, 695), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    if tracking_error:
        cv2.putText(panel, f"tracking error: {tracking_error[:42]}", (20, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    draw_reminder_popup(canvas, reminder_notification)
    draw_board_message_popup(canvas, board_message_notification, 84 if reminder_notification else 18)
    return canvas


def run(args: argparse.Namespace) -> int:
    if args.app_host:
        app_status_url = build_status_url(args.app_host, args.app_port)
        if not args.alarm_url:
            args.alarm_url = app_status_url
        if not args.control_url:
            args.control_url = app_status_url
    elif not args.control_url and args.alarm_url:
        args.control_url = args.alarm_url

    if args.alarm_url:
        args.alarm_url = build_status_url(args.alarm_url, args.app_port)
    if args.control_url:
        args.control_url = build_status_url(args.control_url, args.app_port)

    print(f"[config] alarm_url={args.alarm_url or '(disabled)'}", flush=True)
    print(f"[config] control_url={args.control_url or '(disabled)'}", flush=True)
    print(
        f"[config] status_post_sec={args.status_post_sec} "
        f"status_image_sec={args.status_image_sec} "
        f"status_image_width={args.status_image_width} "
        f"status_jpeg_quality={args.status_jpeg_quality}",
        flush=True,
    )
    log_url_diagnostic("alarm_url", args.alarm_url)
    log_url_diagnostic("control_url", args.control_url)

    initialize_motor_zero(args)

    state = FusionState()
    state.tracking_enabled = bool(args.tracking)
    reminder_manager = ReminderManager(Path(args.reminders_file), args.reminder_due_grace_sec)
    with state.lock:
        state.reminders = reminder_manager.snapshot()
    vt = threading.Thread(target=vision_thread, args=(state, args), daemon=True)
    rt = threading.Thread(target=radar_thread, args=(state, args), daemon=True)
    ct = threading.Thread(target=control_thread, args=(state, args, reminder_manager), daemon=True) if args.control_url else None
    vt.start()
    rt.start()
    if ct is not None:
        ct.start()

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, float(args.output_fps), (1280, 720))

    ai_assistant = None
    if args.ai_assistant:
        try:
            from ai_voice_assistant import AIVoiceAssistant

            ai_assistant = AIVoiceAssistant(args)
            print("[ai] assistant enabled", flush=True)
        except Exception as exc:
            print(f"[ai] assistant init failed: {exc}", flush=True)

    fall_followup_cancel_event = threading.Event()

    def on_mouse(event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or ai_assistant is None:
            return
        if point_in_rect(x, y, AI_TAKEOVER_CANCEL_RECT):
            if ai_assistant.cancel_fall_followup():
                fall_followup_cancel_event.set()
                print("[alarm] fall AI takeover cancelled by screen button", flush=True)
            return
        if point_in_rect(x, y, AI_BUTTON_RECT):
            ai_assistant.toggle_recording()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    fall_alarm_active = False
    last_fall_alarm_time = 0.0
    last_status_post_time = 0.0
    last_status_image_time = 0.0
    status_preview_seq = 0
    pending_fall_alarm_timestamp = 0
    pending_fall_alarm_time = 0.0
    pending_fall_followup_started = False
    active_reminder_notification: dict | None = None
    active_reminder_until = 0.0
    active_board_message_notification: dict | None = None
    active_board_message_until = 0.0
    last_board_message_id = ""
    board_ip = get_board_ip()
    last_board_ip_refresh = 0.0
    print(f"[network] board_ip={board_ip}", flush=True)

    try:
        while not state.stop_event.is_set():
            now = time.time()
            if now - last_board_ip_refresh >= BOARD_IP_REFRESH_SEC:
                board_ip = get_board_ip()
                last_board_ip_refresh = now
            with state.lock:
                vision_prob = None if state.vision_prob is None else state.vision_prob.copy()
                radar_prob = None if state.radar_prob is None else state.radar_prob.copy()
                vision_time = state.vision_time
                radar_time = state.radar_time
                frame = None if state.vision_frame is None else state.vision_frame.copy()
                radar_img = None if state.radar_image is None else state.radar_image.copy()
                vision_label = state.vision_label
                radar_label = state.radar_label
                mode = state.mode
                detection_enabled = state.detection_enabled
                tracking_enabled = state.tracking_enabled
                debug_vision_label = state.debug_vision_label
                debug_radar_label = state.debug_radar_label
                tracking_motor_speed = state.tracking_motor_speed
                tracking_error_px = state.tracking_error_px
                tracking_target_x = state.tracking_target_x
                alarm_ack_timestamp = state.alarm_ack_timestamp
                reminders_snapshot = [dict(item) for item in state.reminders]
                reminder_request_id = state.reminder_request_id
                control_request_id = state.control_request_id
                system_command_request_id = state.system_command_request_id
                board_message_request_id = state.board_message_request_id
                reminder_notification = state.reminder_notification
                incoming_board_message = state.board_message_notification
                errors = (state.vision_error, state.radar_error, state.control_error, state.tracking_error)

            if not detection_enabled:
                vision_prob = None
                radar_prob = None
                vision_time = 0.0
                radar_time = 0.0
                vision_label = None
                radar_label = None

            due_reminders = reminder_manager.check_due(datetime.now())
            if due_reminders:
                reminders_snapshot = reminder_manager.snapshot()
                event = due_reminders[-1]
                if len(due_reminders) > 1:
                    labels = [str(item.get("label") or "提醒") for item in due_reminders]
                    event = dict(event)
                    event["label"] = "、".join(labels)
                    event["message"] = "提醒：" + "；".join(labels)
                active_reminder_notification = event
                active_reminder_until = now + args.reminder_popup_sec
                with state.lock:
                    state.reminders = reminders_snapshot
                    state.reminder_notification = event
                print(f"[reminder] due id={event.get('id')} label={event.get('label')}", flush=True)
                if ai_assistant is not None:
                    ai_assistant.start_reminder(event.get("message") or f"提醒：{event.get('label')}")

            if active_reminder_notification is not None and now >= active_reminder_until:
                active_reminder_notification = None
                with state.lock:
                    state.reminder_notification = None

            reminder_notification = active_reminder_notification

            if incoming_board_message:
                message_id = str(incoming_board_message.get("id") or "")
                if message_id and message_id != last_board_message_id:
                    active_board_message_notification = dict(incoming_board_message)
                    active_board_message_until = now + 10.0
                    last_board_message_id = message_id
                    text = normalize_message_text(active_board_message_notification.get("text") or "")
                    print(f"[message] app message active id={message_id} text={text}", flush=True)
                    if ai_assistant is not None and text:
                        ai_assistant.start_reminder(f"来自app的信息：{text}")

            if active_board_message_notification is not None and now >= active_board_message_until:
                active_board_message_notification = None
                with state.lock:
                    current = state.board_message_notification
                    if current and str(current.get("id") or "") == last_board_message_id:
                        state.board_message_notification = None

            board_message_notification = active_board_message_notification

            fused_prob, fused_label, vision_w, radar_w = fuse_probs(
                vision_prob,
                vision_time,
                radar_prob,
                radar_time,
                now,
                args.radar_stale_sec,
                args.vision_stale_sec,
                args.dynamic_radar_weight,
                args.static_radar_weight,
            )

            if args.alarm_url:
                if args.status_post_sec > 0 and now - last_status_post_time >= args.status_post_sec:
                    status_post_elapsed = now - last_status_post_time if last_status_post_time > 0 else 0.0
                    if detection_enabled and fused_label is not None and fused_prob is not None:
                        status_label = display_label_for_result(fused_label, vision_label, vision_w, mode) or fused_label
                        status_conf = float(fused_prob[class_index(fused_label)])
                    else:
                        status_label = ""
                        status_conf = 0.0
                    preview_image = None
                    preview_encode_ms = None
                    preview_elapsed = now - last_status_image_time if last_status_image_time > 0 else 0.0
                    if (
                        args.status_image_sec > 0
                        and frame is not None
                        and now - last_status_image_time >= args.status_image_sec
                    ):
                        encode_start = time.perf_counter()
                        preview_image = encode_alarm_image(frame, args.status_image_width, args.status_jpeg_quality)
                        preview_encode_ms = (time.perf_counter() - encode_start) * 1000.0
                        if preview_image:
                            status_preview_seq += 1
                            last_status_image_time = now
                            print(
                                f"[status] preview encoded "
                                f"seq={status_preview_seq} enabled={detection_enabled} mode={mode} "
                                f"post_elapsed_sec={status_post_elapsed:.2f} "
                                f"interval_sec={preview_elapsed:.2f} target_sec={args.status_image_sec} "
                                f"encode_ms={preview_encode_ms:.1f} "
                                f"frame_shape={getattr(frame, 'shape', None)} "
                                f"width={args.status_image_width} quality={args.status_jpeg_quality} "
                                f"chars={len(preview_image)}",
                                flush=True,
                            )
                        else:
                            print(
                                f"[status] preview encode failed "
                                f"enabled={detection_enabled} mode={mode} "
                                f"post_elapsed_sec={status_post_elapsed:.2f} "
                                f"interval_sec={preview_elapsed:.2f} target_sec={args.status_image_sec} "
                                f"encode_ms={preview_encode_ms:.1f} "
                                f"frame_shape={getattr(frame, 'shape', None)} "
                                f"width={args.status_image_width} quality={args.status_jpeg_quality}",
                                flush=True,
                            )
                    elif args.status_image_sec > 0 and frame is None and now - last_status_image_time >= args.status_image_sec:
                        print(
                            f"[status] preview due but no frame "
                            f"enabled={detection_enabled} mode={mode} "
                            f"post_elapsed_sec={status_post_elapsed:.2f} "
                            f"interval_sec={preview_elapsed:.2f} target_sec={args.status_image_sec}",
                            flush=True,
                        )
                    post_status_async(
                        args.alarm_url,
                        make_status_payload(
                            status_label,
                            status_conf,
                            preview_image,
                            reminders_snapshot,
                            reminder_notification,
                            reminder_request_id,
                            system_command_request_id,
                            board_message_request_id,
                            board_message_notification,
                            control_request_id,
                            detection_enabled,
                            mode,
                            debug_vision_label,
                            debug_radar_label,
                            status_preview_seq if preview_image else None,
                            preview_encode_ms,
                            status_post_elapsed,
                            preview_elapsed,
                        ),
                        args.alarm_timeout,
                    )
                    last_status_post_time = now

            if not detection_enabled:
                if args.alarm_url and fall_alarm_active:
                    post_alarm_async(
                        args.alarm_url,
                        make_alarm_payload(False, 0.0, ""),
                        args.alarm_timeout,
                    )
                    fall_alarm_active = False
                pending_fall_alarm_timestamp = 0
                pending_fall_alarm_time = 0.0
                pending_fall_followup_started = False
                fall_followup_cancel_event.clear()

            if args.alarm_url and detection_enabled and fused_label is not None and fused_prob is not None:
                fall_conf = float(fused_prob[class_index("fall")])
                if fused_label == "fall" and fall_conf >= args.alarm_conf and not fall_alarm_active:
                    if now - last_fall_alarm_time >= args.alarm_cooldown_sec:
                        alarm_image = encode_alarm_image(frame, args.alarm_image_width, args.alarm_jpeg_quality) if args.alarm_image else None
                        alarm_payload = make_alarm_payload(True, fall_conf, fused_label, alarm_image)
                        post_alarm_async(
                            args.alarm_url,
                            alarm_payload,
                            args.alarm_timeout,
                        )
                        last_fall_alarm_time = now
                        pending_fall_alarm_timestamp = int(alarm_payload["timestamp"])
                        pending_fall_alarm_time = now
                        pending_fall_followup_started = False
                    else:
                        print(
                            f"[alarm] suppressed by cooldown "
                            f"({now - last_fall_alarm_time:.1f}s < {args.alarm_cooldown_sec:.1f}s)",
                            flush=True,
                        )
                    fall_alarm_active = True
                elif fused_label != "fall" and fall_alarm_active:
                    post_alarm_async(
                        args.alarm_url,
                        make_alarm_payload(False, 0.0, fused_label),
                        args.alarm_timeout,
                    )
                    fall_alarm_active = False

            if fall_followup_cancel_event.is_set() and not pending_fall_alarm_timestamp:
                fall_followup_cancel_event.clear()

            if pending_fall_alarm_timestamp:
                if alarm_ack_timestamp >= pending_fall_alarm_timestamp:
                    print(
                        f"[alarm] acknowledged alarm_timestamp={pending_fall_alarm_timestamp} "
                        f"ack_timestamp={alarm_ack_timestamp}",
                        flush=True,
                    )
                    pending_fall_alarm_timestamp = 0
                    pending_fall_alarm_time = 0.0
                    pending_fall_followup_started = False
                elif fall_followup_cancel_event.is_set():
                    fall_followup_cancel_event.clear()
                    print(
                        f"[alarm] fall AI takeover treated as completed "
                        f"alarm_timestamp={pending_fall_alarm_timestamp}",
                        flush=True,
                    )
                    pending_fall_alarm_timestamp = 0
                    pending_fall_alarm_time = 0.0
                    pending_fall_followup_started = False
                elif (
                    not pending_fall_followup_started
                    and now - pending_fall_alarm_time >= args.fall_ack_timeout_sec
                ):
                    pending_fall_followup_started = True
                    print(
                        f"[alarm] no ack within {args.fall_ack_timeout_sec:.1f}s, starting AI follow-up",
                        flush=True,
                    )
                    if ai_assistant is not None:
                        ai_assistant.start_fall_followup(
                            args.fall_followup_question,
                            args.fall_followup_record_sec,
                        )
                    else:
                        print("[alarm] AI follow-up skipped because AI assistant is disabled", flush=True)

            display = make_display(
                frame,
                radar_img,
                fused_prob,
                fused_label,
                vision_label,
                radar_label,
                vision_w,
                radar_w,
                now - vision_time if vision_time > 0 else 999.0,
                now - radar_time if radar_time > 0 else 999.0,
                tracking_enabled,
                tracking_motor_speed,
                tracking_error_px,
                tracking_target_x,
                errors,
                mode,
                detection_enabled,
                board_ip,
                ai_assistant.snapshot() if ai_assistant is not None else None,
                reminder_notification,
                board_message_notification,
            )
            if writer is not None:
                writer.write(display)
            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                state.stop_event.set()
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        state.stop_event.set()
    finally:
        if writer is not None:
            writer.release()
        if ai_assistant is not None:
            ai_assistant.close()
        cv2.destroyAllWindows()
        vt.join(timeout=2.0)
        rt.join(timeout=2.0)
        if ct is not None:
            ct.join(timeout=2.0)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3588 vision + radar posture fusion.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--output-fps", type=float, default=20.0)
    parser.add_argument("--app-host", default="",
                        help="App/server host for both alarm and control, e.g. your PC IP on the same LAN.")
    parser.add_argument("--app-port", type=int, default=APP_STATUS_PORT)
    parser.add_argument("--alarm-url", default=ALARM_URL,
                        help="POST fall alarm to this /status URL. Set empty string to disable.")
    parser.add_argument("--alarm-conf", type=float, default=0.0,
                        help="Minimum fused fall confidence needed to send an alarm.")
    parser.add_argument("--alarm-cooldown-sec", type=float, default=40.0,
                        help="Do not send another fall alarm within this many seconds after the last fall alarm.")
    parser.add_argument("--fall-ack-timeout-sec", type=float, default=15.0,
                        help="Start AI follow-up if the phone does not acknowledge a fall alarm within this many seconds.")
    parser.add_argument("--fall-followup-record-sec", type=float, default=15.0,
                        help="Seconds to record the elder's answer after the AI fall follow-up question.")
    parser.add_argument("--fall-followup-question", default="您好，我是语音助手，您现在的状态如何",
                        help="Question spoken by the AI assistant when a fall alarm is not acknowledged.")
    parser.add_argument("--alarm-timeout", type=float, default=1.0)
    parser.add_argument("--status-post-sec", type=float, default=STATUS_POST_SEC,
                        help="Seconds between live posture label POSTs to /status. Set 0 to disable.")
    parser.add_argument("--status-image-sec", type=float, default=STATUS_IMAGE_SEC,
                        help="Seconds between live preview image POSTs to /status. Set 0 to disable.")
    parser.add_argument("--status-image-width", type=int, default=STATUS_IMAGE_WIDTH)
    parser.add_argument("--status-jpeg-quality", type=int, default=STATUS_JPEG_QUALITY)
    parser.add_argument("--reminders-file", default=str(BASE_DIR / "reminders.json"),
                        help="Local JSON file used to persist reminder queue.")
    parser.add_argument("--reminder-popup-sec", type=float, default=REMINDER_POPUP_SEC,
                        help="Seconds to keep reminder notification on the board screen.")
    parser.add_argument("--reminder-due-grace-sec", type=float, default=REMINDER_DUE_GRACE_SEC,
                        help="Seconds after scheduled time during which a reminder can still fire.")
    parser.add_argument("--alarm-image", action=argparse.BooleanOptionalAction, default=True,
                        help="Send current vision frame as JPEG base64 with fall alarm.")
    parser.add_argument("--alarm-image-width", type=int, default=640)
    parser.add_argument("--alarm-jpeg-quality", type=int, default=75)
    parser.add_argument("--control-url", default="",
                        help="GET this /status URL for enabled/mode control. Empty uses --app-host or --alarm-url.")
    parser.add_argument("--control-poll-sec", type=float, default=0.5)
    parser.add_argument("--control-timeout", type=float, default=1.0)
    parser.add_argument("--tracking", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable camera tracking at startup. GUI control status can override this.")
    parser.add_argument("--motor-port", default=MOTOR_PORT)
    parser.add_argument("--motor-baudrate", type=int, default=MOTOR_BAUDRATE)
    parser.add_argument("--motor-init-zero", action=argparse.BooleanOptionalAction, default=True,
                        help="Send 01 93 88 01 6B at startup to use the current motor position as zero.")
    parser.add_argument("--motor-init-zero-delay", type=float, default=MOTOR_INIT_ZERO_DELAY_SEC)
    parser.add_argument("--motor-location-settle-sec", type=float, default=MOTOR_LOCATION_SETTLE_SEC,
                        help="Seconds to wait between the home command and target angle command.")
    parser.add_argument("--motor-angle-min", type=float, default=MOTOR_ANGLE_MIN)
    parser.add_argument("--motor-angle-max", type=float, default=MOTOR_ANGLE_MAX)
    parser.add_argument("--motor-recv", action=argparse.BooleanOptionalAction, default=False,
                        help="Deprecated; motor serial responses are no longer read in real-time tracking.")
    parser.add_argument("--track-kp", type=float, default=TRACK_KP)
    parser.add_argument("--track-ki", type=float, default=TRACK_KI)
    parser.add_argument("--track-kd", type=float, default=TRACK_KD)
    parser.add_argument("--track-max-speed", type=float, default=TRACK_MAX_SPEED)
    parser.add_argument("--track-deadband-px", type=float, default=TRACK_DEADBAND_PX)
    parser.add_argument("--track-send-interval", type=float, default=TRACK_SEND_INTERVAL)

    parser.add_argument("--ai-assistant", action=argparse.BooleanOptionalAction, default=True,
                        help="Show the on-screen AI voice assistant record button.")
    parser.add_argument("--ai-work-dir", default=str(BASE_DIR / "ai_voice"))
    parser.add_argument("--audio-control-file", default=DEFAULT_AUDIO_CONTROL_FILE,
                        help="Shared audio priority state file used by AI assistant and voice intercom.")
    parser.add_argument("--ai-arecord-bin", default="arecord")
    parser.add_argument("--ai-capture-device", default="plughw:rockchipnau8822,0")
    parser.add_argument("--ai-capture-format", default="S16_LE")
    parser.add_argument("--ai-capture-rate", type=int, default=16000)
    parser.add_argument("--ai-capture-channels", type=int, default=1)
    parser.add_argument("--ai-capture-release-wait-sec", type=float, default=0.35,
                        help="Seconds to wait after reserving the board microphone before AI arecord starts.")
    parser.add_argument("--ai-gst-launch-bin", default="gst-launch-1.0")
    parser.add_argument("--ai-play-device", default="plughw:1,0")
    parser.add_argument("--ai-tts-voice", default="zh-CN-XiaoxiaoNeural")
    parser.add_argument("--ai-tts-force-ipv4", action=argparse.BooleanOptionalAction, default=True,
                        help="Force edge-tts DNS resolution to IPv4. Useful when IPv6 to speech.platform.bing.com resets.")
    parser.add_argument("--ai-persistent-history", action=argparse.BooleanOptionalAction, default=False,
                        help="Persist AI conversation history to history.json. Default is off, so restart clears context.")
    parser.add_argument("--ai-history-rounds", type=int, default=8)
    parser.add_argument("--ai-control-intents", action=argparse.BooleanOptionalAction, default=True,
                        help="Let DeepSeek parse safe system-control intents before normal chat.")
    parser.add_argument("--ai-control-url", default="",
                        help="POST/GET safe AI control actions to this /status URL. Empty uses --control-url.")
    parser.add_argument("--ai-control-timeout", type=float, default=1.0)
    parser.add_argument("--ai-control-confidence", type=float, default=0.75)
    parser.add_argument("--ai-allow-shutdown", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow explicit AI voice commands to power off the development board.")
    parser.add_argument("--ai-shutdown-delay-sec", type=float, default=0.0,
                        help="Delay before executing poweroff after an accepted AI shutdown command.")
    parser.add_argument("--ai-web-search", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow AI chat to use lightweight Zhipu web search for fresh external facts.")
    parser.add_argument("--zhipu-api-key", default="",
                        help="Zhipu API key for AI web search. Empty reads ZHIPU_API_KEY from environment.")
    parser.add_argument("--zhipu-search-url", default="https://open.bigmodel.cn/api/paas/v4/web_search")
    parser.add_argument("--zhipu-reader-url", default="https://open.bigmodel.cn/api/paas/v4/reader")
    parser.add_argument("--ai-web-search-engines", default="search_pro,search_pro_sogou,search_pro_quark")
    parser.add_argument("--ai-web-search-count", type=int, default=5)
    parser.add_argument("--ai-web-content-size", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--ai-web-max-sources", type=int, default=10)
    parser.add_argument("--ai-web-max-snippet-chars", type=int, default=700)
    parser.add_argument("--ai-web-allow-reader", action=argparse.BooleanOptionalAction, default=True,
                        help="Allow at most one Zhipu reader call when search snippets are insufficient.")
    parser.add_argument("--ai-web-max-reader-chars", type=int, default=2200)
    parser.add_argument("--ai-web-max-parallel-searches", type=int, default=3)
    parser.add_argument("--ai-web-search-timeout", type=float, default=10.0)
    parser.add_argument("--ai-web-reader-timeout", type=float, default=10.0)
    parser.add_argument("--ai-web-reader-page-timeout", type=float, default=8.0)
    parser.add_argument("--ai-web-deepseek-timeout", type=float, default=20.0)
    parser.add_argument("--ai-web-answer-tokens", type=int, default=900)
    parser.add_argument("--deepseek-api-key", default="",
                        help="DeepSeek API key. Empty reads DEEPSEEK_API_KEY from environment.")
    parser.add_argument("--deepseek-base-url", default="https://api.deepseek.com")
    parser.add_argument("--deepseek-model", default="deepseek-v4-flash")
    parser.add_argument("--tencent-secret-id", default="",
                        help="Tencent Cloud SecretId. Empty reads TENCENTCLOUD_SECRET_ID from environment.")
    parser.add_argument("--tencent-secret-key", default="",
                        help="Tencent Cloud SecretKey. Empty reads TENCENTCLOUD_SECRET_KEY from environment.")
    parser.add_argument("--tencent-region", default="ap-guangzhou")
    parser.add_argument("--tencent-asr-service", default="16k_zh")

    parser.add_argument("--yolo-model", default=str(BASE_DIR / "models" / "yolo11.rknn"))
    parser.add_argument("--vision-model", default=str(BASE_DIR / "models" / "posture_classifier_5class.rknn"))
    parser.add_argument("--radar-model", default=str(BASE_DIR / "models" / "radar_nn1_4class_v2.rknn"))
    parser.add_argument("--yolo-npu-core", choices=["0", "1", "2", "all"], default="0")
    parser.add_argument("--vision-npu-core", choices=["0", "1", "2", "all"], default="1")
    parser.add_argument("--radar-npu-core", choices=["0", "1", "2", "all"], default="2")

    parser.add_argument("--yolo-conf", type=float, default=YOLO_OBJ_THRESH)
    parser.add_argument("--yolo-nms", type=float, default=YOLO_NMS_THRESH)
    parser.add_argument("--vision-conf", type=float, default=0.0)
    parser.add_argument("--vision-temperature", type=float, default=VISION_TEMPERATURE,
                        help="Soften overconfident vision probabilities. 1.0 keeps original probabilities.")
    parser.add_argument("--uncertainty-threshold", type=float, default=1.0)
    parser.add_argument("--no-speed", action="store_true")
    parser.add_argument("--speed-threshold", type=float, default=SPEED_THRESHOLD)
    parser.add_argument("--speed-window", type=int, default=SPEED_WINDOW)
    parser.add_argument("--speed-motor-comp-k", type=float, default=SPEED_MOTOR_COMP_K,
                        help="Compensate stand/walk speed by subtracting k*motor_speed/person_height from horizontal motion. Negative values flip direction.")
    parser.add_argument("--vision-filter-window", type=int, default=VISION_FILTER_WINDOW,
                        help="Deprecated; kept for compatibility. Use --vision-lying-stable-frames.")
    parser.add_argument("--vision-lying-stable-frames", type=int, default=VISION_LYING_STABLE_FRAMES,
                        help="Consecutive vision frames required before lying/other posture state changes.")
    parser.add_argument("--vision-lying-fall-hold-sec", type=float, default=VISION_LYING_FALL_HOLD_SEC,
                        help="Seconds to keep vision lying as fall before settling to lying.")
    parser.add_argument("--vision-stale-sec", type=float, default=1.0)

    parser.add_argument("--radar-ip", default=RADAR_IP)
    parser.add_argument("--radar-input-format", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--radar-temperature", type=float, default=RADAR_TEMPERATURE,
                        help="Soften radar probabilities. 1.0 keeps original probabilities.")
    parser.add_argument("--dynamic-radar-weight", type=float, default=DYNAMIC_RADAR_WEIGHT,
                        help="Radar fusion weight for fall/walk/bend.")
    parser.add_argument("--static-radar-weight", type=float, default=STATIC_RADAR_WEIGHT,
                        help="Radar fusion weight for stand/lying/sit.")
    parser.add_argument("--radar-weight-start", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--radar-weight-end", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--radar-stale-sec", type=float, default=2.5)
    parser.add_argument("--scan-start-ps", type=int, default=SCAN_START_PS)
    parser.add_argument("--scan-stop-ps", type=int, default=SCAN_STOP_PS)
    parser.add_argument("--interval-us", type=int, default=INTERVAL_US)
    parser.add_argument("--window-scans", type=int, default=WINDOW_SCANS)
    parser.add_argument("--distance-min", type=float, default=DISTANCE_MIN_M)
    parser.add_argument("--distance-max", type=float, default=DISTANCE_MAX_M)
    parser.add_argument("--cable-compensation", type=float, default=CABLE_COMPENSATION_M)
    parser.add_argument("--mti-alpha", type=float, default=DEFAULT_MTI_ALPHA)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
