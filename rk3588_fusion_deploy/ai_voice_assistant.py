from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from audio_priority import (
    DEFAULT_AUDIO_CONTROL_FILE,
    clear_audio_owner,
    is_audio_owner_active,
    mark_audio_owner,
)


SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "你是一个运行在老人跌倒检测系统中的中文语音助手。"
        "用户的话来自语音识别，可能有错别字。"
        "请结合之前的对话上下文回答。"
        "你的能力限制：你只能通过语音向用户回答。"
        "你不能直接打电话、不能直接联系家属、不能直接呼叫急救、不能保证报警已经被他人看到。"
        "不要说“我会帮你联系家人”“我已经帮你打电话”“我马上通知医生”等你无法真实完成的事情。"
        "如果系统已经发送过摔倒报警，只能说“系统已经发送报警信息到手机端”，并提醒用户不能完全依赖这个结果。"
        "当用户表示摔倒、受伤、不舒服、无法起身、头晕、胸闷、流血、疼痛或意识不清时，"
        "建议用户先不要强行起身，保持安全姿势，大声呼救，让身边人帮助。"
        "如果情况严重，明确建议立即拨打120，或让身边人、家属拨打120。"
        "可以提醒用户使用手机、小程序或身边设备联系家人，但不要声称你已经替他完成。"
        "当用户表示没事时，提醒用户慢慢活动，确认没有头晕、疼痛或不适，并建议继续观察。"
        "请用自然、简洁、适合朗读的中文回答。"
        "不要使用 Markdown 表格。"
        "不要使用太多特殊符号。"
        "每次回复控制在2到4句话。"
    ),
}

DEFAULT_DEEPSEEK_API_KEY = ""
DEFAULT_TENCENTCLOUD_SECRET_ID = ""
DEFAULT_TENCENTCLOUD_SECRET_KEY = ""

FALL_FOLLOWUP_CONTEXT = (
    "系统刚刚检测到老人可能摔倒，并且手机端15秒内没有确认收到报警。"
    "请根据老人回答判断是否需要帮助，语气要简短、关心、适合语音播报。"
    "注意：你不能替用户打电话或联系家属，只能建议用户、提醒用户，并说明系统能力边界。"
    "如果老人表示受伤、不能起身、头晕或需要帮助，请建议立即拨打120，或让身边人、家属拨打120。"
    "如果老人表示没事，请提醒慢慢起身并注意安全。"
)

CONTROL_INTENT_PROMPT = {
    "role": "system",
    "content": (
        "你是一个老人跌倒检测系统的语音控制意图解析器。"
        "你的任务是把用户中文语音识别文本转换成严格 JSON，不要输出解释。"
        "如果用户不是明确要控制或查询系统，输出 {\"intent\":\"chat\",\"confidence\":1.0}。"
        "只允许这些 intent: chat, set_tracking, set_mode, get_status, get_ip。"
        "set_tracking 必须包含 enabled，true 表示打开自动跟踪，false 表示关闭自动跟踪。"
        "set_mode 必须包含 mode，vision 表示视觉/日常模式，radar 表示隐私/雷达模式。"
        "get_status 表示查询检测状态、跟踪状态、模式、姿态等。"
        "get_ip 表示查询开发板 IP 地址。"
        "只有语义明确时才输出控制 intent；否则输出 chat。"
        "confidence 范围是 0 到 1。"
    ),
}

CONTROL_INTENTS = {
    "set_tracking",
    "set_mode",
    "get_status",
    "get_ip",
    "set_detection",
    "set_motor_angle",
    "move_motor_relative",
    "set_motor_zero",
    "shutdown_board",
    "reboot_board",
    "send_message",
    "clear_messages",
    "add_reminder",
    "update_reminder",
    "delete_reminder",
    "list_reminders",
}
MODE_NAMES = {"vision": "视觉模式", "radar": "隐私模式"}

CONTROL_PLAN_PROMPT = {
    "role": "system",
    "content": (
        "You are a safe intent parser for a Chinese elder-care fall detection system. "
        "Return strict JSON only, with no markdown and no explanation. "
        "Use this schema: "
        "{\"actions\":[{\"intent\":\"set_tracking|set_mode|get_status|get_ip|set_detection|set_motor_angle|move_motor_relative|set_motor_zero|shutdown_board|reboot_board|send_message|clear_messages|add_reminder|update_reminder|delete_reminder|list_reminders\","
        "\"enabled\":true,\"mode\":\"vision|radar\",\"angle\":0.0,\"message\":\"\","
        "\"reminderId\":\"\",\"targetLabel\":\"\",\"reminder\":{\"label\":\"\",\"repeat\":\"once|weekly\",\"date\":\"YYYY-MM-DD\",\"time\":\"HH:MM\",\"weekdays\":[1,3,5],\"enabled\":true},"
        "\"confidence\":0.0}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\","
        "\"search_recency\":\"oneDay|oneWeek|oneMonth|oneYear|noLimit\","
        "\"answer_style\":\"short|summary\",\"direct_answer\":\"\",\"confidence\":0.0}. "
        "The actions array may contain multiple actions in the original order. "
        "Allowed actions are set_tracking, set_mode, set_detection, get_status, get_ip. "
        "Also allowed are set_motor_angle, move_motor_relative, set_motor_zero, shutdown_board, reboot_board, send_message, clear_messages. "
        "Also allowed are add_reminder, update_reminder, delete_reminder, list_reminders. "
        "For set_tracking, enabled=true means turn on tracking and enabled=false means turn it off. "
        "For set_detection, enabled=true means turn on fall detection and enabled=false means turn it off. "
        "For set_mode, mode=vision means visual/daily mode and mode=radar means privacy/radar mode. "
        "For set_motor_angle, angle is the absolute target gimbal angle in degrees. "
        "For move_motor_relative, angle is the relative rotation in degrees; positive and negative values are allowed. "
        "Counterclockwise rotation is negative, and clockwise rotation is positive. "
        "For set_motor_zero, set the current gimbal position as zero; no angle is required. "
        "For shutdown_board, only output it when the user clearly asks to shut down or power off the development board. "
        "For reboot_board, only output it when the user clearly asks to restart or reboot the development board. "
        "For send_message, extract the exact message to send to the app into message. Examples include 请帮我发送信息：..., 帮我发送一条...的信息, 给App发消息说.... "
        "For clear_messages, use it when the user asks to clear, delete, or empty the app/chat/two-way communication message history. "
        "For add_reminder, fill reminder.label and reminder.time. Use repeat=once with date=YYYY-MM-DD for one-time reminders. "
        "Use repeat=weekly with weekdays using ISO weekdays 1=Monday ... 7=Sunday for weekly alarm-clock style reminders. "
        "Resolve relative reminder times such as later, in half an hour, tonight, tomorrow morning, or next Monday using the Current date context. "
        "For update_reminder and delete_reminder, fill reminderId if known, otherwise fill targetLabel with the reminder label mentioned by the user. "
        "For update_reminder, fill only the fields that should change in reminder. "
        "For list_reminders, no reminder fields are required. "
        "If the user also asks a normal question, put only the non-control question in chat_text. "
        "If there is no control action, actions must be an empty array and chat_text should contain the original request. "
        "For chat_text, set chat_need_web=true only for fresh external facts, news, weather, prices, exchange rates, "
        "schedules, public policies, or other information likely to have changed. "
        "If local date/time context is enough, set chat_need_web=false and fill direct_answer. "
        "When chat_need_web=true, create exactly one concise search_query and resolve relative dates such as today, "
        "tomorrow, yesterday, latest, or recent into concrete dates when useful. "
        "Only extract a control action when the user's meaning is clear. "
        "Examples: "
        "打开隐私模式并且关闭跟踪 -> "
        "{\"actions\":[{\"intent\":\"set_mode\",\"mode\":\"radar\",\"confidence\":0.95},"
        "{\"intent\":\"set_tracking\",\"enabled\":false,\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "推荐糖尿病饮食，顺便关闭跟踪 -> "
        "{\"actions\":[{\"intent\":\"set_tracking\",\"enabled\":false,\"confidence\":0.95}],"
        "\"chat_text\":\"推荐一些糖尿病患者的饮食\","
        "\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "电机旋转到30度 -> "
        "{\"actions\":[{\"intent\":\"set_motor_angle\",\"angle\":30,\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "电机逆时针相对旋转15度 -> "
        "{\"actions\":[{\"intent\":\"move_motor_relative\",\"angle\":-15,\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "电机顺时针相对旋转15度 -> "
        "{\"actions\":[{\"intent\":\"move_motor_relative\",\"angle\":15,\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "把当前位置设为零点 -> "
        "{\"actions\":[{\"intent\":\"set_motor_zero\",\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "关闭开发板 -> "
        "{\"actions\":[{\"intent\":\"shutdown_board\",\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "明天早上八点提醒我吃药 -> "
        "{\"actions\":[{\"intent\":\"add_reminder\",\"reminder\":{\"label\":\"吃药\",\"repeat\":\"once\",\"date\":\"2026-07-19\",\"time\":\"08:00\",\"weekdays\":[],\"enabled\":true},\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "每周一三五晚上八点提醒我锻炼 -> "
        "{\"actions\":[{\"intent\":\"add_reminder\",\"reminder\":{\"label\":\"锻炼\",\"repeat\":\"weekly\",\"date\":\"\",\"time\":\"20:00\",\"weekdays\":[1,3,5],\"enabled\":true},\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "把吃药提醒改到九点 -> "
        "{\"actions\":[{\"intent\":\"update_reminder\",\"targetLabel\":\"吃药\",\"reminder\":{\"time\":\"09:00\"},\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}. "
        "删除吃药提醒 -> "
        "{\"actions\":[{\"intent\":\"delete_reminder\",\"targetLabel\":\"吃药\",\"confidence\":0.95}],"
        "\"chat_text\":\"\",\"chat_need_web\":false,\"search_query\":\"\",\"direct_answer\":\"\",\"confidence\":0.95}."
    ),
}


@contextlib.contextmanager
def force_ipv4_dns(enabled: bool):
    if not enabled:
        yield
        return

    original_getaddrinfo = socket.getaddrinfo

    def getaddrinfo_ipv4(host, port, family=0, type=0, proto=0, flags=0):
        if family in (0, socket.AF_UNSPEC):
            family = socket.AF_INET
        return original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = getaddrinfo_ipv4
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


class AIVoiceAssistant:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.work_dir = Path(args.ai_work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.persistent_history = bool(getattr(args, "ai_persistent_history", False))
        self.history_file = self.work_dir / "history.json"
        self.input_audio = self.work_dir / "input.wav"
        self.reply_audio = self.work_dir / "reply.mp3"
        self.lock = threading.Lock()
        self.state = "idle"
        self.status = "Ready"
        self.last_error = ""
        self.last_asr = ""
        self.last_reply = ""
        self.record_proc: subprocess.Popen | None = None
        self.play_proc: subprocess.Popen | None = None
        self.worker: threading.Thread | None = None
        self.session_id = 0
        self.current_cancel_event: threading.Event | None = None
        self.current_kind = "idle"
        self.record_session_id = 0
        self.record_cancel_event: threading.Event | None = None
        self.messages = self.load_history()

    def deepseek_api_key(self) -> str:
        return self.args.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "") or DEFAULT_DEEPSEEK_API_KEY

    def control_url(self) -> str:
        return str(
            getattr(self.args, "ai_control_url", "")
            or getattr(self.args, "control_url", "")
            or "http://127.0.0.1:8889/status"
        )

    def control_timeout(self) -> float:
        return max(0.2, float(getattr(self.args, "ai_control_timeout", 1.0)))

    def get_board_ip(self) -> str:
        ips: list[str] = []
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                ips.append(sock.getsockname()[0])
        except OSError:
            pass

        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                ips.append(info[4][0])
        except OSError:
            pass

        for ip in ips:
            if ip and not ip.startswith("127.") and ip != "0.0.0.0":
                return ip
        return "-"

    def date_context(self) -> str:
        now = datetime.now()
        today = now.date()
        return (
            f"current_datetime={now.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"today={today.isoformat()}, "
            f"yesterday={(today - timedelta(days=1)).isoformat()}, "
            f"tomorrow={(today + timedelta(days=1)).isoformat()}, "
            "timezone=Asia/Shanghai"
        )

    def finish_cancelled_session(self, session_id: int, cancel_event: threading.Event, status: str = "Interrupted") -> None:
        self.finish_session(session_id, cancel_event, status)

    def fail_session(self, session_id: int, cancel_event: threading.Event, status: str, error: str) -> None:
        with self.lock:
            if session_id != self.session_id or self.current_cancel_event is not cancel_event:
                return
            self.current_cancel_event = None
            self.current_kind = "idle"
            self.state = "idle"
            self.status = "Ready"
            self.last_error = error
        clear_audio_owner(self.audio_control_file(), "ai_record")
        print(f"[ai] state=idle status={status} error={error}", flush=True)

    def snapshot(self) -> dict:
        self.cancel_if_phone_audio()
        with self.lock:
            state = self.state
        if state == "recording":
            self.mark_ai_recording(1.0)
        with self.lock:
            fall_takeover_active = self.current_kind == "fall" and self.state in ("recording", "processing", "speaking")
            return {
                "state": self.state,
                "status": self.status,
                "last_error": self.last_error,
                "last_asr": self.last_asr,
                "last_reply": self.last_reply,
                "kind": self.current_kind,
                "fall_takeover_active": fall_takeover_active,
                "enabled": self.state not in ("processing", "speaking"),
            }

    def set_state(self, state: str, status: str, error: str = "") -> None:
        with self.lock:
            self.state = state
            self.status = status
            if error:
                self.last_error = error
        print(f"[ai] state={state} status={status}{' error=' + error if error else ''}", flush=True)

    def audio_control_file(self) -> str:
        return str(getattr(self.args, "audio_control_file", DEFAULT_AUDIO_CONTROL_FILE))

    def phone_audio_active(self) -> bool:
        return is_audio_owner_active(self.audio_control_file(), "phone_speaker")

    def new_session(self, kind: str) -> tuple[int, threading.Event]:
        with self.lock:
            self.session_id += 1
            cancel_event = threading.Event()
            self.current_cancel_event = cancel_event
            self.current_kind = kind
            return self.session_id, cancel_event

    def is_cancelled(self, session_id: int, cancel_event: threading.Event) -> bool:
        if cancel_event.is_set() or self.phone_audio_active():
            return True
        with self.lock:
            return session_id != self.session_id

    def terminate_proc(self, proc: subprocess.Popen | None, sigint: bool = False) -> None:
        if proc is None or proc.poll() is not None:
            return
        try:
            if sigint:
                proc.send_signal(signal.SIGINT)
            else:
                proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def cancel_current(self, reason: str, status: str = "Interrupted") -> None:
        with self.lock:
            cancel_event = self.current_cancel_event
            self.session_id += 1
            self.current_cancel_event = None
            self.current_kind = "idle"
            self.state = "idle"
            self.status = status
            self.last_error = ""
            record_proc = self.record_proc
            play_proc = self.play_proc
            self.record_proc = None
            self.play_proc = None
        if cancel_event is not None:
            cancel_event.set()
        self.terminate_proc(record_proc, sigint=True)
        self.terminate_proc(play_proc)
        clear_audio_owner(self.audio_control_file(), "ai_record")
        print(f"[ai] cancelled current session: {reason}", flush=True)

    def cancel_fall_followup(self) -> bool:
        with self.lock:
            active = self.current_kind == "fall" and self.state in ("recording", "processing", "speaking", "error")
        if not active:
            return False
        self.cancel_current("fall follow-up cancelled by screen button", "Ready")
        return True

    def finish_session(self, session_id: int, cancel_event: threading.Event, status: str = "Ready") -> None:
        with self.lock:
            if session_id != self.session_id or self.current_cancel_event is not cancel_event:
                return
            self.current_cancel_event = None
            self.current_kind = "idle"
            self.state = "idle"
            self.status = status
            self.last_error = ""

    def cancel_if_phone_audio(self) -> None:
        if not self.phone_audio_active():
            return
        with self.lock:
            busy = self.state in ("recording", "processing", "speaking")
        if busy:
            self.cancel_current("phone voice has higher priority", "Interrupted by phone")

    def mark_ai_recording(self, hold_sec: float = 1.0) -> None:
        mark_audio_owner(self.audio_control_file(), "ai_record", "ai recording", hold_sec)

    def reserve_microphone_for_recording(self, hold_sec: float = 3.0) -> None:
        self.mark_ai_recording(hold_sec)
        wait_sec = max(0.0, float(getattr(self.args, "ai_capture_release_wait_sec", 0.35)))
        if wait_sec > 0:
            time.sleep(wait_sec)

    def toggle_recording(self) -> None:
        self.cancel_if_phone_audio()
        with self.lock:
            state = self.state
        if state in ("processing", "speaking"):
            print("[ai] ignored click while busy", flush=True)
            return
        if state == "recording":
            self.stop_recording_and_process()
        else:
            self.start_recording()

    def start_fall_followup(self, question: str, record_seconds: float) -> bool:
        if self.phone_audio_active():
            self.cancel_current("phone voice active before fall follow-up", "Interrupted by phone")
            return True

        with self.lock:
            busy = self.state in ("recording", "processing", "speaking")
            current_kind = self.current_kind
            current_state = self.state
        if busy:
            if current_kind == "fall":
                print("[ai] fall follow-up already running", flush=True)
                return True
            print(f"[ai] fall follow-up preempts current AI state={current_state}", flush=True)
            self.cancel_current("fall follow-up has higher priority than manual AI", "Fall follow-up")

        session_id, cancel_event = self.new_session("fall")

        with self.lock:
            self.state = "speaking"
            self.status = "Asking..."
            self.last_error = ""
            self.last_asr = ""
            self.last_reply = question

        self.worker = threading.Thread(
            target=self.run_fall_followup,
            args=(session_id, cancel_event, question, record_seconds),
            daemon=True,
        )
        self.worker.start()
        return True

    def start_reminder(self, message: str) -> bool:
        text = str(message or "").strip()
        if not text:
            return False
        if self.phone_audio_active():
            print("[ai] reminder voice skipped because phone voice is active", flush=True)
            return False

        with self.lock:
            busy = self.state in ("recording", "processing", "speaking")
            current_kind = self.current_kind
            current_state = self.state
        if busy:
            if current_kind == "fall":
                print("[ai] reminder voice skipped because fall follow-up is active", flush=True)
                return False
            print(f"[ai] reminder preempts current AI state={current_state}", flush=True)
            self.cancel_current("reminder voice", "Reminder")

        session_id, cancel_event = self.new_session("reminder")
        with self.lock:
            self.state = "speaking"
            self.status = "Reminder"
            self.last_error = ""
            self.last_asr = ""
            self.last_reply = text

        self.worker = threading.Thread(
            target=self.run_reminder,
            args=(session_id, cancel_event, text),
            daemon=True,
        )
        self.worker.start()
        return True

    def run_reminder(self, session_id: int, cancel_event: threading.Event, text: str) -> None:
        try:
            print(f"[ai] reminder text: {text}", flush=True)
            asyncio.run(self.text_to_audio(text, self.reply_audio))
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event)
                return
            self.play_audio(self.reply_audio, session_id, cancel_event)
            self.finish_session(session_id, cancel_event)
        except InterruptedError as exc:
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event, str(exc) or "Interrupted")
            else:
                self.finish_session(session_id, cancel_event, str(exc) or "Interrupted")
        except Exception as exc:
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event, "Interrupted")
            else:
                self.fail_session(session_id, cancel_event, "Reminder failed", str(exc))

    def run_fall_followup(
        self,
        session_id: int,
        cancel_event: threading.Event,
        question: str,
        record_seconds: float,
    ) -> None:
        try:
            print(f"[ai] fall question text: {question}", flush=True)
            asyncio.run(self.text_to_audio(question, self.reply_audio))
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event)
                return
            self.play_audio(self.reply_audio, session_id, cancel_event)

            seconds = max(1, int(round(float(record_seconds))))
            self.set_state("processing", f"Auto recording {seconds}s...")
            self.record_for_seconds(seconds, session_id, cancel_event)
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event)
                return
            self.set_state("processing", "Recognizing...")
            self.process_recording(session_id, cancel_event, context_prefix=FALL_FOLLOWUP_CONTEXT)
        except InterruptedError as exc:
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event, str(exc) or "Interrupted")
            else:
                self.finish_session(session_id, cancel_event, str(exc) or "Interrupted")
        except Exception as exc:
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event, "Interrupted")
            else:
                self.fail_session(session_id, cancel_event, "AI failed", str(exc))

    def start_recording(self) -> None:
        if self.phone_audio_active():
            print("[ai] manual recording blocked by phone voice", flush=True)
            return
        with self.lock:
            if self.state in ("recording", "processing", "speaking"):
                return
        session_id, cancel_event = self.new_session("manual")
        with self.lock:
            self.state = "recording"
            self.status = "Recording..."
            self.last_error = ""
            self.last_asr = ""
            self.last_reply = ""

        try:
            if self.input_audio.exists():
                self.input_audio.unlink()
            self.reserve_microphone_for_recording()
            cmd = [
                self.args.ai_arecord_bin,
                "-D",
                self.args.ai_capture_device,
                "-f",
                self.args.ai_capture_format,
                "-r",
                str(self.args.ai_capture_rate),
                "-c",
                str(self.args.ai_capture_channels),
                "-t",
                "wav",
                str(self.input_audio),
            ]
            print("[ai] start recording:", " ".join(cmd), flush=True)
            self.record_proc = subprocess.Popen(cmd)
            self.record_session_id = session_id
            self.record_cancel_event = cancel_event
            self.mark_ai_recording(2.0)
        except Exception as exc:
            self.record_proc = None
            self.fail_session(session_id, cancel_event, "Record failed", str(exc))

    def record_for_seconds(self, seconds: int, session_id: int, cancel_event: threading.Event) -> None:
        if self.input_audio.exists():
            self.input_audio.unlink()
        self.reserve_microphone_for_recording(hold_sec=max(3.0, float(seconds) + 3.0))
        cmd = [
            self.args.ai_arecord_bin,
            "-D",
            self.args.ai_capture_device,
            "-f",
            self.args.ai_capture_format,
            "-r",
            str(self.args.ai_capture_rate),
            "-c",
            str(self.args.ai_capture_channels),
            "-d",
            str(seconds),
            "-t",
            "wav",
            str(self.input_audio),
        ]
        print("[ai] auto record:", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd)
        self.record_proc = proc
        end_time = time.time() + seconds + 3.0
        try:
            while proc.poll() is None:
                self.mark_ai_recording(1.0)
                if self.is_cancelled(session_id, cancel_event):
                    self.terminate_proc(proc, sigint=True)
                    raise InterruptedError("Interrupted by phone")
                if time.time() > end_time:
                    self.terminate_proc(proc, sigint=True)
                    raise RuntimeError("timed recording did not finish")
                time.sleep(0.05)
        finally:
            self.record_proc = None
            clear_audio_owner(self.audio_control_file(), "ai_record")
        if proc.returncode not in (0, None):
            raise RuntimeError(f"arecord failed with code {proc.returncode}")

    def stop_recording_and_process(self) -> None:
        session_id = getattr(self, "record_session_id", self.session_id)
        cancel_event = getattr(self, "record_cancel_event", self.current_cancel_event)
        if cancel_event is None:
            cancel_event = threading.Event()
        proc = self.record_proc
        self.record_proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.terminate()
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        clear_audio_owner(self.audio_control_file(), "ai_record")
        if self.is_cancelled(session_id, cancel_event):
            self.finish_cancelled_session(session_id, cancel_event)
            return

        self.set_state("processing", "Recognizing...")
        self.worker = threading.Thread(
            target=self.process_recording,
            args=(session_id, cancel_event),
            daemon=True,
        )
        self.worker.start()

    def process_recording(
        self,
        session_id: int,
        cancel_event: threading.Event,
        context_prefix: str = "",
    ) -> None:
        try:
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event)
                return
            if not self.input_audio.exists() or self.input_audio.stat().st_size <= 44:
                raise RuntimeError("recording file is empty")

            text = self.tencent_asr(self.input_audio)
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event)
                return
            if not text:
                raise RuntimeError("ASR returned empty text")
            with self.lock:
                self.last_asr = text
                self.status = "Thinking..."
            print(f"[ai] microphone text: {text}", flush=True)

            reply = None
            if not context_prefix:
                reply = self.try_handle_control_intent(text)
            if reply is None:
                if context_prefix:
                    reply = self.ask_deepseek(text, context_prefix=context_prefix)
                else:
                    reply = self.ask_chat_text(text)
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event)
                return
            if not reply:
                raise RuntimeError("DeepSeek returned empty reply")
            with self.lock:
                self.last_reply = reply
                self.state = "speaking"
                self.status = "Speaking..."
            print(f"[ai] reply text: {reply}", flush=True)

            asyncio.run(self.text_to_audio(reply, self.reply_audio))
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event)
                return
            self.play_audio(self.reply_audio, session_id, cancel_event)
            self.finish_session(session_id, cancel_event, "Ready")
        except InterruptedError as exc:
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event, str(exc) or "Interrupted")
            else:
                self.finish_session(session_id, cancel_event, str(exc) or "Interrupted")
        except Exception as exc:
            if self.is_cancelled(session_id, cancel_event):
                self.finish_cancelled_session(session_id, cancel_event, "Interrupted")
            else:
                self.fail_session(session_id, cancel_event, "AI failed", str(exc))

    def load_history(self) -> list[dict]:
        if not self.persistent_history:
            return [SYSTEM_PROMPT]
        if self.history_file.exists():
            try:
                messages = json.loads(self.history_file.read_text(encoding="utf-8"))
                if messages and messages[0].get("role") == "system":
                    return [SYSTEM_PROMPT] + messages[1:]
            except Exception:
                pass
        return [SYSTEM_PROMPT]

    def save_history(self) -> None:
        if not self.persistent_history:
            return
        self.history_file.write_text(json.dumps(self.messages, ensure_ascii=False, indent=2), encoding="utf-8")

    def trim_messages(self, max_rounds: int = 8) -> None:
        system_message = self.messages[0]
        history = self.messages[1:]
        self.messages = [system_message] + history[-max_rounds * 2 :]

    async def text_to_audio(self, text: str, filename: Path) -> None:
        import edge_tts

        force_ipv4 = bool(getattr(self.args, "ai_tts_force_ipv4", True))
        with force_ipv4_dns(force_ipv4):
            communicate = edge_tts.Communicate(text=text, voice=self.args.ai_tts_voice)
            await communicate.save(str(filename))

    def tencent_asr(self, audio_path: Path) -> str:
        from tencentcloud.asr.v20190614 import asr_client, models
        from tencentcloud.common import credential
        from tencentcloud.common.profile.client_profile import ClientProfile
        from tencentcloud.common.profile.http_profile import HttpProfile

        secret_id = self.args.tencent_secret_id or os.environ.get("TENCENTCLOUD_SECRET_ID", "") or DEFAULT_TENCENTCLOUD_SECRET_ID
        secret_key = self.args.tencent_secret_key or os.environ.get("TENCENTCLOUD_SECRET_KEY", "") or DEFAULT_TENCENTCLOUD_SECRET_KEY
        if not secret_id or not secret_key:
            raise RuntimeError("missing Tencent Cloud credentials")

        audio_bytes = audio_path.read_bytes()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        cred = credential.Credential(secret_id, secret_key)
        http_profile = HttpProfile()
        http_profile.endpoint = "asr.tencentcloudapi.com"
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile
        client = asr_client.AsrClient(cred, self.args.tencent_region, client_profile)

        req = models.SentenceRecognitionRequest()
        params = {
            "ProjectId": 0,
            "SubServiceType": 2,
            "EngSerViceType": self.args.tencent_asr_service,
            "SourceType": 1,
            "VoiceFormat": audio_path.suffix.lower().replace(".", ""),
            "UsrAudioKey": f"rk3588_ai_{int(time.time() * 1000)}",
            "Data": audio_base64,
            "DataLen": len(audio_bytes),
        }
        req.from_json_string(json.dumps(params))
        resp = client.SentenceRecognition(req)
        return str(resp.Result or "").strip()

    def extract_json_object(self, text: str) -> dict:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            pass

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def parse_control_intent(self, text: str) -> dict:
        from openai import OpenAI

        api_key = self.deepseek_api_key()
        if not api_key:
            raise RuntimeError("missing DeepSeek API key")

        client = OpenAI(api_key=api_key, base_url=self.args.deepseek_base_url)
        response = client.chat.completions.create(
            model=self.args.deepseek_model,
            messages=[
                CONTROL_PLAN_PROMPT,
                {
                    "role": "user",
                    "content": (
                        f"Current date context: {self.date_context()}\n"
                        "Parse this Chinese voice input into the JSON schema. Text:\n"
                        f"{text}"
                    ),
                },
            ],
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        content = str(response.choices[0].message.content or "").strip()
        data = self.extract_json_object(content)
        print(f"[ai-control] intent raw={content} parsed={data}", flush=True)
        return data

    def normalize_control_plan(self, data: dict) -> tuple[list[dict], str, float]:
        if not isinstance(data, dict):
            return [], "", 0.0

        chat_text = str(data.get("chat_text") or "").strip()
        try:
            plan_confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            plan_confidence = 0.0

        raw_actions = data.get("actions")
        actions: list[dict] = []
        if isinstance(raw_actions, list):
            actions = [item for item in raw_actions if isinstance(item, dict)]
        else:
            name = str(data.get("intent") or "chat").strip()
            if name in CONTROL_INTENTS:
                actions = [data]
            elif name == "chat" and not chat_text:
                chat_text = str(data.get("text") or "").strip()

        return actions, chat_text, plan_confidence

    def action_confidence(self, action: dict, fallback: float = 0.0) -> float:
        try:
            return float(action.get("confidence", fallback))
        except (TypeError, ValueError):
            return fallback

    def get_control_status(self) -> dict:
        url = self.control_url()
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.control_timeout()) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        return data if isinstance(data, dict) else {}

    def post_control_action(self, payload: dict) -> dict:
        url = self.control_url()
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.control_timeout()) as resp:
            response_body = resp.read().decode("utf-8")
        try:
            data = json.loads(response_body)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def schedule_poweroff(self) -> None:
        if not bool(getattr(self.args, "ai_allow_shutdown", True)):
            raise RuntimeError("AI shutdown is disabled")
        delay = max(0.0, float(getattr(self.args, "ai_shutdown_delay_sec", 0.0)))

        def worker() -> None:
            if delay > 0:
                time.sleep(delay)
            print("[ai-control] executing poweroff", flush=True)
            subprocess.Popen(["poweroff"])

        threading.Thread(target=worker, daemon=True).start()

    def schedule_reboot(self) -> None:
        if not bool(getattr(self.args, "ai_allow_shutdown", True)):
            raise RuntimeError("AI reboot is disabled")

        def worker() -> None:
            print("[ai-control] executing reboot", flush=True)
            subprocess.Popen(["reboot"])

        threading.Thread(target=worker, daemon=True).start()

    def format_status_reply(self, status: dict, include_ip: bool = True) -> str:
        enabled = "开启" if bool(status.get("enabled", False)) else "关闭"
        tracking = "开启" if bool(status.get("tracking", False)) else "关闭"
        mode = MODE_NAMES.get(str(status.get("mode") or ""), str(status.get("mode") or "未知"))
        label = str(status.get("label") or "暂无")
        confidence = status.get("confidence")
        if isinstance(confidence, (int, float)):
            label_text = f"{label}，置信度{float(confidence):.0%}"
        else:
            label_text = label
        parts = [
            f"当前检测{enabled}",
            f"模式是{mode}",
            f"自动跟踪{tracking}",
            f"当前姿态是{label_text}",
        ]
        if include_ip:
            parts.append(f"开发板IP是{self.get_board_ip()}")
        command = str(status.get("motorCommand") or "")
        command_angle = status.get("motorCommandAngle")
        if command == "absolute" and isinstance(command_angle, (int, float)):
            parts.append(f"最近电机命令是旋转到{float(command_angle):.1f}度")
        elif command == "relative" and isinstance(command_angle, (int, float)):
            parts.append(f"最近电机命令是相对旋转{float(command_angle):+.1f}度")
        elif command == "zero":
            parts.append("最近电机命令是当前位置设为零点")
        if bool(status.get("alarm", False)):
            parts.append("当前有跌倒报警")
        reminders = status.get("reminders")
        if isinstance(reminders, list) and reminders:
            parts.append(f"当前有{len(reminders)}个提醒")
        return "，".join(parts) + "。"

    def format_reminder_schedule(self, reminder: dict) -> str:
        repeat = str(reminder.get("repeat") or "once")
        label = str(reminder.get("label") or "提醒")
        time_text = str(reminder.get("time") or "")
        if repeat == "weekly":
            names = ["一", "二", "三", "四", "五", "六", "日"]
            weekdays = reminder.get("weekdays") if isinstance(reminder.get("weekdays"), list) else []
            normalized_days = []
            for day in weekdays:
                try:
                    number = int(day)
                except (TypeError, ValueError):
                    continue
                if 1 <= number <= 7:
                    normalized_days.append(names[number - 1])
            days = "".join(normalized_days)
            return f"{label}，每周{days}{time_text}"
        date_text = str(reminder.get("date") or "")
        return f"{label}，{date_text} {time_text}".strip()

    def format_reminders_reply(self, reminders: list) -> str:
        active = [item for item in reminders if isinstance(item, dict) and bool(item.get("enabled", True))]
        if not active:
            return "当前没有启用的提醒。"
        lines = [self.format_reminder_schedule(item) for item in active[:5]]
        suffix = "" if len(active) <= 5 else f"，另外还有{len(active) - 5}个提醒"
        return "当前提醒有：" + "；".join(lines) + suffix + "。"

    def reminder_payload(self, intent: dict) -> dict:
        reminder = intent.get("reminder")
        payload = reminder if isinstance(reminder, dict) else {}
        return dict(payload)

    def coerce_bool(self, value) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
        text = str(value or "").strip().lower()
        if text in {"true", "1", "yes", "on", "enable", "enabled", "open", "start"}:
            return True
        if text in {"false", "0", "no", "off", "disable", "disabled", "close", "stop"}:
            return False
        return None

    def coerce_angle(self, value) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value or "").strip()
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def normalize_mode(self, value) -> str:
        text = str(value or "").strip().lower()
        if text in {"vision", "visual", "camera", "daily", "normal"}:
            return "vision"
        if text in {"radar", "privacy", "private", "hidden"}:
            return "radar"
        return text

    def execute_control_intent(self, intent: dict) -> str | None:
        name = str(intent.get("intent") or "chat").strip()
        if name not in CONTROL_INTENTS:
            return None

        if name == "get_ip":
            return f"开发板当前IP地址是{self.get_board_ip()}。"

        if name == "get_status":
            status = self.get_control_status()
            return self.format_status_reply(status, include_ip=True)

        if name == "set_tracking":
            enabled = self.coerce_bool(intent.get("enabled"))
            if enabled is None:
                return "我没有听清楚是打开还是关闭自动跟踪，请再说一遍。"
            self.post_control_action({"action": "enable_tracking" if enabled else "disable_tracking"})
            return "已打开自动跟踪。" if enabled else "已关闭自动跟踪。"

        if name == "set_detection":
            enabled = self.coerce_bool(intent.get("enabled"))
            if enabled is None:
                return "我没有听清楚是打开还是关闭跌倒检测，请再说一遍。"
            self.post_control_action({"action": "enable_detection" if enabled else "disable_detection"})
            return "已打开跌倒检测。" if enabled else "已关闭跌倒检测。"

        if name == "set_mode":
            mode = self.normalize_mode(intent.get("mode"))
            if mode not in ("vision", "radar"):
                return "我没有听清楚要切换到视觉模式还是隐私模式，请再说一遍。"
            self.post_control_action({"action": "set_vision_mode" if mode == "vision" else "set_radar_mode"})
            return "已切换到视觉模式。" if mode == "vision" else "已切换到隐私模式。"

        if name in {"set_motor_angle", "move_motor_relative", "set_motor_zero"}:
            try:
                status = self.get_control_status()
                if bool(status.get("tracking", False)):
                    return "当前自动跟踪已开启，程序会忽略手动电机角度命令。请先关闭自动跟踪。"
            except Exception as exc:
                print(f"[ai-control] status check before motor command failed: {exc}", flush=True)

        if name == "set_motor_angle":
            angle = self.coerce_angle(intent.get("angle"))
            if angle is None:
                return "我没有听清楚要旋转到多少度，请再说一遍。"
            self.post_control_action({"action": "set_motor_angle", "motorCommand": "absolute", "angle": angle})
            return f"已发送电机命令，旋转到{angle:.1f}度。"

        if name == "move_motor_relative":
            angle = self.coerce_angle(intent.get("angle"))
            if angle is None:
                return "我没有听清楚要相对旋转多少度，请再说一遍。"
            self.post_control_action({"action": "move_motor_relative", "angle": angle})
            return f"已发送电机命令，相对旋转{angle:+.1f}度。"

        if name == "set_motor_zero":
            self.post_control_action({"action": "set_motor_zero"})
            return "已发送电机命令，将当前位置设为零点。"

        if name == "shutdown_board":
            self.schedule_poweroff()
            delay = max(0.0, float(getattr(self.args, "ai_shutdown_delay_sec", 0.0)))
            if delay <= 0:
                return "已收到关机指令，开发板正在关机。"
            return f"已收到关机指令，开发板将在{delay:.0f}秒后关机。"

        if name == "reboot_board":
            self.schedule_reboot()
            return "已收到重启指令，开发板正在重启。"

        if name == "send_message":
            message = str(intent.get("message") or intent.get("text") or "").strip()
            if not message:
                return "我没有听清楚要发送的信息内容，请再说一遍。"
            self.post_control_action({"action": "message_to_app", "text": message, "source": "ai"})
            return f"已发送信息：{message}。"

        if name == "clear_messages":
            self.post_control_action({"action": "clear_messages", "source": "ai"})
            return "已清除双向交流的聊天记录。"

        if name == "list_reminders":
            status = self.get_control_status()
            reminders = status.get("reminders") if isinstance(status.get("reminders"), list) else []
            return self.format_reminders_reply(reminders)

        if name == "add_reminder":
            reminder = self.reminder_payload(intent)
            if not reminder.get("label"):
                return "我没有听清楚提醒内容，请再说一遍。"
            if not reminder.get("time"):
                return "我没有听清楚提醒时间，请再说一遍。"
            self.post_control_action({"action": "reminder_add", "reminder": reminder})
            return f"已新增提醒：{self.format_reminder_schedule(reminder)}。"

        if name == "update_reminder":
            reminder = self.reminder_payload(intent)
            reminder_id = str(intent.get("reminderId") or "").strip()
            target_label = str(intent.get("targetLabel") or reminder.get("label") or "").strip()
            if not reminder_id and not target_label:
                return "我没有听清楚要修改哪一个提醒，请说出提醒名称。"
            self.post_control_action({
                "action": "reminder_update",
                "reminderId": reminder_id,
                "targetLabel": target_label,
                "reminder": reminder,
            })
            return f"已发送修改提醒的指令：{target_label or reminder_id}。"

        if name == "delete_reminder":
            reminder_id = str(intent.get("reminderId") or "").strip()
            target_label = str(intent.get("targetLabel") or "").strip()
            if not reminder_id and not target_label:
                return "我没有听清楚要删除哪一个提醒，请说出提醒名称。"
            self.post_control_action({
                "action": "reminder_delete",
                "reminderId": reminder_id,
                "targetLabel": target_label,
            })
            return f"已发送删除提醒的指令：{target_label or reminder_id}。"

        return None

    def plan_needs_web(self, plan: dict) -> bool:
        if not bool(getattr(self.args, "ai_web_search", True)):
            return False
        value = plan.get("chat_need_web", plan.get("need_web", False))
        if isinstance(value, bool):
            return value
        return str(value or "").strip().lower() in {"true", "1", "yes", "on"}

    def web_plan_from_control_plan(self, plan: dict, text: str) -> dict:
        recency = str(plan.get("search_recency") or plan.get("recency") or "noLimit")
        if recency not in {"oneDay", "oneWeek", "oneMonth", "oneYear", "noLimit"}:
            recency = "noLimit"
        return {
            "search_query": str(plan.get("search_query") or text).strip(),
            "search_recency": recency,
            "answer_style": str(plan.get("answer_style") or "short").strip() or "short",
            "must_include": plan.get("must_include") if isinstance(plan.get("must_include"), list) else [],
        }

    def ask_web_search(self, text: str, plan: dict | None = None) -> str:
        from ai_web_search import FastWebSearch

        web = FastWebSearch(self.args, self.deepseek_api_key())
        return web.answer(text, plan or {})

    def ask_chat_text(self, text: str, plan: dict | None = None) -> str:
        plan = plan if isinstance(plan, dict) else {}
        direct_answer = str(plan.get("direct_answer") or "").strip()
        if direct_answer and not self.plan_needs_web(plan):
            return direct_answer
        if self.plan_needs_web(plan):
            try:
                return self.ask_web_search(text, self.web_plan_from_control_plan(plan, text))
            except Exception as exc:
                print(f"[ai-web] failed, skip web answer: {exc}", flush=True)
                return "联网搜索暂时失败，我现在无法确认最新信息。"
        return self.ask_deepseek(text)

    def try_handle_control_intent(self, text: str) -> str | None:
        if not bool(getattr(self.args, "ai_control_intents", True)):
            return None
        try:
            plan = self.parse_control_intent(text)
            actions, chat_text, plan_confidence = self.normalize_control_plan(plan)
            threshold = float(getattr(self.args, "ai_control_confidence", 0.75))
            if not actions:
                if self.plan_needs_web(plan) or str(plan.get("direct_answer") or "").strip():
                    return self.ask_chat_text(chat_text or text, plan)
                return None

            if any(str(action.get("intent") or "").strip() == "get_status" for action in actions):
                actions = [
                    action
                    for action in actions
                    if str(action.get("intent") or "").strip() != "get_ip"
                ]

            replies: list[str] = []
            executed_names: list[str] = []
            for action in actions:
                name = str(action.get("intent") or "").strip()
                confidence = self.action_confidence(action, plan_confidence)
                if name not in CONTROL_INTENTS or confidence < threshold:
                    continue
                try:
                    action_reply = self.execute_control_intent(action)
                except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                    action_reply = f"控制{name}时连接本地服务失败。"
                    print(f"[ai-control] action failed intent={name} error={exc}", flush=True)
                if action_reply:
                    replies.append(action_reply)
                    executed_names.append(name)

            if not replies:
                return None

            if chat_text:
                try:
                    chat_reply = self.ask_chat_text(chat_text, plan)
                    if chat_reply:
                        replies.append(chat_reply)
                except Exception as exc:
                    replies.append("另外的问题我暂时没有回答成功。")
                    print(f"[ai-control] chat part failed after control: {exc}", flush=True)

            reply = "".join(replies)
            print(f"[ai-control] executed intents={executed_names} reply={reply}", flush=True)
            return reply
        except Exception as exc:
            print(f"[ai-control] intent handling failed, fallback to chat: {exc}", flush=True)
            return None

    def ask_deepseek(self, text: str, context_prefix: str = "") -> str:
        from openai import OpenAI

        api_key = self.deepseek_api_key()
        if not api_key:
            raise RuntimeError("missing DeepSeek API key")

        user_content = text
        if context_prefix:
            user_content = f"{context_prefix}\n老人回答：{text}"
        self.messages.append({"role": "user", "content": user_content})
        self.trim_messages(max_rounds=self.args.ai_history_rounds)

        client = OpenAI(api_key=api_key, base_url=self.args.deepseek_base_url)
        response = client.chat.completions.create(
            model=self.args.deepseek_model,
            messages=self.messages,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        answer = str(response.choices[0].message.content or "").strip()
        self.messages.append({"role": "assistant", "content": answer})
        self.save_history()
        return answer

    def play_audio(self, filename: Path, session_id: int, cancel_event: threading.Event) -> None:
        cmd = [
            self.args.ai_gst_launch_bin,
            "-q",
            "filesrc",
            f"location={filename}",
            "!",
            "decodebin",
            "!",
            "audioconvert",
            "!",
            "audioresample",
            "!",
            "alsasink",
            f"device={self.args.ai_play_device}",
        ]
        print("[ai] play:", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd)
        self.play_proc = proc
        try:
            while proc.poll() is None:
                if self.is_cancelled(session_id, cancel_event):
                    self.terminate_proc(proc)
                    raise InterruptedError("Interrupted by phone")
                time.sleep(0.05)
        finally:
            if self.play_proc is proc:
                self.play_proc = None
        if proc.returncode != 0:
            raise RuntimeError(f"audio playback failed with code {proc.returncode}")

    def close(self) -> None:
        self.cancel_current("assistant closing", "Closed")
        proc = self.record_proc
        self.record_proc = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
