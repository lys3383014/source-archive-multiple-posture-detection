from __future__ import annotations

import json
import time
from pathlib import Path


DEFAULT_AUDIO_CONTROL_FILE = "/tmp/rk3588_audio_control.json"


def _load(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {"owners": {}}
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(data.get("owners"), dict):
            return data
    except Exception:
        pass
    return {"owners": {}}


def _save(path: str | Path, data: dict) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = file_path.with_name(file_path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(file_path)


def mark_audio_owner(path: str | Path, owner: str, reason: str = "", hold_sec: float = 1.0) -> None:
    now = time.time()
    data = _load(path)
    owners = data.setdefault("owners", {})
    owners[owner] = {
        "reason": reason,
        "updated": now,
        "until": now + max(0.0, float(hold_sec)),
    }
    _save(path, data)


def clear_audio_owner(path: str | Path, owner: str) -> None:
    data = _load(path)
    owners = data.setdefault("owners", {})
    owners.pop(owner, None)
    _save(path, data)


def is_audio_owner_active(path: str | Path, owner: str, now: float | None = None) -> bool:
    now = time.time() if now is None else now
    data = _load(path)
    info = data.get("owners", {}).get(owner)
    if not isinstance(info, dict):
        return False
    try:
        return float(info.get("until", 0.0)) > now
    except (TypeError, ValueError):
        return False
