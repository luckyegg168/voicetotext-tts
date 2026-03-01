"""Configuration loading and persistence helpers."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"

DEFAULT_CONFIG = {
    "openai_api_key": "",
    "openrouter_api_key": "",
    "api_provider": "本地 (Ollama/LM Studio)",
    "polish_model": "",
    "whisper_source": "本地 (GPU)",
    "whisper_local_model": "medium",
    "whisper_device": "cuda",
    "local_api_url": "http://localhost:11434/v1",
    "local_model_name": "",
    "transcription_language": "auto",
    "output_language": "original",
    "template": "general",
    "auto_switch_template": True,
    "auto_paste": True,
    "auto_clear": True,
    "auto_translate": False,
    "default_translate_lang": "英文",
    "hotkey": "ctrl+shift+space",
    "theme": "dark",
}

_CONFIG_LOCK = threading.RLock()


def _normalize_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _normalize_str(value: Any, default: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    if isinstance(value, (bool, int, float)):
        return str(value)
    return default


def _normalize_config(raw: Any) -> dict:
    normalized = DEFAULT_CONFIG.copy()
    if not isinstance(raw, dict):
        return normalized

    for key, default_value in DEFAULT_CONFIG.items():
        candidate = raw.get(key, default_value)
        if isinstance(default_value, bool):
            normalized[key] = _normalize_bool(candidate, default_value)
        elif isinstance(default_value, str):
            normalized[key] = _normalize_str(candidate, default_value)
        else:
            normalized[key] = candidate if isinstance(candidate, type(default_value)) else default_value

    for key, value in raw.items():
        if key not in normalized:
            normalized[key] = value

    return normalized


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            json.dump(payload, temp_file, ensure_ascii=False, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temp_path)
        raise


def _corrupt_backup_path(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.name}.corrupt.{timestamp}")


def _quarantine_corrupt_json(path: Path, exc: Exception) -> None:
    backup = _corrupt_backup_path(path)
    try:
        path.replace(backup)
        warnings.warn(f"Corrupt JSON moved to: {backup} ({exc})")
    except OSError:
        warnings.warn(f"Corrupt JSON detected but failed to backup: {path} ({exc})")


def _lock_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.lock")


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    lock_path = _lock_path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"0")
            lock_file.flush()

        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_config() -> dict:
    with _CONFIG_LOCK:
        if CONFIG_PATH.exists():
            with _file_lock(CONFIG_PATH):
                try:
                    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return _normalize_config(data)
                except json.JSONDecodeError as exc:
                    _quarantine_corrupt_json(CONFIG_PATH, exc)
                except OSError:
                    pass
        return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> None:
    normalized = _normalize_config(config)
    with _CONFIG_LOCK:
        with _file_lock(CONFIG_PATH):
            _atomic_write_json(CONFIG_PATH, normalized)
