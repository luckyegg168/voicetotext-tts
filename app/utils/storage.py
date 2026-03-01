"""History and dictionary storage helpers."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

HISTORY_PATH = Path(__file__).parent.parent.parent / "history.json"
DICT_PATH = Path(__file__).parent.parent.parent / "dictionary.json"

_HISTORY_LOCK = threading.RLock()
_DICT_LOCK = threading.RLock()
_MAX_HISTORY_RECORDS = 500


def _atomic_write_json(path: Path, payload: Any) -> None:
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


def _load_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        _quarantine_corrupt_json(path, exc)
        return fallback
    except OSError:
        return fallback


def _normalize_string(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    if isinstance(value, (bool, int, float)):
        return str(value)
    return default


def _normalize_word_count(value: Any, fallback_text: str) -> int:
    if isinstance(value, bool):
        return len(fallback_text)
    if isinstance(value, (int, float)):
        return max(int(value), 0)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return len(fallback_text)


def _normalize_history_record(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None

    original = _normalize_string(raw.get("original"), "")
    polished = _normalize_string(raw.get("polished"), "")
    template = _normalize_string(raw.get("template"), "general") or "general"
    record_id = _normalize_string(raw.get("id"), "")
    timestamp = _normalize_string(raw.get("timestamp"), "")

    if not record_id:
        record_id = str(uuid.uuid4())
    if not timestamp:
        timestamp = datetime.now().isoformat()

    word_count = _normalize_word_count(raw.get("word_count"), polished)

    return {
        "id": record_id,
        "timestamp": timestamp,
        "original": original,
        "polished": polished,
        "template": template,
        "word_count": word_count,
    }


def _normalize_history(records: Any) -> list[dict]:
    if not isinstance(records, list):
        return []
    normalized: list[dict] = []
    for item in records:
        row = _normalize_history_record(item)
        if row is not None:
            normalized.append(row)
    return normalized[:_MAX_HISTORY_RECORDS]


def _normalize_dictionary(words: Any) -> dict[str, str]:
    if not isinstance(words, dict):
        return {}
    normalized: dict[str, str] = {}
    for wrong, correct in words.items():
        wrong_text = _normalize_string(wrong).strip()
        correct_text = _normalize_string(correct).strip()
        if wrong_text and correct_text:
            normalized[wrong_text] = correct_text
    return normalized


def _load_history_unlocked() -> list[dict]:
    return _normalize_history(_load_json(HISTORY_PATH, []))


def _save_history_unlocked(records: list[dict]) -> None:
    normalized = _normalize_history(records)
    _atomic_write_json(HISTORY_PATH, normalized)


def load_history() -> list[dict]:
    with _HISTORY_LOCK:
        with _file_lock(HISTORY_PATH):
            return _load_history_unlocked()


def save_history(records: list[dict]) -> None:
    with _HISTORY_LOCK:
        with _file_lock(HISTORY_PATH):
            _save_history_unlocked(records)


def add_history_record(original: str, polished: str, template: str) -> dict:
    record = _normalize_history_record(
        {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "original": original,
            "polished": polished,
            "template": template,
            "word_count": len(_normalize_string(polished, "")),
        }
    )
    if record is None:
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "original": "",
            "polished": "",
            "template": "general",
            "word_count": 0,
        }

    with _HISTORY_LOCK:
        with _file_lock(HISTORY_PATH):
            records = _load_history_unlocked()
            records.insert(0, record)
            _save_history_unlocked(records[:_MAX_HISTORY_RECORDS])

    return record


def delete_history_record(record_id: str) -> None:
    target = _normalize_string(record_id, "").strip()
    if not target:
        return

    with _HISTORY_LOCK:
        with _file_lock(HISTORY_PATH):
            records = _load_history_unlocked()
            filtered = [record for record in records if record.get("id") != target]
            _save_history_unlocked(filtered)


def load_dictionary() -> dict[str, str]:
    with _DICT_LOCK:
        with _file_lock(DICT_PATH):
            return _normalize_dictionary(_load_json(DICT_PATH, {}))


def save_dictionary(words: dict[str, str]) -> None:
    normalized = _normalize_dictionary(words)
    with _DICT_LOCK:
        with _file_lock(DICT_PATH):
            _atomic_write_json(DICT_PATH, normalized)


def apply_dictionary(text: str) -> str:
    """Apply correction dictionary replacements to text."""
    output = _normalize_string(text, "")
    words = load_dictionary()
    for wrong, correct in words.items():
        output = output.replace(wrong, correct)
    return output


def get_total_word_count() -> int:
    records = load_history()
    return sum(int(record.get("word_count", 0)) for record in records)


def get_recording_count() -> int:
    return len(load_history())
