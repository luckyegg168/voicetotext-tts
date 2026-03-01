"""Clipboard helpers with lazy imports to avoid startup hard-fail."""

from __future__ import annotations

import threading
import time


class ClipboardDependencyError(RuntimeError):
    """Raised when optional clipboard dependencies are unavailable."""


def _require_pyperclip():
    try:
        import pyperclip  # type: ignore
    except Exception as exc:
        raise ClipboardDependencyError(
            "未安裝 pyperclip，請先 pip install pyperclip"
        ) from exc
    return pyperclip


def _require_pyautogui():
    try:
        import pyautogui  # type: ignore
    except Exception as exc:
        raise ClipboardDependencyError(
            "未安裝或無法使用 pyautogui，請先 pip install pyautogui"
        ) from exc
    return pyautogui


def copy_to_clipboard(text: str) -> bool:
    """Copy text to clipboard. Returns False when dependency is unavailable."""
    try:
        pyperclip = _require_pyperclip()
        pyperclip.copy(text)
        return True
    except ClipboardDependencyError:
        return False


def paste_to_foreground(text: str, delay: float = 0.3) -> bool:
    """Paste text into foreground window via Ctrl+V. Returns False when unavailable."""
    try:
        pyperclip = _require_pyperclip()
        pyautogui = _require_pyautogui()
    except ClipboardDependencyError:
        return False

    def _paste():
        time.sleep(delay)
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")

    threading.Thread(target=_paste, daemon=True).start()
    return True
