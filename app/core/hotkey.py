"""Global hotkey registration with defensive error boundaries."""
import threading
from collections.abc import Callable

import keyboard


class HotkeyManager:
    def __init__(self):
        self._callback: Callable[[], None] | None = None
        self._hotkey = "ctrl+shift+space"
        self._registered = False
        self._hotkey_ref: int | str | None = None
        self._lock = threading.RLock()
        self._last_error: str | None = None

    @staticmethod
    def _normalize_hotkey(hotkey: str) -> str:
        if not isinstance(hotkey, str):
            return "ctrl+shift+space"
        normalized = hotkey.strip().lower()
        return normalized or "ctrl+shift+space"

    def _unregister_locked(self) -> None:
        if not self._registered:
            return

        try:
            if self._hotkey_ref is not None:
                keyboard.remove_hotkey(self._hotkey_ref)
            else:
                keyboard.remove_hotkey(self._hotkey)
        except Exception:
            pass
        finally:
            self._registered = False
            self._hotkey_ref = None

    def _register_locked(self, hotkey: str) -> bool:
        try:
            self._hotkey_ref = keyboard.add_hotkey(hotkey, self._on_hotkey, suppress=True)
            self._hotkey = hotkey
            self._registered = True
            self._last_error = None
            return True
        except Exception as exc:
            self._hotkey = hotkey
            self._registered = False
            self._hotkey_ref = None
            self._last_error = str(exc)
            return False

    def start(self, hotkey: str, callback) -> None:
        """Register global hotkey. Errors are contained to avoid app crash."""
        normalized = self._normalize_hotkey(hotkey)
        with self._lock:
            self._callback = callback
            self._unregister_locked()
            self._register_locked(normalized)

    def _on_hotkey(self):
        callback = self._callback
        if callback:
            try:
                callback()
            except Exception:
                # Callback errors should not crash keyboard hook thread.
                pass

    def stop(self) -> None:
        with self._lock:
            self._unregister_locked()

    def update_hotkey(self, new_hotkey: str) -> None:
        normalized = self._normalize_hotkey(new_hotkey)
        with self._lock:
            callback = self._callback
            old_hotkey = self._hotkey
            old_registered = self._registered

            if callback is None:
                self._hotkey = normalized
                self._last_error = None
                return

            self._unregister_locked()
            if self._register_locked(normalized):
                return

            if old_registered:
                self._register_locked(old_hotkey)

    def get_last_error(self) -> str | None:
        with self._lock:
            return self._last_error
