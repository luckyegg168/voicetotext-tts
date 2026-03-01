"""跨應用剪貼簿操作與自動貼上"""
import time
import threading
import pyperclip
import pyautogui


def copy_to_clipboard(text: str) -> None:
    pyperclip.copy(text)


def paste_to_foreground(text: str, delay: float = 0.3) -> None:
    """複製文字到剪貼簿並模擬 Ctrl+V 貼到前景視窗"""
    def _paste():
        time.sleep(delay)
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")

    threading.Thread(target=_paste, daemon=True).start()
