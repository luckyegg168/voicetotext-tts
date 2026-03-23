"""Application entrypoint for Steven's Voice Workspace."""

from __future__ import annotations

import sys
import threading

import customtkinter as ctk

from app.core.cuda_setup import setup as _cuda_setup
from app.core.hotkey import HotkeyManager
from app.core.model_prewarmer import prewarm_models
from app.utils.config import load_config
from app.ui.dict_page import DictPage
from app.ui.history_page import HistoryPage
from app.ui.main_window import (
    APP_NAME,
    AsrAppPage,
    HomePage,
    SettingsPage,
    TtsAppPage,
)
from app.version import __version__


_cuda_setup()


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME}  v{__version__}")
        self.geometry("1180x760")
        self.minsize(940, 620)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._hotkey_manager = HotkeyManager()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(500, self._start_prewarm)

    def _build_ui(self):
        sidebar = ctk.CTkFrame(self, width=170, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="Steven's\nVoice",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(padx=16, pady=(20, 4))
        ctk.CTkLabel(
            sidebar,
            text="Workspace",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(padx=16, pady=(0, 4))
        ctk.CTkLabel(
            sidebar,
            text=f"v{__version__}",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(padx=16, pady=(0, 24))

        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.pack(side="right", fill="both", expand=True, padx=20, pady=16)

        home_page = HomePage(self._content, hotkey_manager=self._hotkey_manager)
        self._pages: dict[str, ctk.CTkFrame] = {
            "one_shot": home_page,
            "asr_app": AsrAppPage(self._content, home_page=home_page),
            "tts_app": TtsAppPage(self._content, home_page=home_page),
            "settings": SettingsPage(self._content, home_page=home_page),
            "history": HistoryPage(self._content),
            "dict": DictPage(self._content),
        }

        nav_items = [
            ("one_shot", "一次完成"),
            ("asr_app", "ASR APP"),
            ("tts_app", "TTS APP"),
            ("settings", "設定"),
            ("history", "歷史紀錄"),
            ("dict", "字典"),
        ]

        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for key, label in nav_items:
            btn = ctk.CTkButton(
                sidebar,
                text=label,
                anchor="w",
                font=ctk.CTkFont(size=14),
                fg_color="transparent",
                text_color=("gray10", "gray90"),
                hover_color=("gray80", "gray30"),
                command=lambda k=key: self._show_page(k),
            )
            btn.pack(fill="x", padx=8, pady=2)
            self._nav_buttons[key] = btn

        self._current_page: str | None = None
        self._show_page("one_shot")

    def _show_page(self, key: str):
        if self._current_page == key:
            return

        if self._current_page:
            self._pages[self._current_page].pack_forget()
            self._nav_buttons[self._current_page].configure(fg_color="transparent")

        self._pages[key].pack(fill="both", expand=True)
        self._nav_buttons[key].configure(fg_color=("gray75", "gray25"))
        self._current_page = key

        if key == "history":
            self._pages["history"].refresh()

    def _on_close(self):
        self._hotkey_manager.stop()
        self.destroy()
        sys.exit(0)

    def _start_prewarm(self) -> None:
        """Launch background model pre-warming after UI is ready."""
        cfg = load_config()
        home_page = self._pages["one_shot"]

        def _status_cb(msg: str, color: str) -> None:
            self.after(0, lambda m=msg, c=color: home_page.update_model_status(m, c))
            if "就緒" in msg or "ready" in msg.lower():
                self.after(0, lambda: self.after(3000, lambda: home_page.update_model_status("", "gray")))

        threading.Thread(
            target=prewarm_models,
            args=(cfg, _status_cb),
            daemon=True,
        ).start()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
