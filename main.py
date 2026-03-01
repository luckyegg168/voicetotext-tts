"""
Steven's Voice Workspace — 主程式進入點
自然說話，快速成文
"""
# ⚠ 必須在所有其他 import 之前執行 CUDA 預載入
from app.core.cuda_setup import setup as _cuda_setup
_cuda_setup()

import sys
import customtkinter as ctk

from app.core.hotkey import HotkeyManager
from app.ui.main_window import HomePage, InputSettingsPage, APP_NAME
from app.ui.history_page import HistoryPage
from app.ui.dict_page import DictPage
from app.version import __version__


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME}  v{__version__}")
        self.geometry("1100x720")
        self.minsize(900, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._hotkey_manager = HotkeyManager()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # 左側導覽列
        sidebar = ctk.CTkFrame(self, width=148, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # 標題
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

        # 頁面容器
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.pack(side="right", fill="both", expand=True, padx=20, pady=16)

        # 建立各頁面
        home_page = HomePage(self._content, hotkey_manager=self._hotkey_manager)
        self._pages: dict[str, ctk.CTkFrame] = {
            "home": home_page,
            "settings": InputSettingsPage(self._content, home_page=home_page),
            "history": HistoryPage(self._content),
            "dict": DictPage(self._content),
        }

        nav_items = [
            ("home", "首頁"),
            ("settings", "輸入設定"),
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
        self._show_page("home")

    def _show_page(self, key: str):
        if self._current_page == key:
            return
        # 隱藏舊頁面
        if self._current_page:
            self._pages[self._current_page].pack_forget()
            self._nav_buttons[self._current_page].configure(fg_color="transparent")

        # 顯示新頁面
        self._pages[key].pack(fill="both", expand=True)
        self._nav_buttons[key].configure(fg_color=("gray75", "gray25"))
        self._current_page = key

        # 歷史頁面自動重新整理
        if key == "history":
            self._pages["history"].refresh()

    def _on_close(self):
        self._hotkey_manager.stop()
        self.destroy()
        sys.exit(0)


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
