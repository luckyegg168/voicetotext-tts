"""字典管理頁面。"""

import customtkinter as ctk
from tkinter import messagebox

from app.utils.storage import load_dictionary, save_dictionary


class DictPage(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._entries: list[tuple[ctk.CTkEntry, ctk.CTkEntry]] = []
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(
            self,
            text="字典管理",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(
            self,
            text="轉寫完成後會先套用此替換表，再進入整理流程。",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(anchor="w", pady=(0, 12))

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            header,
            text="錯字",
            width=200,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left", padx=(0, 8))
        ctk.CTkLabel(
            header,
            text="替換成",
            width=200,
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(side="left")

        self._scroll = ctk.CTkScrollableFrame(self, height=320)
        self._scroll.pack(fill="x", pady=(0, 12))

        self._load_words()

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x")

        ctk.CTkButton(
            btn_row,
            text="+ 新增一列",
            width=110,
            command=self._add_row,
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row,
            text="儲存",
            width=80,
            command=self._save,
        ).pack(side="left")

    def _load_words(self):
        for w in self._scroll.winfo_children():
            w.destroy()

        self._entries = []

        try:
            words = load_dictionary()
        except Exception as e:
            self._show_error(f"讀取字典失敗：{e}")
            self._add_row()
            return

        for wrong, correct in words.items():
            self._add_row(wrong, correct)

        if not words:
            self._add_row()

    def _add_row(self, wrong: str = "", correct: str = ""):
        row = ctk.CTkFrame(self._scroll, fg_color="transparent")
        row.pack(fill="x", pady=2)

        e1 = ctk.CTkEntry(row, width=200, placeholder_text="錯字")
        e1.insert(0, wrong)
        e1.pack(side="left", padx=(0, 8))

        e2 = ctk.CTkEntry(row, width=200, placeholder_text="替換成")
        e2.insert(0, correct)
        e2.pack(side="left", padx=(0, 8))

        pair = (e1, e2)
        self._entries.append(pair)

        ctk.CTkButton(
            row,
            text="刪",
            width=30,
            height=28,
            fg_color="#c0392b",
            hover_color="#922b21",
            command=lambda p=pair, r=row: self._delete_row(p, r),
        ).pack(side="left")

    def _delete_row(self, pair, row):
        row.destroy()
        if pair in self._entries:
            self._entries.remove(pair)

    def _save(self):
        words = {}
        for e1, e2 in self._entries:
            wrong = e1.get().strip()
            correct = e2.get().strip()
            if wrong and correct:
                words[wrong] = correct

        try:
            save_dictionary(words)
        except Exception as e:
            self._show_error(f"儲存字典失敗：{e}")
            return

        self._show_toast("字典已儲存")

    def _show_error(self, msg: str):
        messagebox.showerror("字典", msg)

    def _show_toast(self, msg: str):
        toast = ctk.CTkLabel(
            self,
            text=msg,
            fg_color="#2ecc71",
            text_color="white",
            corner_radius=6,
            font=ctk.CTkFont(size=13),
        )
        toast.pack(pady=4)
        self.after(2000, toast.destroy)