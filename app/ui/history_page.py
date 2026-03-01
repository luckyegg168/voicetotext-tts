"""歷史紀錄頁面。"""

import csv
import customtkinter as ctk
from datetime import datetime
from tkinter import filedialog, messagebox

from app.utils import storage
from app.utils.clipboard import copy_to_clipboard


class HistoryPage(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._records: list[dict] = []
        self._build_ui()

    def _build_ui(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            header,
            text="歷史紀錄",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(side="left")

        ctk.CTkButton(
            header,
            text="重新整理",
            width=90,
            command=self.refresh,
        ).pack(side="right", padx=(6, 0))

        ctk.CTkButton(
            header,
            text="匯出 CSV",
            width=90,
            command=lambda: self._export("csv"),
        ).pack(side="right", padx=(6, 0))

        ctk.CTkButton(
            header,
            text="匯出 TXT",
            width=90,
            command=lambda: self._export("txt"),
        ).pack(side="right")

        self._scroll = ctk.CTkScrollableFrame(self)
        self._scroll.pack(fill="both", expand=True)

        self.refresh()

    def refresh(self):
        for widget in self._scroll.winfo_children():
            widget.destroy()

        try:
            self._records = storage.load_history()
        except Exception as e:
            self._records = []
            self._show_inline_error(f"讀取歷史紀錄失敗：{e}")
            return

        if not self._records:
            ctk.CTkLabel(
                self._scroll,
                text="尚無歷史紀錄",
                text_color="gray",
                font=ctk.CTkFont(size=14),
            ).pack(pady=40)
            return

        for record in self._records:
            self._add_record_card(record)

    def _show_inline_error(self, msg: str):
        ctk.CTkLabel(
            self._scroll,
            text=msg,
            text_color="#e74c3c",
            font=ctk.CTkFont(size=13),
            wraplength=720,
            justify="left",
        ).pack(fill="x", padx=8, pady=20)

    def _add_record_card(self, record: dict):
        card = ctk.CTkFrame(self._scroll, corner_radius=10)
        card.pack(fill="x", pady=4, padx=2)

        try:
            dt = datetime.fromisoformat(record["timestamp"])
            time_str = dt.strftime("%Y/%m/%d %H:%M")
        except Exception:
            time_str = record.get("timestamp", "")

        top_row = ctk.CTkFrame(card, fg_color="transparent")
        top_row.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            top_row,
            text=f"🕐 {time_str}  ·  {record.get('word_count', 0)} 字",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        ).pack(side="left")

        btn_frame = ctk.CTkFrame(top_row, fg_color="transparent")
        btn_frame.pack(side="right")

        ctk.CTkButton(
            btn_frame,
            text="複製",
            width=60,
            height=26,
            font=ctk.CTkFont(size=12),
            command=lambda r=record: self._copy_record_text(r.get("polished", r.get("original", ""))),
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            btn_frame,
            text="刪除",
            width=60,
            height=26,
            font=ctk.CTkFont(size=12),
            fg_color="#c0392b",
            hover_color="#922b21",
            command=lambda r=record: self._delete(r.get("id", "")),
        ).pack(side="left", padx=2)

        ctk.CTkLabel(
            card,
            text="原文",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#888",
        ).pack(anchor="w", padx=12)
        ctk.CTkLabel(
            card,
            text=record.get("original", ""),
            font=ctk.CTkFont(size=13),
            wraplength=580,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(0, 6))

        if record.get("polished"):
            ctk.CTkLabel(
                card,
                text="整理後",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="#888",
            ).pack(anchor="w", padx=12)
            ctk.CTkLabel(
                card,
                text=record.get("polished", ""),
                font=ctk.CTkFont(size=13),
                wraplength=580,
                justify="left",
                anchor="w",
            ).pack(fill="x", padx=12, pady=(0, 10))

    def _delete(self, record_id: str):
        if not record_id:
            messagebox.showerror("歷史紀錄", "無效的紀錄 ID，無法刪除")
            return

        try:
            storage.delete_history_record(record_id)
        except Exception as e:
            messagebox.showerror("歷史紀錄", f"刪除失敗：{e}")
            return

        self.refresh()

    def _copy_record_text(self, text: str):
        if not text:
            messagebox.showinfo("複製", "沒有可複製的內容")
            return
        if not copy_to_clipboard(text):
            messagebox.showerror("複製", "複製失敗：請安裝 pyperclip")
            return
        messagebox.showinfo("複製", "已複製到剪貼簿")

    def _export(self, fmt: str):
        try:
            records = storage.load_history()
        except Exception as e:
            messagebox.showerror("匯出", f"讀取歷史紀錄失敗：{e}")
            return

        if not records:
            messagebox.showinfo("匯出", "沒有可匯出的歷史紀錄")
            return

        if fmt == "csv":
            path = filedialog.asksaveasfilename(
                title="匯出 CSV",
                defaultextension=".csv",
                filetypes=[("CSV 檔案", "*.csv")],
                initialfile="voice_history.csv",
            )
        else:
            path = filedialog.asksaveasfilename(
                title="匯出 TXT",
                defaultextension=".txt",
                filetypes=[("文字檔", "*.txt")],
                initialfile="voice_history.txt",
            )

        if not path:
            return

        try:
            if fmt == "csv":
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["timestamp", "template", "word_count", "original", "polished"],
                    )
                    writer.writeheader()
                    for r in records:
                        writer.writerow(
                            {
                                "timestamp": r.get("timestamp", ""),
                                "template": r.get("template", ""),
                                "word_count": r.get("word_count", 0),
                                "original": r.get("original", ""),
                                "polished": r.get("polished", ""),
                            }
                        )
            else:
                with open(path, "w", encoding="utf-8") as f:
                    for r in records:
                        try:
                            dt = datetime.fromisoformat(r["timestamp"]).strftime("%Y/%m/%d %H:%M")
                        except Exception:
                            dt = r.get("timestamp", "")

                        f.write(f"{'=' * 60}\n")
                        f.write(
                            f"時間：{dt}  模板：{r.get('template', '')}  字數：{r.get('word_count', 0)}\n"
                        )
                        f.write(f"【原文】\n{r.get('original', '')}\n\n")
                        if r.get("polished"):
                            f.write(f"【整理後】\n{r.get('polished', '')}\n")
                        f.write("\n")
        except OSError as e:
            messagebox.showerror("匯出", f"檔案寫入失敗：{e}")
            return
        except Exception as e:
            messagebox.showerror("匯出", f"匯出失敗：{e}")
            return

        messagebox.showinfo("匯出完成", f"已儲存至\n{path}")
