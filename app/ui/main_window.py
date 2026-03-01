"""主視窗 — 首頁（儀表板 + 設定）"""
import logging
import threading
import tkinter as tk
from pathlib import Path
import customtkinter as ctk

from app.core.recorder import AudioRecorder, RecorderStartError, RecordingState
from app.core.transcriber import transcribe, transcribe_local, is_whisper_model_cached, download_whisper_model
from app.core.polisher import polish, translate, TEMPLATE_LABELS, TRANSLATE_TARGETS
from app.core.ollama_helper import is_ollama_running, is_ollama_model_available, pull_ollama_model, list_ollama_models
from app.utils.config import load_config, save_config
from app.utils.storage import (
    add_history_record, apply_dictionary,
    get_total_word_count, get_recording_count,
    load_history,
)
from app.utils.clipboard import paste_to_foreground, copy_to_clipboard

APP_NAME = "Steven's Voice Workspace"
LOGGER = logging.getLogger(__name__)


def _auto_detect_template(cfg: dict) -> dict:
    """
    依前景視窗標題自動切換情境模板。
    規則：含有關鍵字就切換，否則保留目前設定。
    """
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) + 1
        buf = ctypes.create_unicode_buffer(length)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length)
        title = buf.value.lower()
    except Exception:
        return cfg

    keywords = {
        "social":  ["facebook", "instagram", "twitter", "threads", "ig", "fb", "臉書", "社群"],
        "meeting": ["zoom", "teams", "meet", "webex", "會議", "meeting"],
        "email":   ["outlook", "gmail", "thunderbird", "mail", "信箱", "郵件"],
    }
    for template, words in keywords.items():
        if any(w in title for w in words):
            return {**cfg, "template": template}
    return cfg

OPENROUTER_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3-haiku",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct",
]
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
WHISPER_LOCAL_MODELS = ["large-v3", "large-v2", "medium", "small", "base", "tiny"]

LANGUAGE_OPTIONS = {
    "自動偵測":  "auto",
    "繁體中文":  "zh-TW",
    "简体中文":  "zh-CN",
    "English":  "en",
    "日本語":   "ja",
    "한국어":   "ko",
}
OUTPUT_LANG_OPTIONS = {
    "原語言": "original",
    "繁體中文": "zh",
    "English": "en",
}


class HomePage(ctk.CTkFrame):
    def __init__(self, master, hotkey_manager=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._config = load_config()
        self._recorder = AudioRecorder()
        self._state = RecordingState.IDLE
        self._hotkey_manager = hotkey_manager
        self._record_start_time: float = 0.0
        self._timer_job: str | None = None
        self._history_records: list[dict] = []
        self._history_idx: int = -1
        self._build_ui()
        self._update_stats()

        if self._hotkey_manager:
            self._hotkey_manager.start(
                self._config.get("hotkey", "ctrl+shift+space"),
                self._toggle_recording,
            )
            error = self._hotkey_manager.get_last_error()
            if error:
                self._safe_after(lambda err=error: self._show_error(f"熱鍵註冊失敗：{err}"))

    # ─── UI 建構 ────────────────────────────────────────────────

    def _build_ui(self):
        ctk.CTkLabel(
            self,
            text="自然說話，快速成文",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(anchor="w", pady=(0, 2))

        hotkey_str = self._config.get("hotkey", "ctrl+shift+space").replace("+", " + ").upper()
        ctk.CTkLabel(
            self,
            text=f"熱鍵  {hotkey_str}（按一下開始，再按一下停止），說完即轉寫與潤稿。",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(anchor="w", pady=(0, 16))

        # 統計卡片
        stats_row = ctk.CTkFrame(self, fg_color="transparent")
        stats_row.pack(fill="x", pady=(0, 16))
        stats_row.columnconfigure((0, 1, 2, 3), weight=1)

        self._status_card = self._make_stat_card(stats_row, "狀態", "待機中", 0, green=True)
        self._words_card  = self._make_stat_card(stats_row, "累積字數", "0", 1, sub="已貼上內容統計")
        self._count_card  = self._make_stat_card(stats_row, "錄音次數", "0", 2, sub="本機歷史計數")
        self._compute_card = self._make_stat_card(stats_row, "運算裝置", "-", 3, sub="Whisper 轉寫來源")
        self._update_compute_card(self._config)

        # 主體（首頁只保留輸出區，設定移到獨立頁）
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self._build_output(body, column=0)

    def _make_stat_card(self, parent, title, value, col, sub="", green=False):
        card = ctk.CTkFrame(parent, corner_radius=10)
        card.grid(row=0, column=col, padx=6, pady=0, sticky="nsew")
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12), text_color="gray").pack(anchor="w", padx=14, pady=(10, 0))
        lbl = ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=28, weight="bold"),
                           text_color="#2ecc71" if green else None)
        lbl.pack(anchor="w", padx=14)
        sub_lbl = ctk.CTkLabel(card, text=sub or "", font=ctk.CTkFont(size=11), text_color="gray")
        sub_lbl.pack(anchor="w", padx=14, pady=(0, 10))
        lbl._sub_label = sub_lbl
        return lbl

    def _build_settings(self, parent):
        frame = ctk.CTkScrollableFrame(parent, corner_radius=10)
        frame.grid(row=0, column=0, padx=(0, 8), sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="輸入設定", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w")

        row = 1

        # 情境模板
        ctk.CTkLabel(frame, text="情境模板", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._template_var = ctk.StringVar(value=self._label_of_template(self._config.get("template", "general")))
        self._template_menu = ctk.CTkOptionMenu(frame, values=list(TEMPLATE_LABELS.values()), variable=self._template_var)
        self._template_menu.grid(row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # 轉寫語言
        ctk.CTkLabel(frame, text="轉寫語言", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._lang_var = ctk.StringVar(value=self._key_of(LANGUAGE_OPTIONS, self._config.get("transcription_language", "auto")))
        ctk.CTkOptionMenu(frame, values=list(LANGUAGE_OPTIONS.keys()), variable=self._lang_var).grid(
            row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # 輸出語言
        ctk.CTkLabel(frame, text="輸出語言", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._out_lang_var = ctk.StringVar(value=self._key_of(OUTPUT_LANG_OPTIONS, self._config.get("output_language", "original")))
        ctk.CTkOptionMenu(frame, values=list(OUTPUT_LANG_OPTIONS.keys()), variable=self._out_lang_var).grid(
            row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # 自動切換模板
        ctk.CTkLabel(frame, text="自動切換模板", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._auto_switch_var = ctk.BooleanVar(value=self._config.get("auto_switch_template", True))
        ctk.CTkCheckBox(frame, text="", variable=self._auto_switch_var).grid(row=row, column=1, padx=14, pady=4, sticky="w"); row += 1

        # 自動貼上
        ctk.CTkLabel(frame, text="自動貼上", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._auto_paste_var = ctk.BooleanVar(value=self._config.get("auto_paste", True))
        ctk.CTkCheckBox(frame, text="", variable=self._auto_paste_var).grid(row=row, column=1, padx=14, pady=4, sticky="w"); row += 1

        # 錄音前清除文字
        ctk.CTkLabel(frame, text="錄音前清除文字", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._auto_clear_var = ctk.BooleanVar(value=self._config.get("auto_clear", True))
        ctk.CTkCheckBox(frame, text="", variable=self._auto_clear_var).grid(row=row, column=1, padx=14, pady=4, sticky="w"); row += 1

        # 自動翻譯
        ctk.CTkLabel(frame, text="自動翻譯", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._auto_translate_var = ctk.BooleanVar(value=self._config.get("auto_translate", False))
        ctk.CTkCheckBox(frame, text="", variable=self._auto_translate_var).grid(row=row, column=1, padx=14, pady=4, sticky="w"); row += 1

        # 預設翻譯語系
        ctk.CTkLabel(frame, text="預設翻譯語系", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._default_translate_lang_var = ctk.StringVar(value=self._config.get("default_translate_lang", "英文"))
        ctk.CTkOptionMenu(frame, values=list(TRANSLATE_TARGETS.keys()),
                          variable=self._default_translate_lang_var,
                          command=self._on_default_lang_change).grid(
            row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # ── 分隔線 ──
        ctk.CTkLabel(frame, text="─── 轉寫設定 ───", font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=row, column=0, columnspan=2, padx=14, pady=(10, 2), sticky="w"); row += 1

        # Whisper 來源
        ctk.CTkLabel(frame, text="Whisper 來源", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._whisper_src_var = ctk.StringVar(value=self._config.get("whisper_source", "openai"))
        ctk.CTkOptionMenu(frame, values=["openai", "本地 (GPU)"],
                          variable=self._whisper_src_var,
                          command=self._on_whisper_src_change).grid(
            row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # 本地 Whisper 模型大小
        self._local_whisper_label = ctk.CTkLabel(frame, text="本地模型大小", font=ctk.CTkFont(size=13))
        self._local_whisper_label.grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._whisper_model_var = ctk.StringVar(value=self._config.get("whisper_local_model", "large-v3"))
        self._local_whisper_menu = ctk.CTkOptionMenu(
            frame, values=WHISPER_LOCAL_MODELS, variable=self._whisper_model_var,
            command=lambda _: self._check_whisper_model())
        self._local_whisper_menu.grid(row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # Whisper 模型狀態列
        self._whisper_model_status_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self._whisper_model_status_frame.grid(row=row, column=0, columnspan=2, padx=14, pady=(0, 4), sticky="ew")
        self._whisper_model_status_label = ctk.CTkLabel(
            self._whisper_model_status_frame, text="", font=ctk.CTkFont(size=12))
        self._whisper_model_status_label.pack(side="left")
        self._whisper_download_btn = ctk.CTkButton(
            self._whisper_model_status_frame, text="⬇ 下載模型",
            width=110, height=26, font=ctk.CTkFont(size=12),
            command=self._download_whisper_model)
        self._whisper_download_btn.pack(side="right")
        row += 1

        # GPU / CPU
        self._local_device_label = ctk.CTkLabel(frame, text="運算裝置", font=ctk.CTkFont(size=13))
        self._local_device_label.grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._whisper_device_var = ctk.StringVar(value=self._config.get("whisper_device", "cuda"))
        ctk.CTkOptionMenu(frame, values=["cuda", "cpu"],
                          variable=self._whisper_device_var).grid(
            row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # ── 分隔線 ──
        ctk.CTkLabel(frame, text="─── 潤稿設定 ───", font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=row, column=0, columnspan=2, padx=14, pady=(10, 2), sticky="w"); row += 1

        # 潤稿 API 提供者
        ctk.CTkLabel(frame, text="潤稿 API", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._provider_var = ctk.StringVar(value=self._config.get("api_provider", "openai"))
        ctk.CTkOptionMenu(frame, values=["openai", "openrouter", "本地 (Ollama/LM Studio)"],
                          variable=self._provider_var,
                          command=self._on_provider_change).grid(
            row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # OpenAI API Key
        ctk.CTkLabel(frame, text="OpenAI Key", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._openai_key_entry = ctk.CTkEntry(frame, placeholder_text="sk-...", show="*")
        self._openai_key_entry.insert(0, self._config.get("openai_api_key", ""))
        self._openai_key_entry.grid(row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # OpenRouter API Key
        ctk.CTkLabel(frame, text="OpenRouter Key", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._openrouter_key_entry = ctk.CTkEntry(frame, placeholder_text="sk-or-...", show="*")
        self._openrouter_key_entry.insert(0, self._config.get("openrouter_api_key", ""))
        self._openrouter_key_entry.grid(row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # 本地 API URL（Ollama / LM Studio）
        self._local_url_label = ctk.CTkLabel(frame, text="本地 API URL", font=ctk.CTkFont(size=13))
        self._local_url_label.grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._local_url_entry = ctk.CTkEntry(frame, placeholder_text="http://localhost:11434/v1")
        self._local_url_entry.insert(0, self._config.get("local_api_url", "http://localhost:11434/v1"))
        self._local_url_entry.grid(row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # 潤稿模型
        ctk.CTkLabel(frame, text="潤稿模型", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._model_var = ctk.StringVar(value=self._config.get("polish_model", "gpt-4o-mini"))
        self._model_menu = ctk.CTkOptionMenu(frame, values=OPENAI_MODELS, variable=self._model_var)
        self._model_menu.grid(row=row, column=1, padx=14, pady=4, sticky="ew"); row += 1

        # Ollama 已安裝模型下拉 + 刷新按鈕
        ctk.CTkLabel(frame, text="Ollama 模型", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=4, sticky="w")
        ollama_row_frame = ctk.CTkFrame(frame, fg_color="transparent")
        ollama_row_frame.grid(row=row, column=1, padx=14, pady=4, sticky="ew")
        ollama_row_frame.columnconfigure(0, weight=1)

        saved_model = self._config.get("local_model_name", "")
        self._ollama_model_var = ctk.StringVar(value=saved_model or "（點刷新載入清單）")
        self._ollama_model_menu = ctk.CTkOptionMenu(
            ollama_row_frame,
            values=[saved_model or "（點刷新載入清單）"],
            variable=self._ollama_model_var,
            command=lambda _: self._check_ollama_model(),
        )
        self._ollama_model_menu.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            ollama_row_frame, text="⟳", width=32, height=28,
            font=ctk.CTkFont(size=16),
            command=self._refresh_ollama_models,
        ).grid(row=0, column=1)
        row += 1

        # 手動輸入備用（LM Studio 或不在清單的模型）
        ctk.CTkLabel(frame, text="手動輸入", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=(0, 4), sticky="w")
        self._custom_model_entry = ctk.CTkEntry(frame, placeholder_text="手動輸入模型名稱（選填）")
        self._custom_model_entry.grid(row=row, column=1, padx=14, pady=(0, 4), sticky="ew"); row += 1

        # Ollama 狀態列
        self._ollama_status_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self._ollama_status_frame.grid(row=row, column=0, columnspan=2, padx=14, pady=(0, 4), sticky="ew")
        self._ollama_status_label = ctk.CTkLabel(
            self._ollama_status_frame, text="", font=ctk.CTkFont(size=12))
        self._ollama_status_label.pack(side="left")
        self._ollama_pull_btn = ctk.CTkButton(
            self._ollama_status_frame, text="⬇ 下載模型",
            width=110, height=26, font=ctk.CTkFont(size=12),
            command=self._download_ollama_model)
        self._ollama_pull_btn.pack(side="right")
        row += 1

        # 儲存
        ctk.CTkButton(frame, text="儲存設定", command=self._save_settings).grid(
            row=row, column=0, columnspan=2, padx=14, pady=14, sticky="ew"); row += 1

        ctk.CTkLabel(frame, text="錄音、轉寫、潤稿、跨 App 貼上已啟用",
                     font=ctk.CTkFont(size=11), text_color="gray").grid(
            row=row, column=0, columnspan=2, padx=14, pady=(0, 4), sticky="w")

        # 初始化顯示狀態
        self._on_whisper_src_change(self._whisper_src_var.get())
        self._on_provider_change(self._provider_var.get())
        # 綁定 Ollama 模型名稱輸入框的 FocusOut 事件，自動檢查
        self._custom_model_entry.bind("<FocusOut>", lambda _: self._check_ollama_model())
        self._custom_model_entry.bind("<Return>", lambda _: self._check_ollama_model())

    def _build_output(self, parent, column: int = 0):
        frame = ctk.CTkFrame(parent, corner_radius=10)
        frame.grid(row=0, column=column, sticky="nsew")
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure((0, 1, 2), weight=1)

        # Toolbar
        toolbar = ctk.CTkFrame(frame, fg_color="transparent")
        toolbar.grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 4))

        self._record_btn = ctk.CTkButton(
            toolbar,
            text="● 開始錄音",
            width=120,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=self._toggle_recording,
        )
        self._record_btn.pack(side="left", padx=(0, 4))

        self._timer_label = ctk.CTkLabel(
            toolbar,
            text="",
            width=60,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#e67e22",
        )
        self._timer_label.pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            toolbar,
            text="載入文字",
            width=100,
            command=self._load_text_file,
        ).pack(side="left", padx=(0, 4))

        self._polish_btn = ctk.CTkButton(
            toolbar,
            text="潤稿",
            width=70,
            fg_color="#27ae60",
            hover_color="#1e8449",
            command=self._polish_manual,
        )
        self._polish_btn.pack(side="left", padx=(0, 4))

        self._save_polished_btn = ctk.CTkButton(
            toolbar,
            text="儲存整理後",
            width=110,
            command=self._save_polished_text,
        )
        self._save_polished_btn.pack(side="left", padx=(0, 4))

        self._translate_btn = ctk.CTkButton(
            toolbar,
            text="翻譯",
            width=70,
            fg_color="#8e44ad",
            hover_color="#6c3483",
            command=self._do_translate,
        )
        self._translate_btn.pack(side="right", padx=(4, 0))

        self._translate_lang_var = ctk.StringVar(
            value=self._config.get("default_translate_lang", "英文")
        )
        ctk.CTkOptionMenu(
            toolbar,
            values=list(TRANSLATE_TARGETS.keys()),
            variable=self._translate_lang_var,
            width=110,
        ).pack(side="right", padx=(4, 0))

        ctk.CTkLabel(toolbar, text="翻譯語言：", font=ctk.CTkFont(size=13)).pack(side="right", padx=(8, 0))

        self._translate_src_var = ctk.StringVar(value="整理後")
        ctk.CTkOptionMenu(
            toolbar,
            values=["整理後", "原文"],
            variable=self._translate_src_var,
            width=90,
        ).pack(side="right", padx=(4, 0))

        ctk.CTkLabel(toolbar, text="翻譯來源：", font=ctk.CTkFont(size=13)).pack(side="right", padx=(8, 0))

        # History navigation
        nav_bar = ctk.CTkFrame(frame, fg_color="transparent")
        nav_bar.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(4, 8))

        ctk.CTkButton(
            nav_bar,
            text="← 舊紀錄",
            width=90,
            height=28,
            font=ctk.CTkFont(size=12),
            command=self._nav_prev,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            nav_bar,
            text="新紀錄 →",
            width=90,
            height=28,
            font=ctk.CTkFont(size=12),
            command=self._nav_next,
        ).pack(side="left", padx=(0, 6))

        self._nav_label = ctk.CTkLabel(
            nav_bar,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._nav_label.pack(side="left", padx=4)

        # 三欄文字區（原文 / 整理後 / 翻譯）
        col_defs = [
            ("原文（可編輯）", "_original_text", True),
            ("整理後", "_polished_text", False),
            ("翻譯", "_translate_result", False),
        ]
        for i, (title, attr, editable) in enumerate(col_defs):
            col = ctk.CTkFrame(frame, fg_color="transparent")
            col.grid(
                row=1,
                column=i,
                sticky="nsew",
                padx=(8 if i == 0 else 4, 8 if i == 2 else 4),
                pady=(0, 8),
            )
            col.rowconfigure(1, weight=1)
            col.columnconfigure(0, weight=1)

            hdr = ctk.CTkFrame(col, fg_color="transparent")
            hdr.grid(row=0, column=0, sticky="ew", pady=(0, 4))
            ctk.CTkLabel(
                hdr,
                text=title,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="gray",
            ).pack(side="left")
            ctk.CTkButton(
                hdr,
                text="複製",
                width=55,
                height=24,
                font=ctk.CTkFont(size=12),
                command=lambda a=attr: self._copy_text(getattr(self, a).get("1.0", "end").strip()),
            ).pack(side="right")

            tb = ctk.CTkTextbox(col, font=ctk.CTkFont(size=13), wrap="word")
            tb.grid(row=1, column=0, sticky="nsew")
            if not editable:
                tb.configure(state="disabled")
            setattr(self, attr, tb)

    def _load_text_file(self):
        """開啟檔案對話框，載入 .txt 到原文區"""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="選擇文字檔",
            filetypes=[("文字檔", "*.txt"), ("Markdown", "*.md"), ("所有檔案", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="cp950", errors="replace") as f:
                content = f.read()
        self._set_original(content)
        self._show_toast(f"已載入：{Path(path).name}")

    def _copy_text(self, text: str):
        if not text:
            self._show_toast("沒有可複製的內容")
            return
        if not copy_to_clipboard(text):
            self._show_error("複製失敗：請安裝 pyperclip")
            return
        self._show_toast("已複製到剪貼簿")

    def _can_start_action(self, action_name: str) -> bool:
        if self._state == RecordingState.RECORDING:
            self._set_translate_result(f"⚠ 錄音中，請先停止錄音再執行「{action_name}」。")
            return False
        if self._state in (RecordingState.TRANSCRIBING, RecordingState.POLISHING, RecordingState.TRANSLATING):
            self._set_translate_result(f"⚠ 目前正在處理語音，請稍後再執行「{action_name}」。")
            return False
        return True

    def _save_polished_text(self):
        if not self._can_start_action("儲存整理後"):
            return

        from tkinter import filedialog

        text = self._polished_text.get("1.0", "end").strip()
        if not text:
            self._set_polished("⚠ 沒有可儲存的整理後文字")
            return

        path = filedialog.asksaveasfilename(
            title="儲存整理後",
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")],
            initialfile="polished_output.txt",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            self._set_polished(f"❌ 儲存失敗：{e}")
            return

        self._show_toast(f"已儲存：{path}")

    def _polish_manual(self):
        """對原文區現有文字執行潤稿（不需錄音）"""
        if not self._can_start_action("潤稿"):
            return

        text = self._original_text.get("1.0", "end").strip()
        if not text:
            self._show_error("原文區沒有文字")
            return

        cfg_snapshot = self._get_current_config()
        self._set_state(RecordingState.POLISHING)

        def _do(cfg: dict, original_text: str):
            try:
                api_key, base_url, model = self._resolve_polish_api(cfg)
                polished = polish(
                    original_text,
                    api_key=api_key,
                    template=cfg.get("template", "general"),
                    output_language=cfg.get("output_language", "original"),
                    model=model,
                    base_url=base_url,
                )
                self._safe_after(lambda t=polished: self._set_polished(t))
                if cfg.get("auto_translate", False):
                    self._translate_in_worker(cfg, original_text=original_text, polished_text=polished)
                add_history_record(original_text, polished, cfg.get("template", "general"))
                if cfg.get("auto_paste", True):
                    if not paste_to_foreground(polished):
                        self._safe_after(lambda: self._show_error("自動貼上不可用：請安裝 pyperclip/pyautogui"))
                self._safe_after(lambda: self._set_state(RecordingState.DONE))
                self._safe_after(self._refresh_history_cache)
                self._safe_after(self._update_stats)
                self._safe_after(lambda: self._set_state(RecordingState.IDLE), delay_ms=2000)
            except Exception as e:
                self._safe_after(lambda err=str(e): self._show_error(err))

        threading.Thread(target=_do, args=(cfg_snapshot, text), daemon=True).start()

    def _on_default_lang_change(self, lang: str):
        """設定頁更改預設翻譯語系時，同步更新工具列的翻譯語言選單"""
        self._translate_lang_var.set(lang)

    def _translate_in_worker(self, cfg: dict, original_text: str, polished_text: str):
        target = self._translate_lang_var.get()
        src = self._translate_src_var.get()
        text = polished_text if src == "整理後" else original_text
        if not text.strip():
            return
        self._safe_after(lambda: self._set_state(RecordingState.TRANSLATING))
        self._safe_after(lambda t=target: self._set_translate_result(f"翻譯中（{t}）..."))
        try:
            api_key, base_url, model = self._resolve_polish_api(cfg)
            result = translate(
                text,
                target_lang=target,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
            self._safe_after(lambda r=result: self._set_translate_result(r))
        except Exception as e:
            self._safe_after(lambda err=str(e): self._set_translate_result(f"❌ {err}"))

    def _do_translate(self, skip_state_guard: bool = False):
        """執行翻譯"""
        if self._state == RecordingState.TRANSLATING:
            self._set_translate_result("⚠ 目前正在翻譯，請稍後再試。")
            return
        if not skip_state_guard and not self._can_start_action("翻譯"):
            return

        src = self._translate_src_var.get()
        text = (self._polished_text if src == "整理後" else self._original_text).get("1.0", "end").strip()

        if not text:
            self._set_translate_result("⚠ 沒有可翻譯的文字")
            return

        target = self._translate_lang_var.get()
        cfg_snapshot = self._get_current_config()
        self._set_state(RecordingState.TRANSLATING)
        self._set_translate_result(f"翻譯中（{target}）...")

        def _do(cfg: dict, content_text: str, target_lang: str):
            try:
                api_key, base_url, model = self._resolve_polish_api(cfg)
                result = translate(
                    content_text,
                    target_lang=target_lang,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )
                self._safe_after(lambda r=result: self._set_translate_result(r))
            except Exception as e:
                self._safe_after(lambda err=str(e): self._set_translate_result(f"❌ {err}"))
            finally:
                self._safe_after(lambda: self._set_state(RecordingState.DONE))
                self._safe_after(lambda: self._set_state(RecordingState.IDLE), delay_ms=1200)

        threading.Thread(target=_do, args=(cfg_snapshot, text, target), daemon=True).start()

    def _resolve_polish_api(self, cfg: dict) -> tuple[str, str | None, str]:
        """根據設定回傳 (api_key, base_url, model)"""
        provider = cfg.get("api_provider", "openai")
        if provider == "本地 (Ollama/LM Studio)":
            return "ollama", cfg.get("local_api_url", "http://localhost:11434/v1"), \
                   cfg.get("local_model_name", "llama3.2")
        elif provider == "openrouter":
            return cfg.get("openrouter_api_key", ""), "https://openrouter.ai/api/v1", \
                   cfg.get("polish_model", "openai/gpt-4o-mini")
        else:
            return cfg.get("openai_api_key", ""), None, cfg.get("polish_model", "gpt-4o-mini")

    # ─── 錄音流程 ────────────────────────────────────────────────

    def _toggle_recording(self):
        self._safe_after(self._do_toggle)

    def _do_toggle(self):
        if self._state == RecordingState.IDLE:
            self._start_recording()
        elif self._state == RecordingState.RECORDING:
            self._stop_and_process()
        else:
            self._set_translate_result(f"⚠ 目前狀態為「{self._state.value}」，請稍候再操作。")

    def _start_recording(self):
        import time

        cfg_snapshot = self._get_current_config()
        if cfg_snapshot.get("auto_clear", True):
            self._set_original("")
            self._set_polished("")
            self._set_translate_result("")

        try:
            self._recorder.start()
        except RecorderStartError as e:
            self._record_btn.configure(text="● 開始錄音", fg_color="#e74c3c", hover_color="#c0392b")
            self._timer_label.configure(text="")
            self._set_state(RecordingState.IDLE)
            self._set_polished(f"❌ 錄音啟動失敗：{e}")
            return
        except Exception as e:
            self._record_btn.configure(text="● 開始錄音", fg_color="#e74c3c", hover_color="#c0392b")
            self._timer_label.configure(text="")
            self._set_state(RecordingState.IDLE)
            self._set_polished(f"❌ 錄音啟動失敗：{e}")
            return

        self._set_state(RecordingState.RECORDING)
        self._record_btn.configure(text="■ 停止錄音", fg_color="#e67e22", hover_color="#ca6f1e")
        self._record_start_time = time.time()
        self._tick_timer()

    def _tick_timer(self):
        import time
        if self._state != RecordingState.RECORDING:
            self._timer_label.configure(text="")
            return
        elapsed = int(time.time() - self._record_start_time)
        m, s = divmod(elapsed, 60)
        self._timer_label.configure(text=f"{m:02d}:{s:02d}")
        self._timer_job = self.after(500, self._tick_timer)

    def _stop_and_process(self):
        cfg_snapshot = self._get_current_config()
        self._set_state(RecordingState.TRANSCRIBING)
        self._record_btn.configure(text="● 開始錄音", fg_color="#e74c3c", hover_color="#c0392b")
        self._timer_label.configure(text="")

        def _process(cfg: dict):
            try:
                # 根據前景視窗標題自動切換情境模板
                if cfg.get("auto_switch_template", True):
                    cfg = _auto_detect_template(cfg)

                # 轉寫
                self._safe_after(lambda: self._set_state(RecordingState.TRANSCRIBING))
                whisper_src = cfg.get("whisper_source", "openai")
                audio_payload = self._recorder.stop(
                    as_wav=(whisper_src != "本地 (GPU)")
                )
                if whisper_src == "本地 (GPU)":
                    if getattr(audio_payload, "size", 0) == 0:
                        self._safe_after(lambda: self._show_error("未錄到音訊，請再試一次"))
                        return
                elif not audio_payload:
                    self._safe_after(lambda: self._show_error("未錄到音訊，請再試一次"))
                    return

                if whisper_src == "本地 (GPU)":
                    original = transcribe_local(
                        audio_payload,
                        model_size=cfg.get("whisper_local_model", "large-v3"),
                        language=cfg.get("transcription_language", "auto"),
                        device=cfg.get("whisper_device", "cuda"),
                        compute_type="float16" if cfg.get("whisper_device", "cuda") == "cuda" else "int8",
                    )
                else:
                    openai_key = cfg.get("openai_api_key", "")
                    if not openai_key:
                        self._safe_after(lambda: self._show_error("Whisper 轉寫需要 OpenAI API Key"))
                        return
                    original = transcribe(
                        audio_payload,
                        api_key=openai_key,
                        language=cfg.get("transcription_language", "auto"),
                    )

                original = apply_dictionary(original)
                self._safe_after(lambda t=original: self._set_original(t))

                # 潤稿
                self._safe_after(lambda: self._set_state(RecordingState.POLISHING))
                polish_key, polish_base_url, model = self._resolve_polish_api(cfg)

                if not polish_key:
                    self._safe_after(lambda: self._show_error("請先設定 API Key"))
                    return

                polished = polish(
                    original,
                    api_key=polish_key,
                    template=cfg.get("template", "general"),
                    output_language=cfg.get("output_language", "original"),
                    model=model,
                    base_url=polish_base_url,
                )
                self._safe_after(lambda t=polished: self._set_polished(t))
                if cfg.get("auto_translate", False):
                    self._translate_in_worker(cfg, original_text=original, polished_text=polished)

                add_history_record(original, polished, cfg.get("template", "general"))
                if cfg.get("auto_paste", True):
                    if not paste_to_foreground(polished):
                        self._safe_after(lambda: self._show_error("自動貼上不可用：請安裝 pyperclip/pyautogui"))

                self._safe_after(lambda: self._set_state(RecordingState.DONE))
                self._safe_after(self._update_stats)
                self._safe_after(self._refresh_history_cache)
                self._safe_after(lambda: self._set_state(RecordingState.IDLE), delay_ms=2000)

            except Exception as e:
                self._safe_after(lambda err=str(e): self._show_error(err))

        threading.Thread(target=_process, args=(cfg_snapshot,), daemon=True).start()

    # ─── UI 更新 ─────────────────────────────────────────────────
    def _safe_after(self, callback, delay_ms: int = 0):
        """Schedule UI callback safely from any thread."""
        try:
            if delay_ms <= 0:
                self.after(0, callback)
            else:
                self.after(0, lambda: self.after(delay_ms, callback))
        except tk.TclError:
            # Window is closing/destroyed.
            return
        except Exception as exc:
            LOGGER.exception("Failed to schedule UI callback: %s", exc)

    def _set_state(self, state: RecordingState):
        self._state = state
        color = {
            RecordingState.IDLE:        "#2ecc71",
            RecordingState.RECORDING:   "#e67e22",
            RecordingState.TRANSCRIBING:"#3498db",
            RecordingState.POLISHING:   "#9b59b6",
            RecordingState.TRANSLATING: "#16a085",
            RecordingState.DONE:        "#2ecc71",
            RecordingState.ERROR:       "#e74c3c",
        }.get(state, "#3498db")
        self._status_card.configure(text=state.value, text_color=color)

        action_state = "disabled" if state in (
            RecordingState.RECORDING,
            RecordingState.TRANSCRIBING,
            RecordingState.POLISHING,
            RecordingState.TRANSLATING,
        ) else "normal"
        for btn_name in ("_polish_btn", "_translate_btn", "_save_polished_btn"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.configure(state=action_state)

    def _set_original(self, text: str):
        self._original_text.delete("1.0", "end")
        self._original_text.insert("1.0", text)

    def _set_polished(self, text: str):
        self._polished_text.configure(state="normal")
        self._polished_text.delete("1.0", "end")
        self._polished_text.insert("1.0", text)
        self._polished_text.configure(state="disabled")

    def _set_translate_result(self, text: str):
        self._translate_result.configure(state="normal")
        self._translate_result.delete("1.0", "end")
        self._translate_result.insert("1.0", text)
        self._translate_result.configure(state="disabled")

    def _show_error(self, msg: str):
        self._set_state(RecordingState.ERROR)
        self._set_polished(f"❌ 錯誤：{msg}")
        self._safe_after(lambda: self._set_state(RecordingState.IDLE), delay_ms=3000)

    def _update_stats(self):
        self._words_card.configure(text=str(get_total_word_count()))
        self._count_card.configure(text=str(get_recording_count()))
        self._update_compute_card(self._config)

    def _update_compute_card(self, cfg: dict):
        whisper_source = cfg.get("whisper_source", "openai")
        if whisper_source == "本地 (GPU)":
            device = str(cfg.get("whisper_device", "cuda")).lower()
            label = "GPU" if device == "cuda" else "CPU"
            sub = f"本地 Whisper（{device}）"
        else:
            label = "雲端"
            sub = "OpenAI Whisper API"
        self._compute_card.configure(text=label)
        sub_label = getattr(self._compute_card, "_sub_label", None)
        if sub_label is not None:
            sub_label.configure(text=sub)

    # ─── 模型檢查與下載 ──────────────────────────────────────────

    def _check_whisper_model(self):
        """檢查本地 Whisper 模型是否存在，更新狀態列"""
        model = self._whisper_model_var.get()

        def _check():
            try:
                cached = is_whisper_model_cached(model)
            except Exception as e:
                self._safe_after(lambda err=str(e): self._whisper_model_status_label.configure(
                    text=f"❌ 模型檢查失敗：{err}", text_color="#e74c3c"))
                self._safe_after(lambda: self._whisper_download_btn.configure(
                    state="normal", text="⬇ 下載模型", fg_color=("#3B8ED0", "#1F6AA5")))
                return
            if cached:
                self._safe_after(lambda: self._whisper_model_status_label.configure(
                    text=f"✅ {model} 已下載", text_color="#2ecc71"))
                self._safe_after(lambda: self._whisper_download_btn.configure(
                    state="disabled", text="✅ 已下載", fg_color="gray"))
            else:
                self._safe_after(lambda: self._whisper_model_status_label.configure(
                    text=f"⚠ {model} 尚未下載", text_color="#e67e22"))
                self._safe_after(lambda: self._whisper_download_btn.configure(
                    state="normal", text="⬇ 下載模型", fg_color=("#3B8ED0", "#1F6AA5")))

        threading.Thread(target=_check, daemon=True).start()

    def _download_whisper_model(self):
        """下載 faster-whisper 模型，並在完成後更新快取"""
        model = self._whisper_model_var.get()
        self._whisper_download_btn.configure(state="disabled", text="下載中...")

        def _status(msg: str):
            self._safe_after(lambda m=msg: self._whisper_model_status_label.configure(text=m))

        def _do():
            try:
                download_whisper_model(model, status_callback=_status)
                self._safe_after(lambda: self._whisper_download_btn.configure(
                    state="disabled", text="✅ 已下載", fg_color="gray"))
            except Exception as e:
                self._safe_after(lambda err=str(e): self._whisper_model_status_label.configure(
                    text=f"❌ {err}", text_color="#e74c3c"))
                self._safe_after(lambda: self._whisper_download_btn.configure(
                    state="normal", text="⬇ 下載模型", fg_color=("#3B8ED0", "#1F6AA5")))

        threading.Thread(target=_do, daemon=True).start()

    def _get_ollama_base(self) -> str:
        """取得 Ollama 根 URL（移除 /v1）"""
        url = self._local_url_entry.get().strip() or "http://localhost:11434/v1"
        return url.replace("/v1", "").rstrip("/")

    def _get_selected_ollama_model(self) -> str:
        """回傳手動輸入優先，否則用下拉選單的值"""
        manual = self._custom_model_entry.get().strip()
        return manual if manual else self._ollama_model_var.get().strip()

    def _refresh_ollama_models(self):
        """向 Ollama 查詢已安裝模型並更新下拉選單"""
        self._ollama_status_label.configure(text="載入中...", text_color="gray")
        api_base = self._get_ollama_base()

        def _fetch():
            try:
                if not is_ollama_running(api_base):
                    self._safe_after(lambda: self._ollama_status_label.configure(
                        text="❌ Ollama 未執行，請先啟動 Ollama", text_color="#e74c3c"))
                    return
                models = list_ollama_models(api_base)
                if not models:
                    self._safe_after(lambda: self._ollama_status_label.configure(
                        text="⚠ 尚無已安裝模型，請先 ollama pull", text_color="#e67e22"))
                    return
            except Exception as e:
                self._safe_after(lambda err=str(e): self._ollama_status_label.configure(
                    text=f"❌ 模型讀取失敗：{err}", text_color="#e74c3c"))
                self._safe_after(lambda: self._ollama_pull_btn.configure(
                    state="normal", text="⬇ 下載模型", fg_color=("#3B8ED0", "#1F6AA5")))
                return

            def _update(ms=models):
                self._ollama_model_menu.configure(values=ms)
                # 若目前值仍在清單內就保留，否則選第一個
                current = self._ollama_model_var.get()
                if current not in ms:
                    self._ollama_model_var.set(ms[0])
                self._ollama_status_label.configure(
                    text=f"✅ 找到 {len(ms)} 個模型", text_color="#2ecc71")
                self._ollama_pull_btn.configure(state="disabled", text="✅ 已安裝", fg_color="gray")

            self._safe_after(_update)

        threading.Thread(target=_fetch, daemon=True).start()

    def _check_ollama_model(self):
        """檢查目前選擇的模型是否存在"""
        model = self._get_selected_ollama_model()
        if not model or "點刷新" in model:
            return
        api_base = self._get_ollama_base()

        def _check():
            try:
                if not is_ollama_running(api_base):
                    self._safe_after(lambda: self._ollama_status_label.configure(
                        text="❌ Ollama 未執行", text_color="#e74c3c"))
                    return
                installed = list_ollama_models(api_base)
            except Exception as e:
                self._safe_after(lambda err=str(e): self._ollama_status_label.configure(
                    text=f"❌ 模型檢查失敗：{err}", text_color="#e74c3c"))
                self._safe_after(lambda: self._ollama_pull_btn.configure(
                    state="normal", text="⬇ 下載模型", fg_color=("#3B8ED0", "#1F6AA5")))
                return
            # 精確比對（含 tag）或前綴比對
            found = any(m == model or m.startswith(model.split(":")[0]) for m in installed)
            if found:
                self._safe_after(lambda: self._ollama_status_label.configure(
                    text=f"✅ {model} 已安裝", text_color="#2ecc71"))
                self._safe_after(lambda: self._ollama_pull_btn.configure(
                    state="disabled", text="✅ 已安裝", fg_color="gray"))
            else:
                self._safe_after(lambda: self._ollama_status_label.configure(
                    text=f"⚠ {model} 未安裝，可按下載", text_color="#e67e22"))
                self._safe_after(lambda: self._ollama_pull_btn.configure(
                    state="normal", text="⬇ 下載模型", fg_color=("#3B8ED0", "#1F6AA5")))

        threading.Thread(target=_check, daemon=True).start()

    def _download_ollama_model(self):
        """執行 ollama pull 下載目前選擇的模型"""
        model = self._get_selected_ollama_model()
        if not model or "點刷新" in model:
            return
        self._ollama_pull_btn.configure(state="disabled", text="下載中...")

        def _status(msg: str):
            self._safe_after(lambda m=msg: self._ollama_status_label.configure(text=m))

        def _do():
            try:
                pull_ollama_model(model, status_callback=_status)
                self._safe_after(lambda: self._ollama_pull_btn.configure(
                    state="disabled", text="✅ 已安裝", fg_color="gray"))
                self._safe_after(self._refresh_ollama_models)
            except Exception as e:
                self._safe_after(lambda err=str(e): self._ollama_status_label.configure(
                    text=f"❌ {err}", text_color="#e74c3c"))
                self._safe_after(lambda: self._ollama_pull_btn.configure(
                    state="normal", text="⬇ 下載模型", fg_color=("#3B8ED0", "#1F6AA5")))

        threading.Thread(target=_do, daemon=True).start()

    def _on_whisper_src_change(self, value: str):
        """顯示/隱藏本地 Whisper 設定"""
        is_local = value == "本地 (GPU)"
        state = "normal" if is_local else "disabled"
        self._local_whisper_menu.configure(state=state)
        if is_local:
            self._whisper_model_status_frame.grid()
            self._check_whisper_model()
        else:
            self._whisper_model_status_label.configure(text="")
            self._whisper_download_btn.configure(state="disabled", text="⬇ 下載模型")

    def _on_provider_change(self, provider: str):
        """切換潤稿 API 提供者時更新模型選單"""
        is_local = provider == "本地 (Ollama/LM Studio)"
        if is_local:
            self._model_menu.configure(state="disabled")
            self._custom_model_entry.configure(state="normal")
            self._local_url_entry.configure(state="normal")
            self._ollama_status_frame.grid()
            self.after(200, self._refresh_ollama_models)
        elif provider == "openrouter":
            self._model_menu.configure(state="normal", values=OPENROUTER_MODELS)
            self._custom_model_entry.configure(state="disabled")
            self._local_url_entry.configure(state="disabled")
            self._ollama_status_label.configure(text="")
            self._ollama_pull_btn.configure(state="disabled")
            if self._model_var.get() not in OPENROUTER_MODELS:
                self._model_var.set(OPENROUTER_MODELS[0])
        else:
            self._model_menu.configure(state="normal", values=OPENAI_MODELS)
            self._custom_model_entry.configure(state="disabled")
            self._local_url_entry.configure(state="disabled")
            self._ollama_status_label.configure(text="")
            self._ollama_pull_btn.configure(state="disabled")
            if self._model_var.get() not in OPENAI_MODELS:
                self._model_var.set(OPENAI_MODELS[0])

    def _save_settings(self):
        old_hotkey = self._config.get("hotkey", "ctrl+shift+space")
        cfg = self._get_current_config()
        try:
            save_config(cfg)
        except OSError as e:
            self._show_error(f"設定儲存失敗：{e}")
            return
        except Exception as e:
            self._show_error(f"設定儲存失敗：{e}")
            return
        if self._hotkey_manager:
            self._hotkey_manager.update_hotkey(cfg.get("hotkey", "ctrl+shift+space"))
            error = self._hotkey_manager.get_last_error()
            if error:
                cfg["hotkey"] = old_hotkey
                try:
                    save_config(cfg)
                except Exception:
                    pass
                self._config = cfg
                self._show_error(f"熱鍵更新失敗，已回復舊熱鍵：{error}")
                return
        self._config = cfg
        self._update_compute_card(cfg)
        self._show_toast("設定已儲存")

    def _get_current_config(self) -> dict:
        return {
            **self._config,
            "template": self._key_of(TEMPLATE_LABELS, self._template_var.get()) or "general",
            "transcription_language": LANGUAGE_OPTIONS.get(self._lang_var.get(), "auto"),
            "output_language": OUTPUT_LANG_OPTIONS.get(self._out_lang_var.get(), "original"),
            "auto_switch_template": self._auto_switch_var.get(),
            "auto_paste": self._auto_paste_var.get(),
            "auto_clear": self._auto_clear_var.get(),
            "auto_translate": self._auto_translate_var.get(),
            "default_translate_lang": self._default_translate_lang_var.get(),
            "whisper_source": self._whisper_src_var.get(),
            "whisper_local_model": self._whisper_model_var.get(),
            "whisper_device": self._whisper_device_var.get(),
            "api_provider": self._provider_var.get(),
            "openai_api_key": self._openai_key_entry.get().strip(),
            "openrouter_api_key": self._openrouter_key_entry.get().strip(),
            "local_api_url": self._local_url_entry.get().strip(),
            "local_model_name": self._get_selected_ollama_model(),
            "polish_model": self._model_var.get(),
        }

    # ─── 歷史導覽 ─────────────────────────────────────────────────

    def _refresh_history_cache(self):
        """潤稿完成後刷新快取，並重置導覽指標到最新一筆"""
        self._history_records = load_history()
        self._history_idx = -1
        total = len(self._history_records)
        self._nav_label.configure(text=f"共 {total} 筆" if total else "無紀錄")

    def _nav_prev(self):
        """往更舊一筆（首次按下時先載入清單）"""
        if not self._history_records:
            self._refresh_history_cache()
        if not self._history_records:
            return
        next_idx = self._history_idx + 1
        if next_idx < len(self._history_records):
            self._history_idx = next_idx
            self._show_history_record(self._history_idx)

    def _nav_next(self):
        """往更新一筆"""
        if self._history_idx <= 0:
            return
        self._history_idx -= 1
        self._show_history_record(self._history_idx)

    def _show_history_record(self, idx: int):
        r = self._history_records[idx]
        self._set_original(r.get("original", ""))
        self._set_polished(r.get("polished", ""))
        self._set_translate_result("")
        total = len(self._history_records)
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(r["timestamp"]).strftime("%m/%d %H:%M")
        except Exception:
            ts = r.get("timestamp", "")
        self._nav_label.configure(text=f"第 {idx + 1} / {total} 筆  ·  {ts}")

    def _show_toast(self, msg: str):
        toast = ctk.CTkLabel(self, text=msg, fg_color="#2ecc71", text_color="white",
                             corner_radius=6, font=ctk.CTkFont(size=13))
        toast.place(relx=0.5, rely=0.97, anchor="s")
        self.after(2000, toast.destroy)

    @staticmethod
    def _label_of_template(key: str) -> str:
        return TEMPLATE_LABELS.get(key, "通用")

    @staticmethod
    def _key_of(mapping: dict, value: str) -> str:
        for k, v in mapping.items():
            if v == value:
                return k
        return list(mapping.keys())[0]


class InputSettingsPage(ctk.CTkFrame):
    """獨立的輸入設定頁，沿用 HomePage 的設定邏輯與儲存流程。"""

    def __init__(self, master, home_page: HomePage, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._home_page = home_page
        self._build_ui()

    def _build_ui(self):
        host = ctk.CTkFrame(self, fg_color="transparent")
        host.pack(fill="both", expand=True)
        host.rowconfigure(0, weight=1)
        host.columnconfigure(0, weight=1)
        self._home_page._build_settings(host)

