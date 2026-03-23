"""主視窗 — 首頁（儀表板 + 設定）"""
import importlib.util
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from time import perf_counter
import customtkinter as ctk

from app.core.recorder import AudioRecorder, RecorderStartError, RecordingState
from app.core.asr_router import transcribe_audio
from app.core.tts_router import synthesize as synthesize_tts
from app.core.lyrics_mode import split_lyrics_lines, build_lrc_from_segments, build_srt_from_segments
from app.core import qwen3_asr, qwen3_tts, model_downloader
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
ASR_ENGINES = ["qwen3_asr"]
QWEN3_ASR_MODELS = list(qwen3_asr.QWEN3_ASR_MODELS)
QWEN3_ALIGNER_MODELS = list(qwen3_asr.QWEN3_ALIGNER_MODELS)
QWEN3_TTS_MODELS = list(qwen3_tts.QWEN3_TTS_MODELS)
ASR_DEVICE_OPTIONS = ["cuda", "cpu"]
TTS_ENGINES = ["qwen3_tts"]
TTS_SPEAKERS = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
TTS_LANGUAGES = ["Chinese", "English", "Japanese", "Korean"]
TRANSLATION_MODE_LABELS = {
    "本地優先（失敗可回退雲端）": "local_first",
    "僅本地（Ollama）": "local_only",
    "僅雲端": "cloud_only",
}

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
    @staticmethod
    def _with_runtime_defaults(cfg: dict) -> dict:
        resolved = {
            **cfg,
            "asr_engine": "qwen3_asr",
            "asr_qwen3_model": cfg.get("asr_qwen3_model", "Qwen/Qwen3-ASR-0.6B"),
            "asr_qwen3_aligner_model": cfg.get("asr_qwen3_aligner_model", "Qwen/Qwen3-ForcedAligner-0.6B"),
            "asr_device": cfg.get("asr_device", "cuda"),
            "tts_engine": "qwen3_tts",
            "tts_device": cfg.get("tts_device", "cuda"),
            "tts_qwen3_model": cfg.get("tts_qwen3_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            "tts_speaker": cfg.get("tts_speaker", "Vivian"),
            "tts_language": cfg.get("tts_language", "Chinese"),
            "tts_instruct": cfg.get("tts_instruct", ""),
            "translation_mode": cfg.get("translation_mode", "local_first"),
            "translation_fallback_enabled": cfg.get("translation_fallback_enabled", True),
            "lyrics_mode": cfg.get("lyrics_mode", cfg.get("lyrics_mode_enabled", False)),
        }
        return resolved
    def __init__(self, master, hotkey_manager=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._config = self._with_runtime_defaults(load_config())
        self._ensure_engine_setting_vars()
        self._recorder = AudioRecorder()
        self._state = RecordingState.IDLE
        self._hotkey_manager = hotkey_manager
        self._record_start_time: float = 0.0
        self._timer_job: str | None = None
        self._history_records: list[dict] = []
        self._history_idx: int = -1
        self._step_times: dict[str, float | None] = {"asr": None, "polish": None, "translate": None}
        self._last_error_message: str = ""
        self._build_ui()
        self._render_step_times()
        self._update_stats()

        if self._hotkey_manager:
            self._hotkey_manager.start(
                self._config.get("hotkey", "ctrl+shift+space"),
                self._toggle_recording,
            )
            error = self._hotkey_manager.get_last_error()
            if error:
                self._safe_after(lambda err=error: self._show_error(f"熱鍵註冊失敗：{err}"))

    def _ensure_engine_setting_vars(self) -> None:
        if hasattr(self, "_asr_engine_var"):
            return
        self._asr_engine_var = ctk.StringVar(value="qwen3_asr")
        self._asr_qwen3_model_var = ctk.StringVar(value=self._config.get("asr_qwen3_model", QWEN3_ASR_MODELS[0]))
        self._asr_qwen3_aligner_var = ctk.StringVar(
            value=self._config.get("asr_qwen3_aligner_model", QWEN3_ALIGNER_MODELS[0])
        )
        self._asr_device_var = ctk.StringVar(value=self._config.get("asr_device", "cuda"))

        self._tts_engine_var = ctk.StringVar(value="qwen3_tts")
        self._tts_device_var = ctk.StringVar(value=self._config.get("tts_device", "cuda"))
        self._tts_qwen3_model_var = ctk.StringVar(value=self._config.get("tts_qwen3_model", QWEN3_TTS_MODELS[0]))
        self._tts_speaker_var = ctk.StringVar(value=self._config.get("tts_speaker", TTS_SPEAKERS[0]))
        self._tts_language_var = ctk.StringVar(value=self._config.get("tts_language", "Chinese"))
        self._tts_instruct_var = ctk.StringVar(value=self._config.get("tts_instruct", ""))
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
        self._compute_card = self._make_stat_card(stats_row, "運算裝置", "-", 3, sub="ASR 轉寫來源")
        self._update_compute_card(self._config)

        # 主體（首頁只保留輸出區，設定移到獨立頁）
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self._build_output(body, column=0)

        # ── Model status bar ────────────────────────
        self._model_status_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        )
        self._model_status_label.pack(fill="x", padx=2, pady=(6, 0))

    def update_model_status(self, text: str, color: str = "gray") -> None:
        """Update model status bar. Must be called from the main thread."""
        self._model_status_label.configure(text=text, text_color=color)

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

        # 翻譯模式
        ctk.CTkLabel(frame, text="翻譯模式", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._translation_mode_var = ctk.StringVar(
            value=self._key_of(TRANSLATION_MODE_LABELS, self._config.get("translation_mode", "local_first"))
        )
        ctk.CTkOptionMenu(frame, values=list(TRANSLATION_MODE_LABELS.keys()), variable=self._translation_mode_var).grid(
            row=row, column=1, padx=14, pady=4, sticky="ew"
        ); row += 1

        # 翻譯回退
        ctk.CTkLabel(frame, text="翻譯失敗回退雲端", font=ctk.CTkFont(size=13)).grid(row=row, column=0, padx=14, pady=4, sticky="w")
        self._translation_fallback_var = ctk.BooleanVar(value=self._config.get("translation_fallback_enabled", True))
        ctk.CTkCheckBox(frame, text="", variable=self._translation_fallback_var).grid(row=row, column=1, padx=14, pady=4, sticky="w"); row += 1

        ctk.CTkLabel(
            frame,
            text="ASR / TTS 相關設定已拆到獨立分頁：請至「ASR 設定」與「TTS 設定」。",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            wraplength=520,
            justify="left",
        ).grid(row=row, column=0, columnspan=2, padx=14, pady=(10, 8), sticky="w")
        row += 1

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

        ctk.CTkButton(
            toolbar,
            text="複製錯誤",
            width=86,
            command=self._copy_last_error,
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

        self._tts_btn = ctk.CTkButton(
            toolbar,
            text="TTS 合成",
            width=90,
            fg_color="#2d98da",
            hover_color="#1b7cb7",
            command=self._synthesize_current_text,
        )
        self._tts_btn.pack(side="left", padx=(0, 4))

        self._lyrics_mode_var = ctk.BooleanVar(
            value=self._config.get("lyrics_mode", self._config.get("lyrics_mode_enabled", False))
        )
        ctk.CTkCheckBox(
            toolbar,
            text="歌詞模式",
            variable=self._lyrics_mode_var,
            command=self._on_lyrics_mode_toggle,
        ).pack(side="left", padx=(4, 4))

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

        self._step_timing_label = ctk.CTkLabel(
            nav_bar,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._step_timing_label.pack(side="right", padx=(6, 0))
        self._render_step_times()

        # Lyrics / LRC 區塊
        self._lyrics_frame = ctk.CTkFrame(frame, corner_radius=8)
        self._lyrics_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=8, pady=(0, 8))
        self._lyrics_frame.columnconfigure((0, 1), weight=1)
        self._lyrics_frame.rowconfigure(1, weight=1)

        lyric_hdr = ctk.CTkFrame(self._lyrics_frame, fg_color="transparent")
        lyric_hdr.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        lyric_hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(lyric_hdr, text="歌詞（可編輯）", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray").grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkButton(
            lyric_hdr, text="複製 TXT", width=78, height=24, font=ctk.CTkFont(size=12),
            command=lambda: self._copy_text(self._lyrics_text.get("1.0", "end").strip())
        ).grid(row=0, column=1, padx=(4, 0))
        ctk.CTkButton(
            lyric_hdr, text="匯出 TXT", width=78, height=24, font=ctk.CTkFont(size=12),
            command=self._export_lyrics_txt
        ).grid(row=0, column=2, padx=(4, 0))

        lrc_hdr = ctk.CTkFrame(self._lyrics_frame, fg_color="transparent")
        lrc_hdr.grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 4))
        lrc_hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(lrc_hdr, text="LRC（可編輯）", font=ctk.CTkFont(size=12, weight="bold"), text_color="gray").grid(
            row=0, column=0, sticky="w"
        )
        ctk.CTkButton(
            lrc_hdr, text="複製 LRC", width=78, height=24, font=ctk.CTkFont(size=12),
            command=lambda: self._copy_text(self._lrc_text.get("1.0", "end").strip())
        ).grid(row=0, column=1, padx=(4, 0))
        ctk.CTkButton(
            lrc_hdr, text="匯出 LRC", width=78, height=24, font=ctk.CTkFont(size=12),
            command=self._export_lrc
        ).grid(row=0, column=2, padx=(4, 0))

        self._lyrics_text = ctk.CTkTextbox(self._lyrics_frame, font=ctk.CTkFont(size=13), wrap="word", height=120)
        self._lyrics_text.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))

        self._lrc_text = ctk.CTkTextbox(self._lyrics_frame, font=ctk.CTkFont(size=13), wrap="none", height=120)
        self._lrc_text.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))
        self._on_lyrics_mode_toggle()

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

    def _copy_last_error(self):
        msg = (self._last_error_message or "").strip()
        if not msg:
            self._show_toast("目前沒有可複製的錯誤訊息")
            return
        if not copy_to_clipboard(msg):
            self._set_translate_result("❌ 複製失敗：請安裝 pyperclip")
            return
        self._show_toast("錯誤訊息已複製")

    def _save_text_to_file(self, text: str, title: str, initialfile: str, ext: str) -> None:
        if not text.strip():
            self._show_error("沒有可匯出的內容")
            return
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=ext,
            filetypes=[("文字檔", "*.txt"), ("LRC", "*.lrc"), ("所有檔案", "*.*")],
            initialfile=initialfile,
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            self._show_error(f"匯出失敗：{e}")
            return
        self._show_toast(f"已匯出：{path}")

    def _export_lyrics_txt(self):
        content = self._lyrics_text.get("1.0", "end").strip()
        self._save_text_to_file(content, title="匯出歌詞 TXT", initialfile="lyrics.txt", ext=".txt")

    def _export_lrc(self):
        content = self._lrc_text.get("1.0", "end").strip()
        self._save_text_to_file(content, title="匯出 LRC", initialfile="lyrics.lrc", ext=".lrc")

    def _get_text_for_tts(self) -> str:
        focus_widget = self.focus_get()
        for tb in (
            getattr(self, "_lyrics_text", None),
            getattr(self, "_original_text", None),
            getattr(self, "_polished_text", None),
            getattr(self, "_translate_result", None),
            getattr(self, "_lrc_text", None),
        ):
            inner = getattr(tb, "_textbox", None) if tb is not None else None
            if tb is not None and focus_widget in (tb, inner):
                return tb.get("1.0", "end").strip()
        for tb in (
            getattr(self, "_lyrics_text", None),
            getattr(self, "_polished_text", None),
            getattr(self, "_original_text", None),
            getattr(self, "_translate_result", None),
        ):
            if tb is not None:
                value = tb.get("1.0", "end").strip()
                if value:
                    return value
        return ""

    def _synthesize_current_text(self):
        if not self._can_start_action("TTS 合成"):
            return
        text = self._get_text_for_tts()
        if not text:
            self._show_error("沒有可合成的文字")
            return

        cfg_snapshot = self._get_current_config()
        self._tts_btn.configure(state="disabled", text="合成中...")

        def _status(msg: str):
            self._safe_after(lambda m=msg: self._show_toast(m))

        def _do(cfg: dict, src_text: str):
            try:
                result = synthesize_tts(src_text, cfg, status_callback=_status)
                self._safe_after(lambda p=result.output_path: self._show_toast(f"TTS 已輸出：{p}"))
            except Exception as e:
                self._safe_after(lambda err=str(e): self._show_error(f"TTS 失敗：{err}"))
            finally:
                self._safe_after(lambda: self._tts_btn.configure(state="normal", text="TTS 合成"))

        threading.Thread(target=_do, args=(cfg_snapshot, text), daemon=True).start()

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
        self._clear_step_time("polish")
        self._clear_step_time("translate")
        self._set_state(RecordingState.POLISHING)

        def _do(cfg: dict, original_text: str):
            try:
                api_key, base_url, model = self._resolve_polish_api(cfg)
                polish_started_at = perf_counter()
                try:
                    polished = polish(
                        original_text,
                        api_key=api_key,
                        template=cfg.get("template", "general"),
                        output_language=cfg.get("output_language", "original"),
                        model=model,
                        base_url=base_url,
                    )
                finally:
                    elapsed = perf_counter() - polish_started_at
                    self._safe_after(lambda sec=elapsed: self._set_step_time("polish", sec))
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

    def _resolve_cloud_translation_api(self, cfg: dict) -> tuple[str, str | None, str]:
        provider = str(cfg.get("api_provider", "openai"))
        openrouter_key = cfg.get("openrouter_api_key", "").strip()
        openai_key = cfg.get("openai_api_key", "").strip()
        configured_model = str(cfg.get("polish_model", "") or "")

        openrouter_model = configured_model if "/" in configured_model else "openai/gpt-4o-mini"
        openai_model = configured_model if configured_model and "/" not in configured_model else "gpt-4o-mini"

        if provider == "openrouter" and openrouter_key:
            return openrouter_key, "https://openrouter.ai/api/v1", openrouter_model
        if provider == "openai" and openai_key:
            return openai_key, None, openai_model

        if openrouter_key:
            return openrouter_key, "https://openrouter.ai/api/v1", openrouter_model
        if openai_key:
            return openai_key, None, openai_model
        raise ValueError("未設定可用的雲端翻譯 API Key（OpenAI 或 OpenRouter）")

    def _translate_local_ollama(self, cfg: dict, text: str, target_lang: str) -> str:
        base_url = cfg.get("local_api_url", "http://localhost:11434/v1").strip() or "http://localhost:11434/v1"
        api_base = base_url.replace("/v1", "").rstrip("/")
        model = cfg.get("local_model_name", "").strip()
        if not model:
            raise ValueError("未設定本地翻譯模型（local_model_name）")
        if not is_ollama_running(api_base):
            raise RuntimeError("本地翻譯失敗：Ollama 未啟動")
        if not is_ollama_model_available(model, api_base):
            raise RuntimeError(f"本地翻譯失敗：模型未安裝（{model}）")
        return translate(
            text,
            target_lang=target_lang,
            api_key="ollama",
            model=model,
            base_url=base_url,
        )

    def _translate_cloud(self, cfg: dict, text: str, target_lang: str) -> str:
        api_key, base_url, model = self._resolve_cloud_translation_api(cfg)
        return translate(
            text,
            target_lang=target_lang,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )

    def _translate_with_policy(self, cfg: dict, text: str, target_lang: str) -> str:
        mode = str(cfg.get("translation_mode", "local_first") or "local_first")
        allow_fallback = bool(cfg.get("translation_fallback_enabled", True))

        if mode == "cloud_only":
            return self._translate_cloud(cfg, text, target_lang)
        if mode == "local_only":
            return self._translate_local_ollama(cfg, text, target_lang)

        try:
            return self._translate_local_ollama(cfg, text, target_lang)
        except Exception as local_err:
            if not allow_fallback:
                raise
            try:
                return self._translate_cloud(cfg, text, target_lang)
            except Exception as cloud_err:
                raise RuntimeError(f"本地翻譯失敗：{local_err}；雲端回退也失敗：{cloud_err}") from cloud_err

    def _translate_in_worker(self, cfg: dict, original_text: str, polished_text: str):
        target = self._translate_lang_var.get()
        src = self._translate_src_var.get()
        text = polished_text if src == "整理後" else original_text
        if not text.strip():
            return
        self._safe_after(lambda: self._clear_step_time("translate"))
        self._safe_after(lambda: self._set_state(RecordingState.TRANSLATING))
        self._safe_after(lambda t=target: self._set_translate_result(f"翻譯中（{t}）..."))
        started_at = perf_counter()
        try:
            result = self._translate_with_policy(cfg, text, target)
            self._safe_after(lambda r=result: self._set_translate_result(r))
        except Exception as e:
            self._safe_after(lambda err=str(e): self._set_translate_result(f"❌ {err}"))
        finally:
            elapsed = perf_counter() - started_at
            self._safe_after(lambda sec=elapsed: self._set_step_time("translate", sec))

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
        self._clear_step_time("translate")
        self._set_state(RecordingState.TRANSLATING)
        self._set_translate_result(f"翻譯中（{target}）...")

        def _do(cfg: dict, content_text: str, target_lang: str):
            started_at = perf_counter()
            try:
                result = self._translate_with_policy(cfg, content_text, target_lang)
                self._safe_after(lambda r=result: self._set_translate_result(r))
            except Exception as e:
                self._safe_after(lambda err=str(e): self._set_translate_result(f"❌ {err}"))
            finally:
                elapsed = perf_counter() - started_at
                self._safe_after(lambda sec=elapsed: self._set_step_time("translate", sec))
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
        self._reset_step_times()
        if cfg_snapshot.get("auto_clear", True):
            self._set_original("")
            self._set_lyrics_text("")
            self._set_lrc_text("")
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
                audio_payload = self._recorder.stop(as_wav=True)
                if not audio_payload:
                    self._safe_after(lambda: self._show_error("未錄到音訊，請再試一次"))
                    return

                lyrics_mode = bool(cfg.get("lyrics_mode", False))
                asr_started_at = perf_counter()
                try:
                    asr_result = transcribe_audio(audio_payload, cfg, need_segments=lyrics_mode)
                finally:
                    elapsed = perf_counter() - asr_started_at
                    self._safe_after(lambda sec=elapsed: self._set_step_time("asr", sec))
                original = apply_dictionary(asr_result.text)
                self._safe_after(lambda t=original: self._set_original(t))

                if lyrics_mode:
                    lyric_lines = split_lyrics_lines(original)
                    self._safe_after(lambda t="\n".join(lyric_lines): self._set_lyrics_text(t))
                    seg_rows = [
                        {"start": s.start, "end": s.end, "text": s.text}
                        for s in asr_result.segments
                    ]
                    lrc_text = build_lrc_from_segments(seg_rows, fallback_lines=lyric_lines)
                    self._safe_after(lambda t=lrc_text: self._set_lrc_text(t))

                # 潤稿
                self._safe_after(lambda: self._set_state(RecordingState.POLISHING))
                polish_key, polish_base_url, model = self._resolve_polish_api(cfg)

                if not polish_key:
                    self._safe_after(lambda: self._show_error("請先設定 API Key"))
                    return

                polish_started_at = perf_counter()
                try:
                    polished = polish(
                        original,
                        api_key=polish_key,
                        template=cfg.get("template", "general"),
                        output_language=cfg.get("output_language", "original"),
                        model=model,
                        base_url=polish_base_url,
                    )
                finally:
                    elapsed = perf_counter() - polish_started_at
                    self._safe_after(lambda sec=elapsed: self._set_step_time("polish", sec))
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

    @staticmethod
    def _format_step_time(seconds: float | None) -> str:
        if seconds is None:
            return "--"
        return f"{seconds:.2f}s"

    def _render_step_times(self):
        label = getattr(self, "_step_timing_label", None)
        if label is None:
            return
        label.configure(
            text=(
                f"辨識: {self._format_step_time(self._step_times.get('asr'))}  |  "
                f"潤稿: {self._format_step_time(self._step_times.get('polish'))}  |  "
                f"翻譯: {self._format_step_time(self._step_times.get('translate'))}"
            )
        )

    def _reset_step_times(self):
        self._step_times = {"asr": None, "polish": None, "translate": None}
        self._render_step_times()

    def _clear_step_time(self, step: str):
        if step not in self._step_times:
            return
        self._step_times[step] = None
        self._render_step_times()

    def _set_step_time(self, step: str, seconds: float):
        if step not in self._step_times:
            return
        self._step_times[step] = max(0.0, float(seconds))
        self._render_step_times()

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
        for btn_name in ("_polish_btn", "_translate_btn", "_save_polished_btn", "_tts_btn"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.configure(state=action_state)

    def _set_original(self, text: str):
        self._original_text.delete("1.0", "end")
        self._original_text.insert("1.0", text)

    def _set_lyrics_text(self, text: str):
        self._lyrics_text.delete("1.0", "end")
        self._lyrics_text.insert("1.0", text)

    def _set_lrc_text(self, text: str):
        self._lrc_text.delete("1.0", "end")
        self._lrc_text.insert("1.0", text)

    def _set_polished(self, text: str):
        self._polished_text.configure(state="normal")
        self._polished_text.delete("1.0", "end")
        self._polished_text.insert("1.0", text)
        self._polished_text.configure(state="disabled")
        if isinstance(text, str) and text.strip().startswith("❌"):
            self._last_error_message = text.strip().lstrip("❌").strip()

    def _set_translate_result(self, text: str):
        self._translate_result.configure(state="normal")
        self._translate_result.delete("1.0", "end")
        self._translate_result.insert("1.0", text)
        self._translate_result.configure(state="disabled")
        if isinstance(text, str) and text.strip().startswith("❌"):
            self._last_error_message = text.strip().lstrip("❌").strip()

    def _show_error(self, msg: str):
        self._last_error_message = str(msg).strip()
        self._set_state(RecordingState.ERROR)
        self._set_polished(f"❌ 錯誤：{msg}")
        self._safe_after(lambda: self._set_state(RecordingState.IDLE), delay_ms=3000)

    def _update_stats(self):
        self._words_card.configure(text=str(get_total_word_count()))
        self._count_card.configure(text=str(get_recording_count()))
        self._update_compute_card(self._config)

    def _update_compute_card(self, cfg: dict):
        device = str(cfg.get("asr_device", "cuda")).lower()
        label = "GPU" if device == "cuda" else "CPU"
        sub = f"Qwen3-ASR（{device}）"
        self._compute_card.configure(text=label)
        sub_label = getattr(self._compute_card, "_sub_label", None)
        if sub_label is not None:
            sub_label.configure(text=sub)
    # ─── 模型檢查與下載 ──────────────────────────────────────────

    def _set_model_download_status(self, message: str, color: str = "gray"):
        label = getattr(self, "_model_download_status_label", None)
        if label is not None:
            label.configure(text=message, text_color=color)

    def _set_download_buttons_state(self, state: str):
        for name in ("_download_asr_btn", "_download_tts_btn", "_download_all_btn"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.configure(state=state)

    def _on_asr_engine_change(self, engine: str):
        if engine != "qwen3_asr":
            self._asr_engine_var.set("qwen3_asr")
        self._asr_qwen3_model_menu.configure(state="normal")
        self._asr_qwen3_aligner_menu.configure(state="normal")

    def _on_tts_engine_change(self, engine: str):
        if engine != "qwen3_tts":
            self._tts_engine_var.set("qwen3_tts")
        self._tts_qwen3_model_menu.configure(state="normal")
    def _on_lyrics_mode_toggle(self):
        show = bool(self._lyrics_mode_var.get())
        if show:
            self._lyrics_frame.grid()
        else:
            self._lyrics_frame.grid_remove()

    def _download_current_asr_model(self):
        cfg = self._get_current_config()
        self._set_download_buttons_state("disabled")
        self._set_model_download_status("下載目前 ASR 模型中...")

        def _status(msg: str):
            self._safe_after(lambda m=msg: self._set_model_download_status(m))

        def _do():
            try:
                model_id = cfg.get("asr_qwen3_model", QWEN3_ASR_MODELS[0])
                aligner_id = cfg.get("asr_qwen3_aligner_model", QWEN3_ALIGNER_MODELS[0])
                qwen3_asr.download_repo(model_id, status_callback=_status)
                if aligner_id:
                    qwen3_asr.download_repo(aligner_id, status_callback=_status)
                self._safe_after(lambda: self._set_model_download_status("ASR 模型下載完成", "#2ecc71"))
            except Exception as e:
                self._safe_after(lambda err=str(e): self._set_model_download_status(f"ASR 下載失敗：{err}", "#e74c3c"))
            finally:
                self._safe_after(lambda: self._set_download_buttons_state("normal"))

        threading.Thread(target=_do, daemon=True).start()
    def _download_current_tts_model(self):
        cfg = self._get_current_config()
        self._set_download_buttons_state("disabled")
        self._set_model_download_status("下載目前 TTS 模型中...")

        def _status(msg: str):
            self._safe_after(lambda m=msg: self._set_model_download_status(m))

        def _do():
            try:
                qwen3_tts.download_repo("Qwen/Qwen3-TTS-Tokenizer-12Hz", status_callback=_status)
                model_id = cfg.get("tts_qwen3_model", QWEN3_TTS_MODELS[0])
                qwen3_tts.download_repo(model_id, status_callback=_status)
                self._safe_after(lambda: self._set_model_download_status("TTS 模型下載完成", "#2ecc71"))
            except Exception as e:
                self._safe_after(lambda err=str(e): self._set_model_download_status(f"TTS 下載失敗：{err}", "#e74c3c"))
            finally:
                self._safe_after(lambda: self._set_download_buttons_state("normal"))

        threading.Thread(target=_do, daemon=True).start()
    def _download_all_models(self):
        self._set_download_buttons_state("disabled")
        self._set_model_download_status("開始下載全部模型...")

        def _status(msg: str):
            self._safe_after(lambda m=msg: self._set_model_download_status(m))

        def _do():
            try:
                model_downloader.download_all_models(status_callback=_status)
                self._safe_after(lambda: self._set_model_download_status("全部模型下載完成", "#2ecc71"))
            except Exception as e:
                self._safe_after(lambda err=str(e): self._set_model_download_status(f"全量下載失敗：{err}", "#e74c3c"))
            finally:
                self._safe_after(lambda: self._set_download_buttons_state("normal"))

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
                self._config = self._with_runtime_defaults(cfg)
                self._show_error(f"熱鍵更新失敗，已回復舊熱鍵：{error}")
                return
        self._config = self._with_runtime_defaults(cfg)
        self._update_compute_card(self._config)
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
            "translation_mode": TRANSLATION_MODE_LABELS.get(self._translation_mode_var.get(), "local_first"),
            "translation_fallback_enabled": self._translation_fallback_var.get(),
            "lyrics_mode": self._lyrics_mode_var.get(),
            "lyrics_mode_enabled": self._lyrics_mode_var.get(),
            "asr_engine": "qwen3_asr",
            "asr_qwen3_model": self._asr_qwen3_model_var.get(),
            "asr_qwen3_aligner_model": self._asr_qwen3_aligner_var.get(),
            "asr_device": self._asr_device_var.get(),
            "tts_engine": "qwen3_tts",
            "tts_device": self._tts_device_var.get(),
            "tts_qwen3_model": self._tts_qwen3_model_var.get(),
            "tts_speaker": self._tts_speaker_var.get(),
            "tts_language": self._tts_language_var.get(),
            "tts_instruct": self._tts_instruct_var.get(),
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
        original = r.get("original", "")
        self._set_original(original)
        self._set_lyrics_text("\n".join(split_lyrics_lines(original)))
        self._set_lrc_text("")
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


class AsrSettingsPage(ctk.CTkFrame):
    """ASR 設定獨立頁。共用 HomePage 的設定變數與儲存流程。"""

    def __init__(self, master, home_page: HomePage, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._home_page = home_page
        self._build_ui()

    def _bind_download_widgets(self) -> None:
        self._home_page._download_asr_btn = self._download_asr_btn
        self._home_page._download_all_btn = self._download_all_btn
        self._home_page._model_download_status_label = self._model_download_status_label

    def _download_asr(self):
        self._bind_download_widgets()
        self._home_page._download_current_asr_model()

    def _download_all(self):
        self._bind_download_widgets()
        self._home_page._download_all_models()

    def _build_ui(self):
        frame = ctk.CTkScrollableFrame(self, corner_radius=10)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        row = 0
        ctk.CTkLabel(frame, text="ASR 設定", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=row, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w"
        )
        row += 1

        ctk.CTkLabel(frame, text="ASR 引擎（固定）", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=4, sticky="w"
        )
        ctk.CTkOptionMenu(
            frame,
            values=ASR_ENGINES,
            variable=self._home_page._asr_engine_var,
            command=self._home_page._on_asr_engine_change,
        ).grid(row=row, column=1, padx=14, pady=4, sticky="ew")
        row += 1

        ctk.CTkLabel(frame, text="Qwen3 ASR 模型", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=4, sticky="w"
        )
        self._home_page._asr_qwen3_model_menu = ctk.CTkOptionMenu(
            frame, values=QWEN3_ASR_MODELS, variable=self._home_page._asr_qwen3_model_var
        )
        self._home_page._asr_qwen3_model_menu.grid(row=row, column=1, padx=14, pady=4, sticky="ew")
        row += 1

        ctk.CTkLabel(frame, text="Qwen3 對齊模型", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=4, sticky="w"
        )
        self._home_page._asr_qwen3_aligner_menu = ctk.CTkOptionMenu(
            frame, values=QWEN3_ALIGNER_MODELS, variable=self._home_page._asr_qwen3_aligner_var
        )
        self._home_page._asr_qwen3_aligner_menu.grid(row=row, column=1, padx=14, pady=4, sticky="ew")
        row += 1

        ctk.CTkLabel(frame, text="ASR 裝置", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=4, sticky="w"
        )
        ctk.CTkOptionMenu(
            frame, values=ASR_DEVICE_OPTIONS, variable=self._home_page._asr_device_var
        ).grid(row=row, column=1, padx=14, pady=4, sticky="ew")
        row += 1

        ctk.CTkLabel(frame, text="模型下載", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=(10, 4), sticky="w"
        )
        dl_row = ctk.CTkFrame(frame, fg_color="transparent")
        dl_row.grid(row=row, column=1, padx=14, pady=(10, 4), sticky="ew")
        dl_row.columnconfigure((0, 1), weight=1)
        self._download_asr_btn = ctk.CTkButton(dl_row, text="下載目前 ASR 模型", command=self._download_asr)
        self._download_asr_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self._download_all_btn = ctk.CTkButton(dl_row, text="下載全部模型", command=self._download_all)
        self._download_all_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        row += 1

        self._model_download_status_label = ctk.CTkLabel(frame, text="", text_color="gray", font=ctk.CTkFont(size=12))
        self._model_download_status_label.grid(row=row, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="w")
        row += 1

        ctk.CTkButton(frame, text="儲存設定", command=self._home_page._save_settings).grid(
            row=row, column=0, columnspan=2, padx=14, pady=14, sticky="ew"
        )

        self._home_page._on_asr_engine_change(self._home_page._asr_engine_var.get())


class TtsSettingsPage(ctk.CTkFrame):
    """TTS 設定獨立頁。共用 HomePage 的設定變數與儲存流程。"""

    def __init__(self, master, home_page: HomePage, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._home_page = home_page
        self._build_ui()

    def _bind_download_widgets(self) -> None:
        self._home_page._download_tts_btn = self._download_tts_btn
        self._home_page._download_all_btn = self._download_all_btn
        self._home_page._model_download_status_label = self._model_download_status_label

    def _download_tts(self):
        self._bind_download_widgets()
        self._home_page._download_current_tts_model()

    def _download_all(self):
        self._bind_download_widgets()
        self._home_page._download_all_models()

    def _build_ui(self):
        frame = ctk.CTkScrollableFrame(self, corner_radius=10)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        row = 0
        ctk.CTkLabel(frame, text="TTS 設定", font=ctk.CTkFont(size=16, weight="bold")).grid(
            row=row, column=0, columnspan=2, padx=14, pady=(14, 8), sticky="w"
        )
        row += 1

        ctk.CTkLabel(frame, text="TTS 引擎（固定）", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=4, sticky="w"
        )
        ctk.CTkOptionMenu(
            frame,
            values=TTS_ENGINES,
            variable=self._home_page._tts_engine_var,
            command=self._home_page._on_tts_engine_change,
        ).grid(row=row, column=1, padx=14, pady=4, sticky="ew")
        row += 1

        ctk.CTkLabel(frame, text="Qwen3 TTS 模型", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=4, sticky="w"
        )
        self._home_page._tts_qwen3_model_menu = ctk.CTkOptionMenu(
            frame, values=QWEN3_TTS_MODELS, variable=self._home_page._tts_qwen3_model_var
        )
        self._home_page._tts_qwen3_model_menu.grid(row=row, column=1, padx=14, pady=4, sticky="ew")
        row += 1

        ctk.CTkLabel(frame, text="TTS 裝置", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=4, sticky="w"
        )
        ctk.CTkOptionMenu(
            frame,
            values=ASR_DEVICE_OPTIONS,
            variable=self._home_page._tts_device_var,
        ).grid(row=row, column=1, padx=14, pady=4, sticky="ew")
        row += 1

        ctk.CTkLabel(frame, text="模型下載", font=ctk.CTkFont(size=13)).grid(
            row=row, column=0, padx=14, pady=(10, 4), sticky="w"
        )
        dl_row = ctk.CTkFrame(frame, fg_color="transparent")
        dl_row.grid(row=row, column=1, padx=14, pady=(10, 4), sticky="ew")
        dl_row.columnconfigure((0, 1), weight=1)
        self._download_tts_btn = ctk.CTkButton(dl_row, text="下載目前 TTS 模型", command=self._download_tts)
        self._download_tts_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self._download_all_btn = ctk.CTkButton(dl_row, text="下載全部模型", command=self._download_all)
        self._download_all_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        row += 1

        self._model_download_status_label = ctk.CTkLabel(frame, text="", text_color="gray", font=ctk.CTkFont(size=12))
        self._model_download_status_label.grid(row=row, column=0, columnspan=2, padx=14, pady=(0, 8), sticky="w")
        row += 1

        ctk.CTkButton(frame, text="儲存設定", command=self._home_page._save_settings).grid(
            row=row, column=0, columnspan=2, padx=14, pady=14, sticky="ew"
        )

        self._home_page._on_tts_engine_change(self._home_page._tts_engine_var.get())


class SettingsPage(ctk.CTkFrame):
    """整合一般/ASR/TTS 的設定頁。"""

    def __init__(self, master, home_page: HomePage, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._home_page = home_page
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(
            self,
            text="設定",
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(anchor="w", pady=(0, 10))

        tabs = ctk.CTkTabview(self)
        tabs.pack(fill="both", expand=True)
        tabs.add("一般設定")
        tabs.add("ASR設定")
        tabs.add("TTS設定")

        general_tab = tabs.tab("一般設定")
        asr_tab = tabs.tab("ASR設定")
        tts_tab = tabs.tab("TTS設定")
        for tab in (general_tab, asr_tab, tts_tab):
            tab.rowconfigure(0, weight=1)
            tab.columnconfigure(0, weight=1)

        InputSettingsPage(general_tab, home_page=self._home_page).grid(row=0, column=0, sticky="nsew")
        AsrSettingsPage(asr_tab, home_page=self._home_page).grid(row=0, column=0, sticky="nsew")
        TtsSettingsPage(tts_tab, home_page=self._home_page).grid(row=0, column=0, sticky="nsew")


class AsrAppPage(ctk.CTkFrame):
    """ASR 專用頁：錄音與辨識。"""

    def __init__(self, master, home_page: HomePage, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._home_page = home_page
        self._recorder = AudioRecorder()
        self._state = "idle"
        self._last_error_message = ""
        self._record_start_time: float = 0.0
        self._timer_job: str | None = None
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(
            self,
            text="ASR APP",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(
            self,
            text="僅做語音辨識（錄音 -> 文字），不進行潤稿與翻譯。",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(anchor="w", pady=(0, 14))

        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", pady=(0, 8))

        self._record_btn = ctk.CTkButton(
            toolbar,
            text="● 開始錄音",
            width=120,
            fg_color="#e74c3c",
            hover_color="#c0392b",
            command=self._toggle_recording,
        )
        self._record_btn.pack(side="left", padx=(0, 6))

        self._timer_label = ctk.CTkLabel(
            toolbar,
            text="",
            width=60,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#e67e22",
        )
        self._timer_label.pack(side="left", padx=(0, 8))

        self._video_btn = ctk.CTkButton(
            toolbar,
            text="影片辨識",
            width=100,
            command=self._choose_video_and_transcribe,
        )
        self._video_btn.pack(side="left", padx=(0, 8))

        self._lyrics_mode_var = ctk.BooleanVar(value=self._home_page._config.get("lyrics_mode", False))
        ctk.CTkCheckBox(
            toolbar,
            text="歌詞模式",
            variable=self._lyrics_mode_var,
            command=self._on_lyrics_toggle,
        ).pack(side="left", padx=(0, 8))

        self._srt_mode_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            toolbar,
            text="SRT字幕",
            variable=self._srt_mode_var,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            toolbar,
            text="清空",
            width=80,
            command=self._clear_output,
        ).pack(side="left")

        ctk.CTkButton(
            toolbar,
            text="複製錯誤",
            width=86,
            command=self._copy_last_error,
        ).pack(side="left", padx=(6, 0))

        yt_row = ctk.CTkFrame(self, fg_color="transparent")
        yt_row.pack(fill="x", pady=(0, 8))
        yt_row.columnconfigure(0, weight=1)

        self._youtube_url_var = ctk.StringVar(value="")
        self._youtube_entry = ctk.CTkEntry(
            yt_row,
            textvariable=self._youtube_url_var,
            placeholder_text="貼上 YouTube 連結（https://www.youtube.com/watch?v=...）",
        )
        self._youtube_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._youtube_entry.bind("<Return>", lambda _: self._download_youtube_and_transcribe())

        self._youtube_btn = ctk.CTkButton(
            yt_row,
            text="YouTube辨識",
            width=120,
            command=self._download_youtube_and_transcribe,
        )
        self._youtube_btn.grid(row=0, column=1, sticky="e")

        self._status_label = ctk.CTkLabel(self, text="待機中", font=ctk.CTkFont(size=13), text_color="gray")
        self._status_label.pack(anchor="w", pady=(0, 4))

        self._asr_time_label = ctk.CTkLabel(self, text="辨識耗時：--", font=ctk.CTkFont(size=13), text_color="gray")
        self._asr_time_label.pack(anchor="w", pady=(0, 10))

        self._lyrics_frame = ctk.CTkFrame(self, corner_radius=8)
        self._lyrics_frame.pack(fill="both", expand=False, pady=(0, 8))
        self._lyrics_frame.columnconfigure((0, 1), weight=1)
        self._lyrics_frame.rowconfigure(1, weight=1)

        lyric_hdr = ctk.CTkFrame(self._lyrics_frame, fg_color="transparent")
        lyric_hdr.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        lyric_hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            lyric_hdr,
            text="歌詞（可編輯）",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            lyric_hdr,
            text="複製 TXT",
            width=78,
            height=24,
            font=ctk.CTkFont(size=12),
            command=lambda: self._copy_text(self._lyrics_text.get("1.0", "end").strip()),
        ).grid(row=0, column=1, padx=(4, 0))

        lrc_hdr = ctk.CTkFrame(self._lyrics_frame, fg_color="transparent")
        lrc_hdr.grid(row=0, column=1, sticky="ew", padx=8, pady=(8, 4))
        lrc_hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            lrc_hdr,
            text="LRC（可編輯）",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            lrc_hdr,
            text="複製 LRC",
            width=78,
            height=24,
            font=ctk.CTkFont(size=12),
            command=lambda: self._copy_text(self._lrc_text.get("1.0", "end").strip()),
        ).grid(row=0, column=1, padx=(4, 0))

        self._lyrics_text = ctk.CTkTextbox(self._lyrics_frame, font=ctk.CTkFont(size=13), wrap="word", height=120)
        self._lyrics_text.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))

        self._lrc_text = ctk.CTkTextbox(self._lyrics_frame, font=ctk.CTkFont(size=13), wrap="none", height=120)
        self._lrc_text.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))

        out = ctk.CTkFrame(self, fg_color="transparent")
        out.pack(fill="both", expand=True)
        out.rowconfigure(1, weight=1)
        out.columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(out, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            hdr,
            text="辨識結果（可編輯）",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            hdr,
            text="複製",
            width=58,
            height=24,
            font=ctk.CTkFont(size=12),
            command=lambda: self._copy_text(self._original_text.get("1.0", "end").strip()),
        ).grid(row=0, column=1, padx=(6, 0))
        ctk.CTkButton(
            hdr,
            text="匯出 SRT",
            width=86,
            height=24,
            font=ctk.CTkFont(size=12),
            command=self._export_srt,
        ).grid(row=0, column=2, padx=(6, 0))

        self._original_text = ctk.CTkTextbox(out, font=ctk.CTkFont(size=13), wrap="word")
        self._original_text.grid(row=1, column=0, sticky="nsew")

        self._on_lyrics_toggle()

    def _safe_after(self, callback, delay_ms: int = 0):
        try:
            if delay_ms <= 0:
                self.after(0, callback)
            else:
                self.after(0, lambda: self.after(delay_ms, callback))
        except tk.TclError:
            return
        except Exception as exc:
            LOGGER.exception("ASR APP schedule callback failed: %s", exc)

    def _set_state(self, state: str):
        self._state = state
        if state == "recording":
            self._status_label.configure(text="錄音中...", text_color="#e67e22")
            self._record_btn.configure(text="■ 停止錄音", fg_color="#e67e22", hover_color="#ca6f1e")
            self._video_btn.configure(state="disabled")
            self._youtube_btn.configure(state="disabled")
        elif state == "transcribing":
            self._status_label.configure(text="辨識中...", text_color="#3498db")
            self._record_btn.configure(state="disabled")
            self._video_btn.configure(state="disabled")
            self._youtube_btn.configure(state="disabled")
        elif state == "error":
            self._status_label.configure(text="發生錯誤", text_color="#e74c3c")
            self._record_btn.configure(state="normal", text="● 開始錄音", fg_color="#e74c3c", hover_color="#c0392b")
            self._video_btn.configure(state="normal")
            self._youtube_btn.configure(state="normal")
        else:
            self._status_label.configure(text="待機中", text_color="gray")
            self._record_btn.configure(state="normal", text="● 開始錄音", fg_color="#e74c3c", hover_color="#c0392b")
            self._video_btn.configure(state="normal")
            self._youtube_btn.configure(state="normal")

    def _tick_timer(self):
        import time
        if self._state != "recording":
            self._timer_label.configure(text="")
            return
        elapsed = int(time.time() - self._record_start_time)
        m, s = divmod(elapsed, 60)
        self._timer_label.configure(text=f"{m:02d}:{s:02d}")
        self._timer_job = self.after(500, self._tick_timer)

    def _toggle_recording(self):
        if self._state == "idle":
            self._start_recording()
        elif self._state == "recording":
            self._stop_and_transcribe()

    def _start_recording(self):
        import time

        self._asr_time_label.configure(text="辨識耗時：--")
        try:
            self._recorder.start()
        except Exception as e:
            self._set_state("error")
            self._show_error(f"錄音啟動失敗：{e}")
            return

        self._set_state("recording")
        self._record_start_time = time.time()
        self._tick_timer()

    def _choose_video_and_transcribe(self):
        if self._state == "recording":
            self._show_error("錄音中，請先停止錄音再進行影片辨識")
            return
        if self._state == "transcribing":
            self._show_error("辨識中，請稍後再試")
            return

        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="選擇影片檔",
            filetypes=[
                ("影片檔", "*.mp4;*.mkv;*.mov;*.avi;*.wmv;*.webm;*.m4v"),
                ("所有檔案", "*.*"),
            ],
        )
        if not path:
            return

        cfg_snapshot = self._home_page._get_current_config()
        cfg_snapshot["lyrics_mode"] = bool(self._lyrics_mode_var.get())
        cfg_snapshot["srt_mode"] = bool(self._srt_mode_var.get())
        self._timer_label.configure(text="")
        self._set_state("transcribing")
        self._status_label.configure(text=f"影片處理中：{Path(path).name}", text_color="#3498db")

        threading.Thread(
            target=self._transcribe_video_file,
            args=(cfg_snapshot, path),
            daemon=True,
        ).start()

    @staticmethod
    def _is_youtube_url(url: str) -> bool:
        lowered = url.strip().lower()
        return "youtube.com/" in lowered or "youtu.be/" in lowered

    @staticmethod
    def _resolve_yt_dlp_command_prefix() -> list[str]:
        yt_dlp_bin = shutil.which("yt-dlp")
        if yt_dlp_bin:
            return [yt_dlp_bin]

        if importlib.util.find_spec("yt_dlp") is not None:
            return [sys.executable, "-m", "yt_dlp"]

        raise RuntimeError("YouTube 下載需要 yt-dlp（請先安裝：pip install yt-dlp）")

    def _download_youtube_and_transcribe(self):
        if self._state == "recording":
            self._show_error("錄音中，請先停止錄音再進行 YouTube 辨識")
            return
        if self._state == "transcribing":
            self._show_error("辨識中，請稍後再試")
            return

        url = self._youtube_url_var.get().strip()
        if not url:
            self._show_error("請先貼上 YouTube 連結")
            return
        if not self._is_youtube_url(url):
            self._show_error("僅支援 YouTube 連結（youtube.com / youtu.be）")
            return

        cfg_snapshot = self._home_page._get_current_config()
        cfg_snapshot["lyrics_mode"] = bool(self._lyrics_mode_var.get())
        cfg_snapshot["srt_mode"] = bool(self._srt_mode_var.get())
        self._timer_label.configure(text="")
        self._set_state("transcribing")
        self._status_label.configure(text="YouTube 下載中...", text_color="#3498db")

        threading.Thread(
            target=self._transcribe_youtube_url,
            args=(cfg_snapshot, url),
            daemon=True,
        ).start()

    def _transcribe_youtube_url(self, cfg: dict, url: str):
        try:
            audio_payload, source_name = self._download_audio_from_youtube(url)
            self._run_asr_pipeline(cfg, audio_payload, source_name=source_name)
        except Exception as e:
            self._safe_after(lambda err=str(e): self._show_error(err))

    def _transcribe_video_file(self, cfg: dict, video_path: str):
        try:
            audio_payload = self._extract_audio_from_video(video_path)
            self._run_asr_pipeline(cfg, audio_payload, source_name=Path(video_path).name)
        except Exception as e:
            self._safe_after(lambda err=str(e): self._show_error(err))

    @classmethod
    def _download_audio_from_youtube(cls, url: str) -> tuple[bytes, str]:
        if not cls._is_youtube_url(url):
            raise ValueError("非 YouTube 連結")

        cmd_prefix = cls._resolve_yt_dlp_command_prefix()
        if not shutil.which("ffmpeg"):
            raise RuntimeError("YouTube 辨識需要 ffmpeg（請安裝並加入 PATH）")

        title = "YouTube"
        title_cmd = [*cmd_prefix, "--no-playlist", "--print", "title", url]
        title_proc = subprocess.run(title_cmd, capture_output=True, text=True, check=False)
        if title_proc.returncode == 0:
            maybe_title = (title_proc.stdout or "").strip().splitlines()
            if maybe_title:
                title = maybe_title[0].strip() or title

        with tempfile.TemporaryDirectory(prefix="asr_youtube_") as td:
            out_tmpl = str(Path(td) / "youtube.%(ext)s")
            download_cmd = [*cmd_prefix, "--no-playlist", "-f", "bestaudio/best", "-o", out_tmpl, url]
            proc = subprocess.run(download_cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                detail = (proc.stderr or "").strip() or "yt-dlp 執行失敗"
                raise RuntimeError(f"YouTube 下載失敗：{detail}")

            candidates = sorted(
                [p for p in Path(td).glob("youtube.*") if p.suffix != ".part"],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise RuntimeError("YouTube 下載失敗：找不到下載後音訊檔")

            audio_payload = cls._extract_audio_from_video(str(candidates[0]))
            source_name = f"YouTube: {title[:60]}"
            return audio_payload, source_name

    @staticmethod
    def _extract_audio_from_video(video_path: str) -> bytes:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("影片辨識需要 ffmpeg（請安裝並加入 PATH）")

        cmd = [
            ffmpeg_bin,
            "-v",
            "error",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
            detail = stderr or "ffmpeg 執行失敗"
            raise RuntimeError(f"影片音訊抽取失敗：{detail}")
        if not proc.stdout:
            raise RuntimeError("影片音訊抽取失敗：沒有可用音訊")
        return proc.stdout

    def _run_asr_pipeline(self, cfg: dict, audio_payload: bytes, source_name: str):
        if not audio_payload:
            self._safe_after(lambda: self._show_error("未取得可辨識音訊"))
            return

        need_segments = bool(cfg.get("lyrics_mode", False) or cfg.get("srt_mode", False))
        started_at = perf_counter()
        asr_result = transcribe_audio(audio_payload, cfg, need_segments=need_segments)
        elapsed = perf_counter() - started_at

        original = apply_dictionary(asr_result.text)
        seg_rows = [{"start": s.start, "end": s.end, "text": s.text} for s in asr_result.segments]
        if cfg.get("srt_mode", False):
            fallback_lines = split_lyrics_lines(original)
            srt_text = build_srt_from_segments(seg_rows, fallback_lines=fallback_lines)
            self._safe_after(lambda t=srt_text or original: self._set_original(t))
        else:
            self._safe_after(lambda t=original: self._set_original(t))
        self._safe_after(
            lambda sec=elapsed, e=asr_result.engine, m=asr_result.model, src=source_name: self._asr_time_label.configure(
                text=f"辨識耗時：{sec:.2f}s  ·  引擎：{e}  ·  模型：{m}  ·  來源：{src}"
            )
        )

        if cfg.get("lyrics_mode", False):
            lyric_lines = split_lyrics_lines(original)
            self._safe_after(lambda t="\n".join(lyric_lines): self._set_lyrics_text(t))
            lrc_text = build_lrc_from_segments(seg_rows, fallback_lines=lyric_lines)
            self._safe_after(lambda t=lrc_text: self._set_lrc_text(t))
        else:
            self._safe_after(lambda: self._set_lyrics_text(""))
            self._safe_after(lambda: self._set_lrc_text(""))

        self._safe_after(lambda: self._set_state("idle"))

    def _stop_and_transcribe(self):
        cfg_snapshot = self._home_page._get_current_config()
        cfg_snapshot["lyrics_mode"] = bool(self._lyrics_mode_var.get())
        cfg_snapshot["srt_mode"] = bool(self._srt_mode_var.get())
        self._timer_label.configure(text="")
        self._set_state("transcribing")

        def _do(cfg: dict):
            try:
                audio_payload = self._recorder.stop(as_wav=True)
                if not audio_payload:
                    self._safe_after(lambda: self._show_error("未錄到音訊，請再試一次"))
                    return

                self._run_asr_pipeline(cfg, audio_payload, source_name="麥克風")
            except Exception as e:
                self._safe_after(lambda err=str(e): self._show_error(err))

        threading.Thread(target=_do, args=(cfg_snapshot,), daemon=True).start()

    def _set_original(self, text: str):
        self._original_text.delete("1.0", "end")
        self._original_text.insert("1.0", text)

    def _set_lyrics_text(self, text: str):
        self._lyrics_text.delete("1.0", "end")
        self._lyrics_text.insert("1.0", text)

    def _set_lrc_text(self, text: str):
        self._lrc_text.delete("1.0", "end")
        self._lrc_text.insert("1.0", text)

    def _export_srt(self):
        from tkinter import filedialog

        content = self._original_text.get("1.0", "end").strip()
        if not content:
            self._show_error("沒有可匯出的字幕內容")
            return

        path = filedialog.asksaveasfilename(
            title="匯出 SRT 字幕",
            defaultextension=".srt",
            filetypes=[("SRT 字幕", "*.srt"), ("文字檔", "*.txt"), ("所有檔案", "*.*")],
            initialfile="asr_subtitles.srt",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            self._show_error(f"匯出失敗：{e}")
            return

        self._show_toast(f"已匯出：{path}")

    def _clear_output(self):
        self._set_original("")
        self._set_lyrics_text("")
        self._set_lrc_text("")
        self._asr_time_label.configure(text="辨識耗時：--")

    def _on_lyrics_toggle(self):
        if bool(self._lyrics_mode_var.get()):
            self._lyrics_frame.pack(fill="both", expand=False, pady=(0, 8))
        else:
            self._lyrics_frame.pack_forget()

    def _copy_text(self, text: str):
        if not text:
            self._show_toast("沒有可複製的內容")
            return
        if not copy_to_clipboard(text):
            self._show_error("複製失敗：請安裝 pyperclip")
            return
        self._show_toast("已複製到剪貼簿")

    def _copy_last_error(self):
        msg = (self._last_error_message or "").strip()
        if not msg:
            self._show_toast("目前沒有可複製的錯誤訊息")
            return
        if not copy_to_clipboard(msg):
            self._show_error("複製失敗：請安裝 pyperclip")
            return
        self._show_toast("錯誤訊息已複製")

    def _show_error(self, msg: str):
        self._last_error_message = str(msg).strip()
        self._set_state("error")
        self._status_label.configure(text=f"錯誤：{msg}", text_color="#e74c3c")
        self._safe_after(lambda: self._set_state("idle"), delay_ms=2500)

    def _show_toast(self, msg: str):
        toast = ctk.CTkLabel(self, text=msg, fg_color="#2ecc71", text_color="white",
                             corner_radius=6, font=ctk.CTkFont(size=13))
        toast.place(relx=0.5, rely=0.97, anchor="s")
        self.after(2000, toast.destroy)


class TtsAppPage(ctk.CTkFrame):
    """TTS 專用頁：文字轉語音。"""

    def __init__(self, master, home_page: HomePage, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._home_page = home_page
        self._last_generated_audio_path = ""
        self._last_error_message = ""
        self._build_ui()

    def _build_ui(self):
        ctk.CTkLabel(
            self,
            text="TTS APP",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(
            self,
            text="僅做文字轉語音（Qwen3 TTS）。",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        ).pack(anchor="w", pady=(0, 14))

        toolbar = ctk.CTkFrame(self, fg_color="transparent")
        toolbar.pack(fill="x", pady=(0, 8))

        self._synthesize_btn = ctk.CTkButton(
            toolbar,
            text="TTS 合成",
            width=100,
            fg_color="#2d98da",
            hover_color="#1b7cb7",
            command=self._synthesize,
        )
        self._synthesize_btn.pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            toolbar,
            text="上傳文字",
            width=90,
            command=self._load_text_file,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            toolbar,
            text="清空文字",
            width=90,
            command=lambda: self._input_text.delete("1.0", "end"),
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            toolbar,
            text="貼上",
            width=70,
            command=self._paste_from_clipboard,
        ).pack(side="left")

        ctk.CTkLabel(toolbar, text="說話者", font=ctk.CTkFont(size=12)).pack(side="left", padx=(10, 2))
        ctk.CTkOptionMenu(
            toolbar,
            values=TTS_SPEAKERS,
            variable=self._home_page._tts_speaker_var,
            width=110,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkLabel(toolbar, text="語言", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 2))
        ctk.CTkOptionMenu(
            toolbar,
            values=TTS_LANGUAGES,
            variable=self._home_page._tts_language_var,
            width=90,
        ).pack(side="left", padx=(0, 10))

        self._play_btn = ctk.CTkButton(
            toolbar,
            text="▶ 播放",
            width=80,
            fg_color="#27ae60",
            hover_color="#1e8449",
            state="disabled",
            command=self._play_audio,
        )
        self._play_btn.pack(side="left", padx=(0, 0))

        ctk.CTkButton(
            toolbar,
            text="語音另存",
            width=90,
            command=self._export_audio_as,
        ).pack(side="left", padx=(6, 0))

        ctk.CTkButton(
            toolbar,
            text="複製錯誤",
            width=86,
            command=self._copy_last_error,
        ).pack(side="left", padx=(6, 0))

        instruct_row = ctk.CTkFrame(self, fg_color="transparent")
        instruct_row.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(instruct_row, text="語氣指令", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 6))
        ctk.CTkEntry(
            instruct_row,
            textvariable=self._home_page._tts_instruct_var,
            placeholder_text="例：用開心的語氣說（留空不套用）",
        ).pack(side="left", fill="x", expand=True)

        self._status_label = ctk.CTkLabel(self, text="待機中", font=ctk.CTkFont(size=13), text_color="gray")
        self._status_label.pack(anchor="w", pady=(0, 4))

        self._tts_time_label = ctk.CTkLabel(self, text="合成耗時：--", font=ctk.CTkFont(size=13), text_color="gray")
        self._tts_time_label.pack(anchor="w", pady=(0, 6))

        path_row = ctk.CTkFrame(self, fg_color="transparent")
        path_row.pack(fill="x", pady=(0, 6))
        path_row.columnconfigure(1, weight=1)
        ctk.CTkLabel(path_row, text="輸出路徑", font=ctk.CTkFont(size=12), text_color="gray").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        self._tts_output_path_var = ctk.StringVar(value="")
        self._tts_output_path_entry = ctk.CTkEntry(
            path_row,
            textvariable=self._tts_output_path_var,
            placeholder_text="留空自動命名（backups/tts-*.wav）",
        )
        self._tts_output_path_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            path_row,
            text="選擇",
            width=70,
            command=self._choose_output_path,
        ).grid(row=0, column=2, sticky="e", padx=(0, 6))
        ctk.CTkButton(
            path_row,
            text="清空",
            width=70,
            command=lambda: self._tts_output_path_var.set(""),
        ).grid(row=0, column=3, sticky="e")

        self._keep_timeline_export_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            self,
            text="匯出時保留時間軸（同名 .srt）",
            variable=self._keep_timeline_export_var,
        ).pack(anchor="w", pady=(0, 8))

        self._output_path_label = ctk.CTkLabel(self, text="輸出檔案：--", font=ctk.CTkFont(size=12), text_color="gray")
        self._output_path_label.pack(anchor="w", pady=(0, 10))

        panel = ctk.CTkFrame(self, fg_color="transparent")
        panel.pack(fill="both", expand=True)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        hdr = ctk.CTkFrame(panel, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(
            hdr,
            text="輸入文字",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray",
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            hdr,
            text="複製",
            width=58,
            height=24,
            font=ctk.CTkFont(size=12),
            command=lambda: self._copy_text(self._input_text.get("1.0", "end").strip()),
        ).grid(row=0, column=1, padx=(6, 0))

        self._input_text = ctk.CTkTextbox(panel, font=ctk.CTkFont(size=13), wrap="word")
        self._input_text.grid(row=1, column=0, sticky="nsew")

    def _safe_after(self, callback, delay_ms: int = 0):
        try:
            if delay_ms <= 0:
                self.after(0, callback)
            else:
                self.after(0, lambda: self.after(delay_ms, callback))
        except tk.TclError:
            return
        except Exception as exc:
            LOGGER.exception("TTS APP schedule callback failed: %s", exc)

    def _set_status(self, text: str, color: str = "gray"):
        self._status_label.configure(text=text, text_color=color)

    def _paste_from_clipboard(self):
        try:
            text = self.clipboard_get()
        except Exception:
            self._show_error("剪貼簿沒有可貼上的文字")
            return
        self._input_text.insert("end", text)

    def _load_text_file(self):
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
        except OSError as e:
            self._show_error(f"讀取檔案失敗：{e}")
            return

        self._input_text.delete("1.0", "end")
        self._input_text.insert("1.0", content)
        self._show_toast(f"已載入：{Path(path).name}")

    def _choose_output_path(self):
        from tkinter import filedialog

        ext = ".wav"
        path = filedialog.asksaveasfilename(
            title="設定 TTS 輸出路徑",
            defaultextension=ext,
            filetypes=[("WAV 音訊", "*.wav"), ("所有檔案", "*.*")],
            initialfile=f"tts_output{ext}",
        )
        if path:
            self._tts_output_path_var.set(path)

    @staticmethod
    def _build_timeline_for_export(text: str) -> str | None:
        src = (text or "").strip()
        if not src:
            return None

        if "-->" in src:
            return src

        seg_rows: list[dict] = []
        lrc_pat = re.compile(r"^\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]\s*(.*)$")
        for line in src.splitlines():
            m = lrc_pat.match(line.strip())
            if not m:
                continue
            mm = int(m.group(1))
            ss = int(m.group(2))
            frac = m.group(3) or "0"
            if len(frac) == 1:
                frac_ms = int(frac) * 100
            elif len(frac) == 2:
                frac_ms = int(frac) * 10
            else:
                frac_ms = int(frac[:3])
            content = (m.group(4) or "").strip()
            if not content:
                continue
            start_sec = mm * 60 + ss + (frac_ms / 1000.0)
            seg_rows.append({"start": start_sec, "text": content})

        if not seg_rows:
            return None

        seg_rows = sorted(seg_rows, key=lambda r: float(r["start"]))
        for idx, row in enumerate(seg_rows):
            if idx + 1 < len(seg_rows):
                next_start = float(seg_rows[idx + 1]["start"])
                row["end"] = max(float(row["start"]) + 0.2, next_start)
            else:
                row["end"] = float(row["start"]) + 2.5

        return build_srt_from_segments(seg_rows)

    def _export_audio_as(self):
        from tkinter import filedialog

        src_audio = (self._last_generated_audio_path or "").strip()
        if not src_audio:
            self._show_error("尚未生成語音，請先執行 TTS 合成")
            return
        src_path = Path(src_audio)
        if not src_path.exists():
            self._show_error(f"找不到已生成語音檔：{src_path}")
            return

        ext = ".wav"
        target = filedialog.asksaveasfilename(
            title="語音另存新檔",
            defaultextension=ext,
            filetypes=[("WAV 音訊", "*.wav"), ("所有檔案", "*.*")],
            initialfile=f"{src_path.stem}_export{ext}",
        )
        if not target:
            return

        try:
            shutil.copy2(src_path, target)
        except OSError as e:
            self._show_error(f"語音匯出失敗：{e}")
            return

        timeline_saved = False
        if self._keep_timeline_export_var.get():
            raw_text = self._input_text.get("1.0", "end").strip()
            timeline_text = self._build_timeline_for_export(raw_text)
            if timeline_text:
                srt_path = str(Path(target).with_suffix(".srt"))
                try:
                    with open(srt_path, "w", encoding="utf-8") as f:
                        f.write(timeline_text)
                    timeline_saved = True
                except OSError as e:
                    self._show_error(f"SRT 匯出失敗：{e}")
                    return

        self._tts_output_path_var.set(target)
        self._output_path_label.configure(text=f"輸出檔案：{target}")
        if timeline_saved:
            self._show_toast(f"已匯出語音與時間軸：{target}")
        else:
            self._show_toast(f"已匯出語音：{target}")
    def _synthesize(self):
        text = self._input_text.get("1.0", "end").strip()
        if not text:
            self._show_error("沒有可合成的文字")
            return

        cfg_snapshot = self._home_page._get_current_config()
        custom_output = self._tts_output_path_var.get().strip()
        if custom_output:
            cfg_snapshot["tts_output_path"] = custom_output
        self._synthesize_btn.configure(state="disabled", text="合成中...")
        self._set_status("語音合成中...", "#3498db")
        self._tts_time_label.configure(text="合成耗時：--")

        started_at = perf_counter()

        def _status(msg: str):
            self._safe_after(lambda m=msg: self._set_status(m, "#3498db"))

        def _do(cfg: dict, src_text: str):
            try:
                result = synthesize_tts(src_text, cfg, status_callback=_status)
                elapsed = perf_counter() - started_at
                self._safe_after(lambda sec=elapsed: self._tts_time_label.configure(text=f"合成耗時：{sec:.2f}s"))
                self._safe_after(lambda p=result.output_path: self._output_path_label.configure(text=f"輸出檔案：{p}"))
                self._safe_after(lambda p=result.output_path: self._tts_output_path_var.set(p))
                self._safe_after(lambda p=result.output_path: setattr(self, "_last_generated_audio_path", p))
                self._safe_after(lambda: self._play_btn.configure(state="normal"))
                self._safe_after(lambda: self._set_status("完成", "#2ecc71"))
            except Exception as e:
                self._safe_after(lambda err=str(e): self._show_error(f"TTS 失敗：{err}"))
            finally:
                self._safe_after(lambda: self._synthesize_btn.configure(state="normal", text="TTS 合成"))
        threading.Thread(target=_do, args=(cfg_snapshot, text), daemon=True).start()

    def _play_audio(self):
        import os
        path = (self._last_generated_audio_path or "").strip()
        if not path or not Path(path).exists():
            self._show_error("找不到音頻檔，請先合成")
            return
        try:
            os.startfile(path)
        except Exception as e:
            self._show_error(f"播放失敗：{e}")

    def _copy_text(self, text: str):
        if not text:
            self._show_toast("沒有可複製的內容")
            return
        if not copy_to_clipboard(text):
            self._show_error("複製失敗：請安裝 pyperclip")
            return
        self._show_toast("已複製到剪貼簿")

    def _copy_last_error(self):
        msg = (self._last_error_message or "").strip()
        if not msg:
            self._show_toast("目前沒有可複製的錯誤訊息")
            return
        if not copy_to_clipboard(msg):
            self._show_error("複製失敗：請安裝 pyperclip")
            return
        self._show_toast("錯誤訊息已複製")

    def _show_error(self, msg: str):
        self._last_error_message = str(msg).strip()
        self._set_status(f"錯誤：{msg}", "#e74c3c")
        self._safe_after(lambda: self._set_status("待機中", "gray"), delay_ms=2500)

    def _show_toast(self, msg: str):
        toast = ctk.CTkLabel(self, text=msg, fg_color="#2ecc71", text_color="white",
                             corner_radius=6, font=ctk.CTkFont(size=13))
        toast.place(relx=0.5, rely=0.97, anchor="s")
        self.after(2000, toast.destroy)







