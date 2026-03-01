"""語音轉寫 — 支援 OpenAI Whisper API 與本地 faster-whisper（GPU）"""
import io
import os
import sys
import numpy as np
import scipy.io.wavfile as wavfile
from openai import OpenAI


# CUDA DLL 預載入已由 main.py 開頭的 cuda_setup.setup() 處理

LANGUAGE_MAP = {
    "auto":  None,
    "zh-TW": "zh",   # Whisper 用 "zh"，但繁體/簡體由 initial_prompt 引導
    "zh-CN": "zh",
    "en":    "en",
    "ja":    "ja",
    "ko":    "ko",
}

# 繁/簡體引導 prompt（Whisper 本身不區分，用 initial_prompt 偏向輸出字形）
_INITIAL_PROMPT = {
    "zh-TW": "以下是繁體中文內容。",
    "zh-CN": "以下是简体中文内容。",
}

# 本地 Whisper 模型快取，避免每次重新載入
_local_model_cache: dict = {}


def is_whisper_model_cached(model_size: str) -> bool:
    """檢查 faster-whisper 模型是否已下載到本機快取"""
    from pathlib import Path
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / f"models--Systran--faster-whisper-{model_size}"
    snapshots = model_dir / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def download_whisper_model(
    model_size: str,
    status_callback=None,
) -> None:
    """
    下載 faster-whisper 模型（若已快取則跳過）。
    status_callback(msg: str) 用於更新 UI 進度文字。
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("請先執行：pip install faster-whisper")

    if is_whisper_model_cached(model_size):
        if status_callback:
            status_callback(f"✅ {model_size} 已存在")
        return

    if status_callback:
        status_callback(f"⬇ 下載 {model_size} 中（首次需數分鐘）...")

    # 用 CPU/int8 只為觸發下載，不佔用 GPU
    WhisperModel(model_size, device="cpu", compute_type="int8")

    if status_callback:
        status_callback(f"✅ {model_size} 下載完成")


def transcribe(
    audio_bytes: bytes,
    api_key: str,
    language: str = "auto",
    base_url: str | None = None,
) -> str:
    """OpenAI Whisper API 轉寫（雲端）"""
    if not audio_bytes:
        raise ValueError("音訊資料為空")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"

    lang = LANGUAGE_MAP.get(language)
    req_kwargs: dict = {
        "model": "whisper-1",
        "file": audio_file,
        "response_format": "text",
    }
    if lang:
        req_kwargs["language"] = lang
    prompt = _INITIAL_PROMPT.get(language)
    if prompt:
        req_kwargs["prompt"] = prompt

    result = client.audio.transcriptions.create(**req_kwargs)
    return str(result).strip()


def transcribe_local(
    audio_bytes: bytes,
    model_size: str = "large-v3",
    language: str = "auto",
    device: str = "cuda",
    compute_type: str = "float16",
) -> str:
    """
    本地 faster-whisper 轉寫（GPU 加速）。
    model_size: tiny / base / small / medium / large-v2 / large-v3
    device: cuda（NVIDIA GPU）或 cpu
    compute_type: float16（GPU）或 int8（CPU 省記憶體）
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("請先執行：pip install faster-whisper")

    if not audio_bytes:
        raise ValueError("音訊資料為空")

    # 載入或從快取取得模型
    cache_key = (model_size, device, compute_type)
    if cache_key not in _local_model_cache:
        _local_model_cache[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
    model = _local_model_cache[cache_key]

    # WAV bytes → numpy float32（單聲道）
    buf = io.BytesIO(audio_bytes)
    rate, data = wavfile.read(buf)
    if data.ndim > 1:
        data = data[:, 0]  # 多聲道取第一聲道
    if data.dtype == np.int16:
        audio_np = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio_np = data.astype(np.float32) / 2147483648.0
    else:
        audio_np = data.astype(np.float32)

    lang = LANGUAGE_MAP.get(language)  # None = 自動偵測
    initial_prompt = _INITIAL_PROMPT.get(language)
    transcribe_kwargs: dict = {"language": lang, "beam_size": 5}
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt
    segments, _ = model.transcribe(audio_np, **transcribe_kwargs)
    return "".join(seg.text for seg in segments).strip()
