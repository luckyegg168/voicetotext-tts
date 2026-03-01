"""Speech transcription helpers: OpenAI Whisper API and local faster-whisper."""

from __future__ import annotations

import gc
import io
import os
import threading
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from openai import OpenAI

LANGUAGE_MAP = {
    "auto": None,
    "zh-TW": "zh",
    "zh-CN": "zh",
    "en": "en",
    "ja": "ja",
    "ko": "ko",
}

_INITIAL_PROMPT = {
    "zh-TW": "以下為繁體中文語音，請以繁體中文逐字轉寫。",
    "zh-CN": "以下為简体中文语音，请以简体中文逐字转写。",
}

_local_model_cache: dict[tuple[str, str, str], object] = {}
_LOCAL_CACHE_LOCK = threading.RLock()
_LOCAL_MODEL_CACHE_LIMIT = 1
OPENAI_REQUEST_TIMEOUT_SECONDS = float(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "90"))


def is_whisper_model_cached(model_size: str) -> bool:
    """Check whether a faster-whisper model exists in local HuggingFace cache."""
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / f"models--Systran--faster-whisper-{model_size}"
    snapshots = model_dir / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def download_whisper_model(model_size: str, status_callback=None) -> None:
    """Download faster-whisper model by creating a temporary CPU model instance."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError("缺少 faster-whisper，請先執行 pip install faster-whisper") from exc

    if is_whisper_model_cached(model_size):
        if status_callback:
            status_callback(f"✅ {model_size} 已下載")
        return

    if status_callback:
        status_callback(f"正在下載 {model_size}（首次下載可能較久）...")

    WhisperModel(model_size, device="cpu", compute_type="int8")

    if status_callback:
        status_callback(f"✅ {model_size} 下載完成")


def transcribe(
    audio_bytes: bytes,
    api_key: str,
    language: str = "auto",
    base_url: str | None = None,
) -> str:
    """Transcribe audio bytes with OpenAI Whisper API."""
    if not audio_bytes:
        raise ValueError("沒有音訊資料")

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

    result = client.audio.transcriptions.create(
        **req_kwargs,
        timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
    )
    return str(result).strip()


def transcribe_local(
    audio_bytes: bytes | np.ndarray,
    model_size: str = "large-v3",
    language: str = "auto",
    device: str = "cuda",
    compute_type: str = "float16",
) -> str:
    """Transcribe local audio via faster-whisper (bytes or float32 ndarray)."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError("缺少 faster-whisper，請先執行 pip install faster-whisper") from exc

    if isinstance(audio_bytes, np.ndarray):
        if audio_bytes.size == 0:
            raise ValueError("沒有音訊資料")
    elif not audio_bytes:
        raise ValueError("沒有音訊資料")

    cache_key = (model_size, device, compute_type)
    with _LOCAL_CACHE_LOCK:
        if cache_key not in _local_model_cache:
            if len(_local_model_cache) >= _LOCAL_MODEL_CACHE_LIMIT:
                _local_model_cache.clear()
                gc.collect()
            _local_model_cache[cache_key] = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
        model = _local_model_cache[cache_key]

    if isinstance(audio_bytes, np.ndarray):
        data = audio_bytes
        if data.ndim > 1:
            data = data[:, 0]
        audio_np = data.astype(np.float32, copy=False)
    else:
        buf = io.BytesIO(audio_bytes)
        _, data = wavfile.read(buf)
        if data.ndim > 1:
            data = data[:, 0]
        if data.dtype == np.int16:
            audio_np = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            audio_np = data.astype(np.float32) / 2147483648.0
        else:
            audio_np = data.astype(np.float32)

    lang = LANGUAGE_MAP.get(language)
    initial_prompt = _INITIAL_PROMPT.get(language)
    transcribe_kwargs: dict = {"language": lang, "beam_size": 5}
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    segments, _ = model.transcribe(audio_np, **transcribe_kwargs)
    return "".join(seg.text for seg in segments).strip()
