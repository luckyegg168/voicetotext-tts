"""Qwen3-ASR helpers (download/cache/transcribe)."""

from __future__ import annotations

import gc
import io
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
from app.core.qwen_runtime import ensure_qwen_runtime

QWEN3_ASR_MODELS = [
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
]
QWEN3_ALIGNER_MODELS = [
    "Qwen/Qwen3-ForcedAligner-0.6B",
]

_ASR_MODEL_CACHE: OrderedDict[tuple[str, str], object] = OrderedDict()
_ASR_MODEL_LOCK = threading.RLock()
_ASR_MODEL_LIMIT = 1


def _repo_to_cache_dir(repo_id: str) -> Path:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    safe = repo_id.replace("/", "--")
    return cache_root / f"models--{safe}"


def is_repo_cached(repo_id: str) -> bool:
    model_dir = _repo_to_cache_dir(repo_id)
    snapshots = model_dir / "snapshots"
    return snapshots.exists() and any(snapshots.iterdir())


def download_repo(repo_id: str, status_callback=None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("Missing huggingface_hub; install requirements.txt first.") from exc

    if is_repo_cached(repo_id):
        if status_callback:
            status_callback(f"Cached: {repo_id}")
        return

    if status_callback:
        status_callback(f"Downloading: {repo_id}")

    snapshot_download(repo_id=repo_id, resume_download=True)

    if status_callback:
        status_callback(f"Downloaded: {repo_id}")


def download_all_qwen3_asr_models(status_callback=None) -> None:
    for repo_id in [*QWEN3_ASR_MODELS, *QWEN3_ALIGNER_MODELS]:
        download_repo(repo_id, status_callback=status_callback)


def _to_audio_np(audio_input: bytes | np.ndarray) -> np.ndarray:
    if isinstance(audio_input, np.ndarray):
        data = audio_input
        if data.ndim > 1:
            data = data[:, 0]
        return data.astype(np.float32, copy=False)

    if not audio_input:
        return np.array([], dtype=np.float32)

    buf = io.BytesIO(audio_input)
    _, data = wavfile.read(buf)
    if data.ndim > 1:
        data = data[:, 0]

    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    return data.astype(np.float32)


def _get_model(model_id: str, device: str):
    ensure_qwen_runtime("Qwen3-ASR")
    try:
        from qwen_asr import Qwen3ASRModel
        import torch
    except ImportError as exc:
        raise ImportError(
            "Missing qwen-asr package; run: pip install qwen-asr"
        ) from exc

    key = (model_id, device)
    with _ASR_MODEL_LOCK:
        if key in _ASR_MODEL_CACHE:
            _ASR_MODEL_CACHE.move_to_end(key)
            return _ASR_MODEL_CACHE[key]
        if len(_ASR_MODEL_CACHE) >= _ASR_MODEL_LIMIT:
            _ASR_MODEL_CACHE.popitem(last=False)
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        device_map = "cuda:0" if str(device).lower() == "cuda" else "cpu"
        model = Qwen3ASRModel.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map=device_map,
        )
        _ASR_MODEL_CACHE[key] = model
        return model


def transcribe(
    audio_input: bytes | np.ndarray,
    model_id: str = "Qwen/Qwen3-ASR-0.6B",
    aligner_model: str | None = None,
    language: str = "auto",
    device: str = "cuda",
    return_segments: bool = False,
) -> tuple[str, list[dict]]:
    del aligner_model  # Reserved for future forced-alignment integration.
    audio_np = _to_audio_np(audio_input)
    if audio_np.size == 0:
        raise ValueError("No audio input data.")

    model = _get_model(model_id, device)

    lang = language if language and language != "auto" else None

    results = model.transcribe(
        audio=(audio_np, 16000),
        language=lang,
    )

    if not results:
        return "", []

    result = results[0]
    text = str(result.text).strip() if hasattr(result, "text") else str(result).strip()

    chunks: list[dict] = []
    if return_segments:
        segments = getattr(result, "segments", None) or []
        for seg in segments:
            chunks.append({
                "start": getattr(seg, "start", None),
                "end": getattr(seg, "end", None),
                "text": str(getattr(seg, "text", "")).strip(),
            })

    return text, chunks
