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

_ASR_PIPELINE_CACHE: OrderedDict[tuple[str, str], object] = OrderedDict()
_ASR_PIPELINE_LOCK = threading.RLock()
_ASR_PIPELINE_LIMIT = 1


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


def _device_for_pipeline(device: str) -> int:
    return 0 if str(device).lower() == "cuda" else -1


def _get_pipeline(model_id: str, device: str):
    ensure_qwen_runtime("Qwen3-ASR")
    try:
        import transformers
        import torch
    except ImportError as exc:
        raise ImportError("Missing transformers; install requirements.txt first.") from exc

    key = (model_id, device)
    with _ASR_PIPELINE_LOCK:
        if key in _ASR_PIPELINE_CACHE:
            _ASR_PIPELINE_CACHE.move_to_end(key)
            return _ASR_PIPELINE_CACHE[key]
        if len(_ASR_PIPELINE_CACHE) >= _ASR_PIPELINE_LIMIT:
            _ASR_PIPELINE_CACHE.popitem(last=False)
            gc.collect()
            torch.cuda.empty_cache()
        _ASR_PIPELINE_CACHE[key] = transformers.pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            trust_remote_code=True,
            device=_device_for_pipeline(device),
            torch_dtype=torch.float16,
        )
        return _ASR_PIPELINE_CACHE[key]


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

    asr = _get_pipeline(model_id, device)

    kwargs: dict = {}
    if return_segments:
        kwargs["return_timestamps"] = True

    if language and language != "auto":
        # Not all models support language forcing; keep best-effort.
        kwargs["generate_kwargs"] = {"language": language}

    result = asr({"array": audio_np, "sampling_rate": 16000}, **kwargs)

    if isinstance(result, dict):
        text = str(result.get("text", "")).strip()
        chunks = []
        raw_chunks = result.get("chunks")
        if isinstance(raw_chunks, list):
            for item in raw_chunks:
                if not isinstance(item, dict):
                    continue
                ts = item.get("timestamp") or item.get("timestamps")
                start = None
                end = None
                if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                    start = float(ts[0]) if ts[0] is not None else None
                    end = float(ts[1]) if ts[1] is not None else None
                chunks.append({
                    "start": start,
                    "end": end,
                    "text": str(item.get("text", "")).strip(),
                })
        return text, chunks

    if isinstance(result, str):
        return result.strip(), []

    return str(result).strip(), []
