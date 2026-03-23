"""Qwen3-TTS helper (download/cache/synthesize)."""

from __future__ import annotations

import gc
import io
from pathlib import Path
import threading

import numpy as np
import scipy.io.wavfile as wavfile
from app.core.qwen_runtime import ensure_qwen_runtime

QWEN3_TTS_MODELS = [
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

from collections import OrderedDict
_MODEL_CACHE: OrderedDict[tuple[str, str], object] = OrderedDict()
_MODEL_LOCK = threading.RLock()
_MODEL_LIMIT = 1


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


def download_all_qwen3_tts_models(status_callback=None) -> None:
    for repo_id in QWEN3_TTS_MODELS:
        download_repo(repo_id, status_callback=status_callback)


def _get_model(model_id: str, device: str):
    ensure_qwen_runtime("Qwen3-TTS")
    try:
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise ImportError(
            "Missing qwen_tts; install requirements-qwen.txt first: "
            "python -m pip install qwen-tts"
        ) from exc

    import torch

    use_cuda = str(device).lower() == "cuda" and torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device_map = "cuda" if use_cuda else "cpu"

    key = (model_id, device_map)
    with _MODEL_LOCK:
        if key in _MODEL_CACHE:
            _MODEL_CACHE.move_to_end(key)
            return _MODEL_CACHE[key]
        if len(_MODEL_CACHE) >= _MODEL_LIMIT:
            _MODEL_CACHE.popitem(last=False)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _MODEL_CACHE[key] = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
        )
        return _MODEL_CACHE[key]


def _model_generate_type(model_id: str) -> str:
    """Return 'voice_design' | 'custom_voice' | 'base' based on model_id naming."""
    if "VoiceDesign" in model_id:
        return "voice_design"
    if "CustomVoice" in model_id:
        return "custom_voice"
    return "base"  # *-Base 及其他


def synthesize(
    text: str,
    output_path: str,
    model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device: str = "cuda",
    language: str = "Chinese",
    speaker: str = "Vivian",
    instruct: str = "",
    return_bytes: bool = False,
) -> str | bytes:
    if not text.strip():
        raise ValueError("No text content for synthesis.")

    out_path = Path(output_path) if output_path else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _get_model(model_id, device)

    instruct_arg = instruct.strip() or None
    gen_type = _model_generate_type(model_id)
    if gen_type == "voice_design":
        wavs, sr = model.generate_voice_design(text, instruct=instruct_arg or "", language=language)
    elif gen_type == "custom_voice":
        wavs, sr = model.generate_custom_voice(text, speaker=speaker, language=language, instruct=instruct_arg)
    else:  # base
        wavs, sr = model.generate_voice_clone(text, language=language)

    arr = np.asarray(wavs[0])
    if arr.ndim > 1:
        arr = arr[0]
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767).astype(np.int16)

    if return_bytes:
        buf = io.BytesIO()
        wavfile.write(buf, sr, pcm)
        return buf.getvalue()

    assert out_path is not None, "output_path must be set when return_bytes=False"
    wavfile.write(str(out_path), sr, pcm)
    return str(out_path)
