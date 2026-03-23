# app/core/model_prewarmer.py
"""Background model pre-warming for ASR and TTS."""
from __future__ import annotations

import logging
import threading
from typing import Callable

_LOGGER = logging.getLogger(__name__)

# Lock held while prewarm is running. acquire(blocking=False) is atomic,
# preventing the check-then-set race that threading.Event has.
_prewarm_lock = threading.Lock()


def prewarm_models(
    cfg: dict,
    status_callback: Callable[[str, str], None] | None = None,
) -> None:
    """Load default ASR and TTS models into cache in the calling thread.

    Args:
        cfg: Config dict with asr_qwen3_model, asr_device, tts_qwen3_model, tts_device.
        status_callback: Called as callback(message, color_hex) on status changes.
    """
    if not _prewarm_lock.acquire(blocking=False):
        _LOGGER.debug("Prewarm already in progress — skipping.")
        return

    try:
        _run_prewarm(cfg, status_callback)
    finally:
        _prewarm_lock.release()


def _notify(callback: Callable[[str, str], None] | None, msg: str, color: str) -> None:
    if callback:
        try:
            callback(msg, color)
        except Exception:
            pass


def _run_prewarm(
    cfg: dict,
    callback: Callable[[str, str], None] | None,
) -> None:
    from app.core import qwen3_asr, qwen3_tts

    asr_model = str(cfg.get("asr_qwen3_model", "Qwen/Qwen3-ASR-0.6B"))
    asr_device = str(cfg.get("asr_device", "cuda"))
    tts_model = str(cfg.get("tts_qwen3_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"))
    tts_device = str(cfg.get("tts_device", "cuda"))

    # ── ASR ──────────────────────────────────────────
    if asr_model not in qwen3_asr.QWEN3_ASR_MODELS:
        _LOGGER.warning("ASR model %r not in allowlist — skipping prewarm", asr_model)
    elif not qwen3_asr.is_repo_cached(asr_model):
        _LOGGER.info("ASR model not cached — skipping prewarm for %s", asr_model)
    else:
        _notify(callback, f"⚡ ASR 模型載入中 ({asr_model.split('/')[-1]})...", "#e67e22")
        try:
            qwen3_asr._get_pipeline(asr_model, asr_device)
        except Exception as exc:
            _LOGGER.warning("ASR prewarm failed: %s", exc)
            _notify(callback, "⚠️ ASR 預熱失敗，首次使用時再載入", "#f39c12")
            return

    # ── TTS ──────────────────────────────────────────
    if tts_model not in qwen3_tts.QWEN3_TTS_MODELS:
        _LOGGER.warning("TTS model %r not in allowlist — skipping prewarm", tts_model)
    elif not qwen3_tts.is_repo_cached(tts_model):
        _LOGGER.info("TTS model not cached — skipping prewarm for %s", tts_model)
    else:
        _notify(callback, f"⚡ TTS 模型載入中 ({tts_model.split('/')[-1]})...", "#e67e22")
        try:
            qwen3_tts._get_model(tts_model, tts_device)
        except Exception as exc:
            _LOGGER.warning("TTS prewarm failed: %s", exc)
            _notify(callback, "⚠️ TTS 預熱失敗，首次使用時再載入", "#f39c12")
            return

    _notify(callback, "✅ 模型就緒", "#2ecc71")
