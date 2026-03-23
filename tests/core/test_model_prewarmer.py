# tests/core/test_model_prewarmer.py
"""Tests for app.core.model_prewarmer."""
from __future__ import annotations
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


CFG = {
    "asr_qwen3_model": "Qwen/Qwen3-ASR-0.6B",
    "asr_device": "cpu",
    "tts_qwen3_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "tts_device": "cpu",
}


def test_prewarm_success_calls_callbacks_in_order():
    """Status callback must be called: asr-loading → tts-loading → ready."""
    from app.core.model_prewarmer import prewarm_models
    from app.core import qwen3_asr, qwen3_tts

    calls = []
    callback = lambda msg, color: calls.append((msg, color))

    with (
        patch.object(qwen3_asr, "_get_pipeline", return_value=MagicMock()),
        patch.object(qwen3_tts, "_get_model", return_value=MagicMock()),
        patch.object(qwen3_asr, "is_repo_cached", return_value=True),
        patch.object(qwen3_tts, "is_repo_cached", return_value=True),
    ):
        prewarm_models(CFG, status_callback=callback)

    assert len(calls) == 3
    assert "ASR" in calls[0][0]
    assert "TTS" in calls[1][0]
    assert "就緒" in calls[2][0] or "ready" in calls[2][0].lower()
    assert calls[2][1] == "#2ecc71"


def test_prewarm_failure_calls_failure_callback():
    """If _get_pipeline raises, callback must receive a failure message."""
    from app.core.model_prewarmer import prewarm_models
    from app.core import qwen3_asr, qwen3_tts

    calls = []
    callback = lambda msg, color: calls.append((msg, color))

    with (
        patch.object(qwen3_asr, "_get_pipeline", side_effect=RuntimeError("OOM")),
        patch.object(qwen3_tts, "_get_model", return_value=MagicMock()),
        patch.object(qwen3_asr, "is_repo_cached", return_value=True),
        patch.object(qwen3_tts, "is_repo_cached", return_value=True),
    ):
        prewarm_models(CFG, status_callback=callback)  # must NOT raise

    failure_msgs = [c for c in calls if "失敗" in c[0] or "fail" in c[0].lower()]
    assert failure_msgs, "Expected a failure callback message"


def test_prewarm_no_double_start():
    """Calling prewarm_models concurrently should load each model only once."""
    from app.core.model_prewarmer import prewarm_models, _prewarm_event
    from app.core import qwen3_asr, qwen3_tts

    load_count = {"asr": 0}

    def slow_pipeline(*args, **kwargs):
        load_count["asr"] += 1
        time.sleep(0.05)
        return MagicMock()

    _prewarm_event.clear()

    with (
        patch.object(qwen3_asr, "_get_pipeline", side_effect=slow_pipeline),
        patch.object(qwen3_tts, "_get_model", return_value=MagicMock()),
        patch.object(qwen3_asr, "is_repo_cached", return_value=True),
        patch.object(qwen3_tts, "is_repo_cached", return_value=True),
    ):
        t1 = threading.Thread(target=prewarm_models, args=(CFG,))
        t2 = threading.Thread(target=prewarm_models, args=(CFG,))
        t1.start()
        time.sleep(0.01)
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

    assert load_count["asr"] == 1, "Model should only be loaded once"


def test_prewarm_skips_if_model_not_cached():
    """If is_repo_cached() returns False, prewarm skips without error."""
    from app.core.model_prewarmer import prewarm_models
    from app.core import qwen3_asr, qwen3_tts

    with (
        patch.object(qwen3_asr, "is_repo_cached", return_value=False),
        patch.object(qwen3_tts, "is_repo_cached", return_value=False),
        patch.object(qwen3_asr, "_get_pipeline") as mock_asr,
        patch.object(qwen3_tts, "_get_model") as mock_tts,
    ):
        prewarm_models(CFG)  # no callback needed

    mock_asr.assert_not_called()
    mock_tts.assert_not_called()
