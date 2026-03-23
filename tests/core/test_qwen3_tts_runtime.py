import pytest

from app.core import qwen_runtime


def test_ensure_qwen_runtime_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        qwen_runtime,
        "get_qwen_runtime_status",
        lambda **_: qwen_runtime.QwenRuntimeStatus(
            available=False,
            reason="Qwen runtime is unavailable: Missing required packages: transformers, torch. Install dependencies via requirements.txt.",
        ),
    )

    with pytest.raises(RuntimeError, match="Qwen3-TTS"):
        qwen_runtime.ensure_qwen_runtime("Qwen3-TTS")


def test_ensure_qwen_runtime_noop_when_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        qwen_runtime,
        "get_qwen_runtime_status",
        lambda **_: qwen_runtime.QwenRuntimeStatus(available=True, reason="OK"),
    )

    qwen_runtime.ensure_qwen_runtime("Qwen3-TTS")  # must not raise
