import pytest

from app.core import tts_router


def test_synthesize_defaults_to_qwen3_when_engine_missing(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_qwen(text, output_path, model_id, device, **kwargs):
        captured["text"] = text
        captured["output_path"] = output_path
        captured["model_id"] = model_id
        captured["device"] = device
        captured.update(kwargs)
        return output_path

    monkeypatch.setattr("app.core.tts_router.qwen3_tts.synthesize", _fake_qwen)

    cfg = {
        "tts_qwen3_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "asr_device": "cpu",
        "tts_output_path": "backups/custom_tts_output",
    }
    result = tts_router.synthesize("hello", cfg)

    assert result.engine == "qwen3_tts"
    assert result.output_path.endswith("custom_tts_output.wav")
    assert captured["output_path"].endswith("custom_tts_output.wav")
    assert captured["device"] == "cpu"


def test_synthesize_qwen_alias_maps_to_qwen3_tts(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_qwen(text, output_path, model_id, device, **kwargs):
        captured["output_path"] = output_path
        captured["model_id"] = model_id
        captured["device"] = device
        captured.update(kwargs)
        return output_path

    monkeypatch.setattr("app.core.tts_router.qwen3_tts.synthesize", _fake_qwen)

    cfg = {
        "tts_engine": "qwen3",
        "tts_qwen3_model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "tts_device": "cpu",
        "tts_output_path": "backups/custom_tts_alias",
    }
    result = tts_router.synthesize("hello", cfg)

    assert result.engine == "qwen3_tts"
    assert result.output_path.endswith("custom_tts_alias.wav")
    assert captured["device"] == "cpu"


def test_synthesize_rejects_legacy_edge_engine(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.core.tts_router.qwen3_tts.synthesize", lambda *args, **kwargs: "unused.wav")

    with pytest.raises(ValueError, match="Unsupported TTS engine"):
        tts_router.synthesize("hello", {"tts_engine": "edge_tts"})


def test_synthesize_rejects_unknown_engine(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.core.tts_router.qwen3_tts.synthesize", lambda *args, **kwargs: "unused.wav")

    with pytest.raises(ValueError, match="Unsupported TTS engine"):
        tts_router.synthesize("hello", {"tts_engine": "custom_tts"})
