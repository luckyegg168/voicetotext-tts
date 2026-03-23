"""LRU cache tests for qwen3_asr and qwen3_tts."""
from __future__ import annotations
from collections import OrderedDict
from unittest.mock import MagicMock, call, patch
import gc

import pytest


# ── ASR ──────────────────────────────────────────────────────────────────────

class TestAsrLruCache:
    """Tests for _get_pipeline() LRU behaviour in qwen3_asr."""

    def setup_method(self):
        import app.core.qwen3_asr as asr_mod
        # Clear cache before each test so tests are isolated
        asr_mod._ASR_PIPELINE_CACHE.clear()

    def test_cache_hit_does_not_reload(self):
        """Second call with same key returns cached object without loading again."""
        import app.core.qwen3_asr as asr_mod
        fake_pipe = MagicMock()
        with (
            patch("app.core.qwen3_asr.ensure_qwen_runtime"),
            patch("transformers.pipeline", return_value=fake_pipe) as mock_pipeline,
        ):
            asr_mod._get_pipeline("model-a", "cpu")
            asr_mod._get_pipeline("model-a", "cpu")
        assert mock_pipeline.call_count == 1

    def test_eviction_calls_cuda_empty_cache(self):
        """Evicting a model must call torch.cuda.empty_cache() to free VRAM."""
        import app.core.qwen3_asr as asr_mod
        fake_pipe = MagicMock()
        with (
            patch("app.core.qwen3_asr.ensure_qwen_runtime"),
            patch("transformers.pipeline", return_value=fake_pipe),
            patch("gc.collect") as mock_gc,
            patch("torch.cuda.empty_cache") as mock_cuda_empty,
        ):
            # Fill cache to limit (limit=1)
            asr_mod._get_pipeline("model-a", "cpu")
            # Load a different model — should evict model-a
            asr_mod._get_pipeline("model-b", "cpu")

        mock_cuda_empty.assert_called_once()
        mock_gc.assert_called()

    def test_eviction_removes_oldest_entry(self):
        """After eviction, old key is gone and new key is present."""
        import app.core.qwen3_asr as asr_mod
        fake_pipe = MagicMock()
        with (
            patch("app.core.qwen3_asr.ensure_qwen_runtime"),
            patch("transformers.pipeline", return_value=fake_pipe),
            patch("torch.cuda.empty_cache"),
        ):
            asr_mod._get_pipeline("model-a", "cpu")
            asr_mod._get_pipeline("model-b", "cpu")

        assert ("model-a", "cpu") not in asr_mod._ASR_PIPELINE_CACHE
        assert ("model-b", "cpu") in asr_mod._ASR_PIPELINE_CACHE

    def test_cache_uses_ordered_dict(self):
        """Cache must be an OrderedDict for LRU ordering."""
        import app.core.qwen3_asr as asr_mod
        assert isinstance(asr_mod._ASR_PIPELINE_CACHE, OrderedDict)

    def test_pipeline_uses_float16(self):
        """pipeline() call must pass torch_dtype=torch.float16."""
        import torch
        import app.core.qwen3_asr as asr_mod
        fake_pipe = MagicMock()
        with (
            patch("app.core.qwen3_asr.ensure_qwen_runtime"),
            patch("transformers.pipeline", return_value=fake_pipe) as mock_pipeline,
            patch("torch.cuda.empty_cache"),
        ):
            asr_mod._get_pipeline("model-a", "cpu")

        call_kwargs = mock_pipeline.call_args.kwargs
        assert call_kwargs.get("torch_dtype") == torch.float16
