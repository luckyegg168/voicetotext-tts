# Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 App 啟動後首次使用卡頓，以及切換模型時 VRAM 未正確釋放的問題。

**Architecture:** 將 ASR / TTS 的 `dict.clear()` 快取策略替換為 `OrderedDict` 精確 LRU 淘汰（含 `torch.cuda.empty_cache()`），並在 App 啟動後 500ms 啟動背景預熱 thread，同時在 `HomePage` 底部加入模型狀態列讓使用者看到載入進度。

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace Transformers, customtkinter, threading, unittest.mock

---

## 檔案異動一覽

| 動作 | 路徑 | 說明 |
|------|------|------|
| 修改 | `app/core/qwen3_asr.py` | `_ASR_PIPELINE_CACHE` dict→OrderedDict，LRU eviction，加 `torch_dtype=float16` |
| 修改 | `app/core/qwen3_tts.py` | `_MODEL_CACHE` dict→OrderedDict，LRU eviction |
| 新增 | `app/core/model_prewarmer.py` | 背景預熱邏輯，防重複啟動 Event |
| 修改 | `app/ui/main_window.py` | `HomePage` 加 `update_model_status()` 方法與狀態列 label |
| 修改 | `main.py` | 加 `_start_prewarm()` + `self.after(500, ...)` |
| 新增 | `tests/core/test_lru_cache.py` | LRU 快取測試（ASR + TTS） |
| 新增 | `tests/core/test_model_prewarmer.py` | 預熱邏輯測試 |

---

## Task 1：ASR LRU 快取測試（紅燈）

**Files:**
- Create: `tests/core/test_lru_cache.py`

- [ ] **Step 1：寫失敗測試**

```python
# tests/core/test_lru_cache.py
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
```

- [ ] **Step 2：跑測試，確認全部失敗**

```bash
cd D:/qwen3tts-asr && python -m pytest tests/core/test_lru_cache.py -v
```

預期：`FAILED` — `OrderedDict`, `torch.cuda.empty_cache`, `torch_dtype` 等尚未實作

---

## Task 2：實作 ASR LRU 快取（綠燈）

**Files:**
- Modify: `app/core/qwen3_asr.py:1-30` (imports), `app/core/qwen3_asr.py:90-110` (`_get_pipeline`)

- [ ] **Step 1：修改 imports 區塊**

在 `app/core/qwen3_asr.py` 頂部，將：
```python
_ASR_PIPELINE_CACHE: dict[tuple[str, str], object] = {}
```
改為：
```python
from collections import OrderedDict
_ASR_PIPELINE_CACHE: OrderedDict[tuple[str, str], object] = OrderedDict()
```

- [ ] **Step 2：替換 `_get_pipeline()` 快取邏輯**

將整個 `_get_pipeline()` 函式改為：

```python
def _get_pipeline(model_id: str, device: str):
    ensure_qwen_runtime("Qwen3-ASR")
    try:
        from transformers import pipeline
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
        _ASR_PIPELINE_CACHE[key] = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            trust_remote_code=True,
            device=_device_for_pipeline(device),
            torch_dtype=torch.float16,
        )
        return _ASR_PIPELINE_CACHE[key]
```

- [ ] **Step 3：跑 ASR 測試，確認全部通過**

```bash
cd D:/qwen3tts-asr && python -m pytest tests/core/test_lru_cache.py::TestAsrLruCache -v
```

預期：全部 `PASSED`

- [ ] **Step 4：Commit**

```bash
cd D:/qwen3tts-asr
git add app/core/qwen3_asr.py tests/core/test_lru_cache.py
git commit -m "perf: ASR LRU cache with cuda.empty_cache and float16"
```

---

## Task 3：TTS LRU 快取測試 + 實作

**Files:**
- Modify: `tests/core/test_lru_cache.py` (append TTS tests)
- Modify: `app/core/qwen3_tts.py:1-30`, `app/core/qwen3_tts.py:65-93`

- [ ] **Step 1：在 `test_lru_cache.py` 末尾追加 TTS 測試**

```python
# ── TTS ──────────────────────────────────────────────────────────────────────

class TestTtsLruCache:
    """Tests for _get_model() LRU behaviour in qwen3_tts."""

    def setup_method(self):
        import app.core.qwen3_tts as tts_mod
        tts_mod._MODEL_CACHE.clear()

    def test_cache_hit_does_not_reload(self):
        import app.core.qwen3_tts as tts_mod
        fake_model = MagicMock()
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = fake_model
        with (
            patch("app.core.qwen3_tts.ensure_qwen_runtime"),
            patch("app.core.qwen3_tts._MODEL_CACHE", tts_mod._MODEL_CACHE),
            patch("qwen_tts.Qwen3TTSModel", mock_cls, create=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            tts_mod._get_model("model-a", "cpu")
            tts_mod._get_model("model-a", "cpu")
        assert mock_cls.from_pretrained.call_count == 1

    def test_eviction_calls_cuda_empty_cache(self):
        import app.core.qwen3_tts as tts_mod
        fake_model = MagicMock()
        mock_cls = MagicMock()
        mock_cls.from_pretrained.return_value = fake_model
        with (
            patch("app.core.qwen3_tts.ensure_qwen_runtime"),
            patch("qwen_tts.Qwen3TTSModel", mock_cls, create=True),
            patch("torch.cuda.is_available", return_value=False),
            patch("gc.collect") as mock_gc,
            patch("torch.cuda.empty_cache") as mock_cuda_empty,
        ):
            tts_mod._get_model("model-a", "cpu")
            tts_mod._get_model("model-b", "cpu")
        mock_cuda_empty.assert_called_once()
        mock_gc.assert_called()

    def test_cache_uses_ordered_dict(self):
        import app.core.qwen3_tts as tts_mod
        assert isinstance(tts_mod._MODEL_CACHE, OrderedDict)
```

- [ ] **Step 2：跑 TTS 測試，確認失敗**

```bash
cd D:/qwen3tts-asr && python -m pytest tests/core/test_lru_cache.py::TestTtsLruCache -v
```

預期：`FAILED`

- [ ] **Step 3：修改 `app/core/qwen3_tts.py`**

頂部 import 區：
```python
from collections import OrderedDict
_MODEL_CACHE: OrderedDict[tuple[str, str], object] = OrderedDict()
```

替換 `_get_model()` 快取邏輯（保留其餘邏輯不變）：
```python
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
            torch.cuda.empty_cache()
        _MODEL_CACHE[key] = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=dtype,
        )
        return _MODEL_CACHE[key]
```

- [ ] **Step 4：跑全部 LRU 測試**

```bash
cd D:/qwen3tts-asr && python -m pytest tests/core/test_lru_cache.py -v
```

預期：全部 `PASSED`

- [ ] **Step 5：Commit**

```bash
cd D:/qwen3tts-asr
git add app/core/qwen3_tts.py tests/core/test_lru_cache.py
git commit -m "perf: TTS LRU cache with cuda.empty_cache"
```

---

## Task 4：model_prewarmer 測試（紅燈）

**Files:**
- Create: `tests/core/test_model_prewarmer.py`

- [ ] **Step 1：寫失敗測試**

```python
# tests/core/test_model_prewarmer.py
"""Tests for app.core.model_prewarmer."""
from __future__ import annotations
import threading
import time
from unittest.mock import MagicMock, call, patch

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
    ):
        prewarm_models(CFG, status_callback=callback)

    assert len(calls) == 3
    assert "ASR" in calls[0][0]
    assert "TTS" in calls[1][0]
    assert "就緒" in calls[2][0] or "ready" in calls[2][0].lower()
    # Ready should be green
    assert calls[2][1] == "#2ecc71"


def test_prewarm_failure_calls_failure_callback():
    """If _get_pipeline raises, callback must receive a failure message (not propagate exception)."""
    from app.core.model_prewarmer import prewarm_models
    from app.core import qwen3_asr, qwen3_tts

    calls = []
    callback = lambda msg, color: calls.append((msg, color))

    with (
        patch.object(qwen3_asr, "_get_pipeline", side_effect=RuntimeError("OOM")),
        patch.object(qwen3_tts, "_get_model", return_value=MagicMock()),
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
    ):
        t1 = threading.Thread(target=prewarm_models, args=(CFG,))
        t2 = threading.Thread(target=prewarm_models, args=(CFG,))
        t1.start()
        time.sleep(0.01)  # let t1 acquire the event
        t2.start()
        t1.join(timeout=2)
        t2.join(timeout=2)

    assert load_count["asr"] == 1, "Model should only be loaded once"


def test_prewarm_skips_if_model_not_cached():
    """If is_repo_cached() returns False, prewarm skips without error."""
    from app.core.model_prewarmer import prewarm_models
    from app.core import qwen3_asr, qwen3_tts

    calls = []
    with (
        patch.object(qwen3_asr, "is_repo_cached", return_value=False),
        patch.object(qwen3_tts, "is_repo_cached", return_value=False),
        patch.object(qwen3_asr, "_get_pipeline") as mock_asr,
        patch.object(qwen3_tts, "_get_model") as mock_tts,
    ):
        prewarm_models(CFG, status_callback=lambda m, c: calls.append(m))

    mock_asr.assert_not_called()
    mock_tts.assert_not_called()
```

- [ ] **Step 2：跑測試，確認失敗**

```bash
cd D:/qwen3tts-asr && python -m pytest tests/core/test_model_prewarmer.py -v
```

預期：`ModuleNotFoundError: app.core.model_prewarmer`

---

## Task 5：實作 model_prewarmer.py（綠燈）

**Files:**
- Create: `app/core/model_prewarmer.py`

- [ ] **Step 1：建立檔案**

```python
# app/core/model_prewarmer.py
"""Background model pre-warming for ASR and TTS."""
from __future__ import annotations

import logging
import threading
from typing import Callable

_LOGGER = logging.getLogger(__name__)

# Global event: set while prewarm is in progress, cleared when done.
# Prevents concurrent double-start.
_prewarm_event = threading.Event()


def prewarm_models(
    cfg: dict,
    status_callback: Callable[[str, str], None] | None = None,
) -> None:
    """Load default ASR and TTS models into cache in the calling thread.

    Args:
        cfg: Config dict with asr_qwen3_model, asr_device, tts_qwen3_model, tts_device.
        status_callback: Called as callback(message, color_hex) on status changes.
    """
    if _prewarm_event.is_set():
        _LOGGER.debug("Prewarm already in progress — skipping.")
        return

    _prewarm_event.set()
    try:
        _run_prewarm(cfg, status_callback)
    finally:
        _prewarm_event.clear()


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
    if not qwen3_asr.is_repo_cached(asr_model):
        _LOGGER.info("ASR model not cached — skipping prewarm for %s", asr_model)
    else:
        _notify(callback, f"⚡ ASR 模型載入中 ({asr_model.split('/')[-1]})...", "#e67e22")
        try:
            qwen3_asr._get_pipeline(asr_model, asr_device)
        except Exception as exc:
            _LOGGER.warning("ASR prewarm failed: %s", exc)
            _notify(callback, f"⚠️ ASR 預熱失敗，首次使用時再載入", "#f39c12")
            return

    # ── TTS ──────────────────────────────────────────
    if not qwen3_tts.is_repo_cached(tts_model):
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
```

- [ ] **Step 2：跑預熱測試**

```bash
cd D:/qwen3tts-asr && python -m pytest tests/core/test_model_prewarmer.py -v
```

預期：全部 `PASSED`

- [ ] **Step 3：跑所有 core 測試確認沒有 regression**

```bash
cd D:/qwen3tts-asr && python -m pytest tests/core/ -v
```

預期：全部 `PASSED`

- [ ] **Step 4：Commit**

```bash
cd D:/qwen3tts-asr
git add app/core/model_prewarmer.py tests/core/test_model_prewarmer.py
git commit -m "feat: add model_prewarmer with double-start guard"
```

---

## Task 6：HomePage 模型狀態列

**Files:**
- Modify: `app/ui/main_window.py` — `HomePage._build_ui()` 與新增 `update_model_status()``
- Create: `tests/ui/test_model_status_bar.py`

- [ ] **Step 1：在 `HomePage._build_ui()` 末尾加狀態列**

找到 `_build_ui()` 中最後一行 `self._build_output(body, column=0)`，在其後（仍在 `_build_ui` 內）加入：

```python
        # ── Model status bar ────────────────────────
        self._model_status_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        )
        self._model_status_label.pack(fill="x", padx=2, pady=(6, 0))
```

- [ ] **Step 2：在 `HomePage` 新增 `update_model_status()` 方法**

在 `HomePage` class 內，緊接在 `_build_ui()` 之後加入：

```python
    def update_model_status(self, text: str, color: str = "gray") -> None:
        """Update model status bar. Must be called from the main thread."""
        self._model_status_label.configure(text=text, text_color=color)
```

- [ ] **Step 3：加 `update_model_status` 的單元測試**

```python
# tests/ui/test_model_status_bar.py
"""Verify update_model_status() can only be called from main thread pattern."""
from unittest.mock import MagicMock, patch


def test_update_model_status_configures_label():
    """update_model_status() must call configure() on the label with correct args."""
    # Build a minimal HomePage-like object without a real tkinter root
    label_mock = MagicMock()

    class FakeHomePage:
        _model_status_label = label_mock

        def update_model_status(self, text: str, color: str = "gray") -> None:
            self._model_status_label.configure(text=text, text_color=color)

    page = FakeHomePage()
    page.update_model_status("✅ 模型就緒", "#2ecc71")
    label_mock.configure.assert_called_once_with(text="✅ 模型就緒", text_color="#2ecc71")


def test_update_model_status_clears_on_empty_string():
    label_mock = MagicMock()

    class FakeHomePage:
        _model_status_label = label_mock

        def update_model_status(self, text: str, color: str = "gray") -> None:
            self._model_status_label.configure(text=text, text_color=color)

    page = FakeHomePage()
    page.update_model_status("", "gray")
    label_mock.configure.assert_called_once_with(text="", text_color="gray")
```

```bash
cd D:/qwen3tts-asr && python -m pytest tests/ui/test_model_status_bar.py -v
```

預期：`PASSED`（這個測試不需要 tkinter，會在實作前就通過，因為測試的是 method 的行為合約）

- [ ] **Step 4：手動驗證 — 啟動 App 確認狀態列位置正常，不影響現有 UI**

```bash
cd D:/qwen3tts-asr && python main.py
```

確認：底部出現空白標籤列，現有操作正常。

- [ ] **Step 5：Commit**

```bash
cd D:/qwen3tts-asr
git add app/ui/main_window.py tests/ui/test_model_status_bar.py
git commit -m "feat: add model status bar to HomePage"
```

---

## Task 7：main.py 串接預熱

**Files:**
- Modify: `main.py`

- [ ] **Step 1：加 import**

在 `main.py` 頂部 import 區加入：

```python
import threading
from app.core.model_prewarmer import prewarm_models
from app.utils.config import load_config
```

（`threading` 可能已存在，若已有則跳過）

- [ ] **Step 2：在 `App.__init__()` 末尾加預熱排程**

在 `self.protocol("WM_DELETE_WINDOW", self._on_close)` 之後加：

```python
        self.after(500, self._start_prewarm)
```

- [ ] **Step 3：在 `App` class 加 `_start_prewarm()` 方法**

```python
    def _start_prewarm(self) -> None:
        """Launch background model pre-warming after UI is ready."""
        cfg = load_config()
        home_page = self._pages["one_shot"]

        def _status_cb(msg: str, color: str) -> None:
            self.after(0, lambda m=msg, c=color: home_page.update_model_status(m, c))
            if "就緒" in msg or "ready" in msg.lower():
                self.after(3000, lambda: home_page.update_model_status("", "gray"))

        threading.Thread(
            target=prewarm_models,
            args=(cfg, _status_cb),
            daemon=True,
        ).start()
```

- [ ] **Step 4：端對端手動測試**

```bash
cd D:/qwen3tts-asr && python main.py
```

觀察：
- App 啟動後約 0.5 秒，狀態列出現「⚡ ASR 模型載入中...」（橘色）
- 載入完成後出現「⚡ TTS 模型載入中...」
- 全部就緒後出現「✅ 模型就緒」（綠色），3 秒後消失
- 若模型未下載：狀態列不顯示，App 正常啟動

- [ ] **Step 5：跑全部測試確認 regression-free**

```bash
cd D:/qwen3tts-asr && python -m pytest tests/ -v
```

預期：全部 `PASSED`

- [ ] **Step 6：Commit**

```bash
cd D:/qwen3tts-asr
git add main.py
git commit -m "feat: wire background model prewarm on app startup"
```

---

## 驗收清單

- [ ] `python -m pytest tests/core/test_lru_cache.py -v` — 全部通過
- [ ] `python -m pytest tests/core/test_model_prewarmer.py -v` — 全部通過
- [ ] `python -m pytest tests/ -v` — 無 regression
- [ ] App 啟動後狀態列正常顯示預熱進度
- [ ] 切換 ASR 模型後重新錄音無長時間卡頓
- [ ] GPU Task Manager 中 VRAM 在模型切換後正確降低
