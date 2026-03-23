# App 效能優化設計文件

**日期：** 2026-03-23
**範圍：** 效能優化（模型載入速度）
**優先順序：** 效能 > UI/UX > 功能 > 架構

---

## 背景與問題

Steven's Voice Workspace 使用 Qwen3-ASR 和 Qwen3-TTS 本地模型進行語音轉文字和文字轉語音。目前有兩個主要效能痛點：

1. **App 啟動後首次使用卡頓**：模型在第一次實際呼叫時才載入，使用者按下錄音後需等待較長時間。
2. **切換模型時重新載入**：`_MODEL_LIMIT = 1` 導致切換模型時清空整個快取，原本已載入的模型被丟棄。

執行環境：Windows、NVIDIA GPU 8–12 GB VRAM。

VRAM 估算（`_MODEL_LIMIT = 1` 時的穩態）：
- Qwen3-ASR-0.6B：目前 `qwen3_asr.py` 的 `pipeline()` 未指定 `torch_dtype`，預設 float32，約 **2.4 GB**。本次優化同步改為 `torch_dtype=torch.float16`，降至 ≈ 1.2 GB。
- Qwen3-TTS-0.6B：已使用 float16，約 **1.5 GB**（`qwen3_tts.py` 第 78 行）。
- 穩態合計 ≈ 2.7 GB（1 ASR + 1 TTS，分屬不同 cache），對 8 GB 卡安全無虞，餘留 >5 GB 供推論使用。

> 注意：ASR 與 TTS 使用**各自獨立的 cache**（`_ASR_PIPELINE_CACHE` / `_MODEL_CACHE`），因此兩個 cache 的模型可同時存在 VRAM。`_MODEL_LIMIT` 控制的是「同類型快取最多持有幾個不同模型版本」，維持 1 即可避免同類型切換時重複佔用 VRAM。

---

## 方案選擇

採用 **方案 B：擴大模型快取 + 背景預熱**，理由：

- 收益最大：同時解決啟動慢與切換慢兩個問題
- 改動範圍可控：不需重構呼叫鏈
- 硬體安全：ASR + TTS 各自獨立 cache，limit=1 時穩態 ≈ 2.7 GB，對 8 GB 卡安全無虞

---

## 設計

### 1. LRU 模型快取（`app/core/qwen3_asr.py`、`app/core/qwen3_tts.py`）

**目標：** 切換模型時只踢最少使用的，而非全清。

**改動（兩個檔案均需修改）：**

`qwen3_asr.py`（變數名 `_ASR_PIPELINE_CACHE` / `_ASR_PIPELINE_LIMIT`）：
- `_ASR_PIPELINE_CACHE: dict` → `OrderedDict`
- `_ASR_PIPELINE_LIMIT` 維持 `1`（避免同類型兩模型並存超過 VRAM）
- `_get_pipeline()` 舊的 `_ASR_PIPELINE_CACHE.clear()` 替換為精確單項 LRU 邏輯（見下）
- `pipeline()` 呼叫新增 `torch_dtype=torch.float16`，將 ASR VRAM 從 ~2.4 GB 降至 ~1.2 GB

`qwen3_tts.py`（變數名 `_MODEL_CACHE` / `_MODEL_LIMIT`）：
- `_MODEL_CACHE: dict` → `OrderedDict`
- `_MODEL_LIMIT` 維持 `1`
- `_get_model()` 舊的 `_MODEL_CACHE.clear()` 替換為精確單項 LRU 邏輯（見下）

**兩個檔案共用的 LRU 邏輯模式：**
```python
with _LOCK:
    if key in _CACHE:
        _CACHE.move_to_end(key)       # 命中 → 標記為最近使用
        return _CACHE[key]
    if len(_CACHE) >= _LIMIT:
        _CACHE.popitem(last=False)    # 踢最舊的
        gc.collect()
        torch.cuda.empty_cache()      # 必須呼叫，否則 CUDA VRAM 不釋放
    _CACHE[key] = load_model(...)
    return _CACHE[key]
```

**影響範圍：** `qwen3_asr.py`、`qwen3_tts.py` 各約 12 行。

---

### 2. 背景預熱（`app/core/model_prewarmer.py`、`main.py`）

**目標：** UI 渲染完畢後立即在背景載入預設模型，使用者操作時模型已就緒。

**新增 `app/core/model_prewarmer.py`：**

```
prewarm_models(cfg, status_callback=None)
  ├─ 從 cfg 讀取 asr_qwen3_model、tts_qwen3_model、asr_device、tts_device
  ├─ 呼叫 qwen3_asr._get_pipeline(model_id, device)  → 觸發快取預熱
  ├─ 呼叫 qwen3_tts._get_model(model_id, device)     → 觸發快取預熱
  └─ 各步驟透過 status_callback 回報進度（成功/失敗）
```

**`main.py` 的 `App.__init__()` 新增：**

```python
def __init__(self):
    super().__init__()
    # ... 現有初始化 ...
    _cuda_setup()
    self._build_ui()                          # ← HomePage 在此建立
    self.protocol("WM_DELETE_WINDOW", self._on_close)
    self.after(500, self._start_prewarm)      # ← 必須在 _build_ui() 之後

def _start_prewarm(self):
    cfg = load_config()
    home_page = self._pages["one_shot"]       # ← HomePage 已存在，安全取用
    threading.Thread(
        target=prewarm_models,
        args=(cfg, lambda msg, color: self.after(0, lambda: home_page.update_model_status(msg, color))),
        daemon=True,
    ).start()
```

**規範：**
- `self.after(500, self._start_prewarm)` **必須在 `_build_ui()` 返回之後**排程，確保 `self._pages["one_shot"]`（`HomePage`）已存在
- `daemon=True`：App 關閉時 thread 自動結束
- 預熱失敗僅 log，不彈錯誤對話框，不影響正常使用
- **防止重複啟動：** `model_prewarmer.py` 維護一個 `_prewarm_in_progress: threading.Event`；`prewarm_models()` 開始前呼叫 `event.set()`，結束後 `event.clear()`；`_start_prewarm()` 若 event 已 set 則直接返回
- 「就緒」狀態的 3 秒自動隱藏透過 `self.after(3000, lambda: home_page.update_model_status("", ""))` 排程，走主執行緒，避免 `TclError`

---

### 3. UI 模型狀態列（`app/ui/main_window.py` → `HomePage`）

**目標：** 讓使用者知道模型目前的載入狀態，消除「為什麼按了沒反應」的困惑。

**新增 `ModelStatusBar` 元件（內嵌於 `HomePage`）：**

| 狀態 | 顯示文字 | 顏色 |
|------|---------|------|
| 預熱中 | `⚡ ASR 模型載入中...` | 橘色 |
| 就緒 | `✅ 模型就緒` | 綠色，3 秒後自動隱藏 |
| 失敗 | `⚠️ 模型預熱失敗，首次使用時再載入` | 黃色 |

**技術實作：**
- 預熱 thread 的所有 UI 更新**必須**透過 `self.after(0, callback)` 排程回主執行緒，絕不直接從背景 thread 操作 tkinter widget（符合專案 threading 規範）
- `_on_prewarm_status(msg)` 在 `App` 層接收 callback，呼叫 `self.after(0, lambda: self._pages["one_shot"].update_model_status(...))`
- `HomePage` 暴露 `update_model_status(text, color)` 方法供外部呼叫
- 狀態列放置於 `HomePage` 底部，獨立 `CTkLabel`，不影響現有 UI 佈局

---

## 資料流

```
App 啟動
  │
  ├─ _cuda_setup()
  ├─ _build_ui()         ← UI 可用
  └─ after(500ms)
       └─ Thread: prewarm_models(cfg)
            ├─ status_callback("⚡ ASR 載入中")  → after(0) → ModelStatusBar
            ├─ qwen3_asr._get_pipeline()          ← 填入 LRU cache
            ├─ status_callback("⚡ TTS 載入中")  → after(0) → ModelStatusBar
            ├─ qwen3_tts._get_model()             ← 填入 LRU cache
            └─ status_callback("✅ 就緒")        → after(0) → ModelStatusBar (3s 後隱藏)

使用者按錄音
  └─ transcribe_audio() → qwen3_asr._get_pipeline()  ← 快取命中，立即返回
```

---

## 錯誤處理

- 預熱中 GPU OOM：捕捉 `RuntimeError`，callback 回報失敗狀態，快取不留損壞物件
- 預熱中 ImportError（缺套件）：同上，使用者可在設定頁看到提示
- 模型下載未完成：`is_repo_cached()` 回傳 False 時跳過預熱，不報錯

---

## 測試計畫

- [ ] `test_lru_cache_eviction`：驗證超過 limit 時踢最舊、命中時更新順序；並 mock `torch.cuda.empty_cache` 驗證確實被呼叫
- [ ] `test_lru_cache_hit`：快取命中時 `move_to_end` 被呼叫，不觸發載入
- [ ] `test_prewarm_success`：mock `_get_pipeline` / `_get_model`，驗證 callback 順序與內容
- [ ] `test_prewarm_failure`：mock 拋出 RuntimeError，驗證 callback 回報失敗、App 不崩潰
- [ ] `test_prewarm_no_double_start`：連續呼叫兩次 `_start_prewarm()`，驗證第二次被 `_prewarm_in_progress` 攔截，模型只載入一次
- [ ] `test_status_bar_update`：驗證 `update_model_status()` 透過 `after(0, ...)` 在主執行緒被呼叫，而非直接從 thread 呼叫

---

## 不在此次範圍

- UI/UX 重設計（列為下一輪優化）
- 功能新增（列為下一輪）
- 架構重構（列為下一輪）
- 方案 C（持久推論服務）：保留為未來選項
