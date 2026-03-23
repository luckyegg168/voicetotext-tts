# App 效能優化設計文件

**日期：** 2026-03-23
**範圍：** 效能優化（模型載入速度）
**優先順序：** 效能 > UI/UX > 功能 > 架構

---

## 背景與問題

Steven's Voice Workspace 使用 Qwen3-ASR 和 Qwen3-TTS 本地模型進行語音轉文字和文字轉語音。目前有兩個主要效能痛點：

1. **App 啟動後首次使用卡頓**：模型在第一次實際呼叫時才載入，使用者按下錄音後需等待較長時間。
2. **切換模型時重新載入**：`_MODEL_LIMIT = 1` 導致切換模型時清空整個快取，原本已載入的模型被丟棄。

執行環境：Windows、NVIDIA GPU 8–12 GB VRAM。Qwen3-ASR-0.6B ≈ 1.5 GB、Qwen3-TTS-0.6B ≈ 1.5 GB，同時持有兩個模型安全無虞。

---

## 方案選擇

採用 **方案 B：擴大模型快取 + 背景預熱**，理由：

- 收益最大：同時解決啟動慢與切換慢兩個問題
- 改動範圍可控：不需重構呼叫鏈
- 硬體安全：8–12 GB VRAM 同時持有兩個 0.6B 模型綽綽有餘

---

## 設計

### 1. LRU 模型快取（`app/core/qwen3_asr.py`、`app/core/qwen3_tts.py`）

**目標：** 切換模型時只踢最少使用的，而非全清。

**改動：**
- `_MODEL_CACHE` 型別從 `dict` 改為 `collections.OrderedDict`
- `_MODEL_LIMIT` 從 `1` 調整為 `2`
- `_get_pipeline()` / `_get_model()` 內加入 LRU 邏輯：
  - 快取命中 → `move_to_end(key)`（標記為最近使用）
  - 快取未命中且已滿 → `popitem(last=False)`（踢最舊的）+ `gc.collect()`
  - 載入新模型後存入快取

**影響範圍：** `qwen3_asr.py`、`qwen3_tts.py` 各約 10 行。

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
self.after(500, self._start_prewarm)

def _start_prewarm(self):
    cfg = load_config()
    threading.Thread(
        target=prewarm_models,
        args=(cfg, self._on_prewarm_status),
        daemon=True,
    ).start()
```

**規範：**
- `daemon=True`：App 關閉時 thread 自動結束
- `after(500, ...)` 確保 UI 完整渲染後再啟動預熱
- 預熱失敗僅 log，不彈錯誤對話框，不影響正常使用

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
- `_on_prewarm_status(msg)` 在 `App` 層接收 callback，透過 `self.after(0, ...)` 轉回主執行緒
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

- [ ] `test_lru_cache`：驗證超過 limit 時踢最舊、命中時更新順序
- [ ] `test_prewarm_success`：mock `_get_pipeline` / `_get_model`，驗證 callback 順序與內容
- [ ] `test_prewarm_failure`：mock 拋出 RuntimeError，驗證不影響 App 啟動
- [ ] `test_status_bar_update`：驗證 `update_model_status()` 在主執行緒被呼叫

---

## 不在此次範圍

- UI/UX 重設計（列為下一輪優化）
- 功能新增（列為下一輪）
- 架構重構（列為下一輪）
- 方案 C（持久推論服務）：保留為未來選項
