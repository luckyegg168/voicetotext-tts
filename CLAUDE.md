# VoiceToText-TTS — AI Agent 專案說明

> 適用於所有 AI（Claude、GPT、Gemini、Cursor 等）快速理解本專案

## 專案概述

本專案是 **Steven's Voice Workspace** — 一個 Windows 語音轉文字工具。
核心理念：「自然說話，快速成文」——使用者按下熱鍵說話，AI 自動轉寫並潤稿，直接貼到任何應用程式。

**平台**：Windows 10+
**語言**：Python 3.10+
**UI 框架**：customtkinter（現代深色/淺色主題）
**AI 後端**：OpenAI API（Whisper 轉寫 + GPT-4o 潤稿）

---

## 目錄結構

```
voicetotext-tts/
├── CLAUDE.md               # 本文件（AI 理解入口）
├── agents.md               # 多 Agent 協作說明
├── main.py                 # 程式進入點
├── requirements.txt        # Python 相依套件
├── config.json             # 使用者設定（API Key、語言等）
├── history.json            # 歷史紀錄資料
├── app/
│   ├── __init__.py
│   ├── ui/
│   │   ├── main_window.py  # 主視窗（首頁）
│   │   ├── history_page.py # 歷史紀錄頁面
│   │   └── dict_page.py    # 字典頁面
│   ├── core/
│   │   ├── recorder.py     # 麥克風錄音
│   │   ├── transcriber.py  # Whisper API 轉寫
│   │   ├── polisher.py     # GPT 潤稿
│   │   └── hotkey.py       # 全域熱鍵監聽
│   └── utils/
│       ├── clipboard.py    # 跨應用剪貼簿貼上
│       ├── storage.py      # JSON 資料持久化
│       └── config.py       # 設定讀寫
└── assets/
    └── icon.ico            # 系統匣圖示
```

---

## 核心功能規格

### 1. 全域熱鍵
- **快捷鍵**：`Ctrl + Shift + Space`
- **行為**：第一次按下 → 開始錄音；再按一次 → 停止錄音並開始處理
- **實作**：`keyboard` 套件，背景執行緒監聽
- **狀態機**：`待機中` → `錄音中` → `轉寫中` → `潤稿中` → `完成`

### 2. 音訊錄音
- **套件**：`sounddevice` + `numpy`
- **格式**：WAV，16kHz，mono
- **行為**：即時串流錄音，停止後存為暫存檔

### 3. 語音轉寫
- **API**：OpenAI Whisper (`whisper-1`)
- **語言**：支援自動偵測或指定語言（zh/en/ja 等）
- **輸出**：原始轉寫文字

### 4. AI 潤稿
- **API**：OpenAI Chat Completions (`gpt-4o-mini` 或 `gpt-4o`)
- **情境模板**：
  - 通用：清理口語、修正標點
  - 社群貼文：轉換為臉書/IG 友善格式
  - 會議記錄：條列式整理
  - 信件：正式書信格式
- **輸出語言**：原語言 / 繁體中文 / 英文

### 5. 跨應用自動貼上
- **行為**：處理完成後，自動將潤稿結果複製到剪貼簿，並模擬 `Ctrl+V` 貼到前景視窗
- **套件**：`pyperclip` + `pyautogui`

### 6. 統計資料
- 累積字數（所有已貼上文字）
- 錄音次數（本機歷史計數）

### 7. 歷史紀錄
- 儲存每次的原文 + 潤稿後結果
- 顯示時間戳記
- 支援複製、刪除

### 8. 字典功能
- 自訂詞彙替換（例：口頭禪、專有名詞修正）
- 在轉寫後、潤稿前套用

---

## 設定檔格式（config.json）

```json
{
  "openai_api_key": "sk-...",
  "openrouter_api_key": "sk-or-...",
  "api_provider": "openai",
  "polish_model": "gpt-4o-mini",
  "transcription_language": "auto",
  "output_language": "original",
  "template": "general",
  "auto_switch_template": true,
  "auto_paste": true,
  "hotkey": "ctrl+shift+space",
  "theme": "dark"
}
```

### API 提供者說明

| 提供者 | api_provider 值 | 說明 |
|--------|----------------|------|
| OpenAI | `"openai"` | 直接使用 OpenAI API（Whisper + GPT） |
| OpenRouter | `"openrouter"` | 透過 OpenRouter 存取多種模型 |

OpenRouter 可用模型（`polish_model` 欄位）：
- `openai/gpt-4o-mini`、`openai/gpt-4o`
- `anthropic/claude-3-haiku`、`anthropic/claude-3.5-sonnet`
- `google/gemini-flash-1.5`
- `meta-llama/llama-3.1-8b-instruct`

---

## 歷史紀錄格式（history.json）

```json
[
  {
    "id": "uuid",
    "timestamp": "2026-02-26T10:00:00",
    "original": "原始轉寫文字",
    "polished": "潤稿後文字",
    "template": "general",
    "word_count": 42
  }
]
```

---

## 情境模板 System Prompt 規格

### 通用（general）
```
你是文字潤稿助手。將以下口語錄音轉寫文字整理成通順的書面文字，
修正標點符號，移除重複詞語，但保留原意。直接輸出整理後文字，不要加任何說明。
```

### 社群貼文（social）
```
你是社群媒體內容創作助手。將以下文字改寫成適合臉書/Instagram 的貼文，
加入適當換行，語氣自然親切。直接輸出結果，不要加任何說明。
```

### 會議記錄（meeting）
```
你是會議記錄助手。將以下口語內容整理成條列式會議記錄，
包含重要決議、行動項目。直接輸出結果，不要加任何說明。
```

### 信件（email）
```
你是專業書信助手。將以下內容改寫成正式電子郵件格式，
語氣專業有禮。直接輸出結果，不要加任何說明。
```

---

## 安裝與執行

```bash
# 安裝相依套件
pip install -r requirements.txt

# 執行
python main.py
```

---

## 相依套件（requirements.txt）

```
customtkinter>=5.2.0
openai>=1.0.0
sounddevice>=0.4.6
numpy>=1.24.0
scipy>=1.10.0
keyboard>=0.13.5
pyperclip>=1.8.2
pyautogui>=0.9.54
Pillow>=10.0.0
faster-whisper>=1.0.0
nvidia-cublas-cu12
nvidia-cudnn-cu12
```

---

## AI 協作注意事項

1. **不要修改** `config.json` 和 `history.json` 的格式，向後相容很重要
2. **熱鍵監聽** 在背景執行緒，UI 更新必須使用 `after()` 方法回主執行緒
3. **錄音狀態** 以 `app/core/recorder.py` 的 `RecordingState` enum 為準
4. **所有 OpenAI API 呼叫** 都在背景執行緒進行，避免 UI 凍結
5. **Windows 路徑** 使用 `pathlib.Path` 處理，不要硬編碼分隔符

---

## 已知限制

- 需要麥克風存取權限
- OpenAI API 需要網路連線
- `keyboard` 套件在 Windows 需要管理員權限（或以一般使用者執行，部分功能可能受限）

---

*本文件由 Claude Code 自動生成，請在功能有重大變更時同步更新。*
