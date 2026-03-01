# AGENTS.md

本文件定義本專案的多 Agent 並行協作規範。目標是讓多位 Agent 同時開發時，降低衝突、避免回滾他人變更、並維持可驗證的交付品質。

## 1) 基本原則

- 僅修改自己被指派的檔案。
- 發現其他檔案有同時修改時，直接忽略，不回滾、不格式化、不順手重構。
- 不可用破壞性 git 操作覆蓋他人工作（例如 `reset --hard`、`checkout --`）。
- 任何跨執行緒 UI 更新，必須透過 `root.after(0, callback)` 回到主執行緒。

## 2) 並行分工（建議）

- UI Agent
  - 負責：視窗、按鈕、頁面互動、錯誤提示呈現、主執行緒 UI 安全
  - 主要檔案：`app/ui/*.py`
- Core Agent
  - 負責：錄音、轉寫、潤稿、翻譯流程與狀態機
  - 主要檔案：`app/core/*.py`
- Data Agent
  - 負責：設定與資料存取（config/history/dictionary）
  - 主要檔案：`app/utils/*.py`, `*.json`
- Integration Agent
  - 負責：熱鍵、剪貼簿、自動貼上與前景視窗整合
  - 主要檔案：`app/core/hotkey.py`, `app/utils/clipboard.py`

## 3) 檔案所有權範例（本輪）

- UI/文件 Agent 擁有：
  - `app/ui/main_window.py`
  - `app/ui/history_page.py`
  - `app/ui/dict_page.py`
  - `AGENTS.md`
- 其他 Agent 不應編輯上述檔案；若必須改動，先同步再做。

## 4) 變更策略

- 小步提交：每個需求點對應明確修改，避免一次大範圍改動。
- 對外介面優先穩定：保留既有 class / method 名稱，避免連鎖影響。
- 錯誤可見化：I/O 或 API 失敗時，需有 UI 訊息，不可靜默失敗。
- 狀態守衛：錄音/轉寫進行中應阻止不安全操作（例如手動潤稿/翻譯）。

## 5) 驗證流程（每位 Agent 交付前必做）

1. 語法驗證
   - `python -m py_compile app/ui/main_window.py app/ui/history_page.py app/ui/dict_page.py`
2. 基本手動流程
   - 啟動 `python main.py`
   - 測試錄音開始/停止與狀態切換
   - 錄音中嘗試手動潤稿/翻譯，確認被阻擋且有提示
   - 檢查「儲存整理後」可存出 txt
   - 在歷史頁測試刷新、刪除、匯出（包含失敗提示）
   - 在字典頁測試載入與儲存（包含失敗提示）
3. 回報格式
   - 修改摘要（檔案 + 功能點）
   - 驗證結果（通過/未執行）
   - 已知風險與待補測項

## 6) 衝突處理

- 若發現同檔案被其他 Agent 同時改動：
  - 先停止編輯並比對差異來源
  - 只保留自己需求必要變更
  - 不得直接覆蓋整檔

## 7) 版本註記

- 0.2.0（2026-02-28）
  - 新增並行 Agent 檔案所有權規範
  - 新增交付前驗證流程與回報格式