@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

set FW_OK=0
set OLLAMA_INSTALLED=0

echo.
echo  ================================================
echo   Steven's Voice Workspace  - Install
echo  ================================================
echo.

:: ════════════════════════════════════════════════
:: 1/4  檢查 Python 版本
:: ════════════════════════════════════════════════
echo [1/4] 檢查 Python 環境...
python --version >nul 2>&1
if errorlevel 1 goto :no_python

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PY_VER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% LSS 3 goto :py_old
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 goto :py_old
echo  OK  Python %PY_VER%
echo.
goto :step2

:no_python
echo  [錯誤] 找不到 Python，請安裝 3.10 以上版本。
echo         https://www.python.org/downloads/
echo         安裝時請勾選「Add Python to PATH」
echo.
pause & exit /b 1

:py_old
echo  [錯誤] Python 版本過舊 (偵測到 %PY_VER%)，需要 3.10 以上。
pause & exit /b 1

:: ════════════════════════════════════════════════
:: 2/4  升級 pip + 安裝主要套件
:: ════════════════════════════════════════════════
:step2
echo [2/4] 安裝主要套件...
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [錯誤] 套件安裝失敗，請檢查上方訊息。
    pause & exit /b 1
)
echo  OK  主要套件安裝完成
echo.

:: ════════════════════════════════════════════════
:: 3/4  faster-whisper（已併入 requirements.txt）
:: ════════════════════════════════════════════════
echo [3/4] 檢查本地語音辨識 (faster-whisper)
echo.
python -c "import faster_whisper" >nul 2>&1
if errorlevel 1 (
    echo  [警告] faster-whisper 尚未可用。
    echo         請重新執行：python -m pip install -r requirements.txt
    goto :step4
)
echo  OK  faster-whisper 已安裝（由 requirements.txt 提供）
set FW_OK=1

:: ════════════════════════════════════════════════
:: 4/4  Ollama（本地 LLM）
:: ════════════════════════════════════════════════
:step4
echo.
echo [4/4] 本地大型語言模型 (Ollama)
echo.
echo  是否安裝 Ollama？(本機執行 Qwen / Llama 等，不需 API Key)

set INSTALL_OLLAMA=
set /p INSTALL_OLLAMA=  安裝? [Y/N] (Enter=N): 
if "%INSTALL_OLLAMA%"=="" set INSTALL_OLLAMA=N
if /i not "%INSTALL_OLLAMA%"=="Y" goto :skip_ollama

:: 已安裝？
ollama --version >nul 2>&1
if not errorlevel 1 (
    echo  OK  Ollama 已安裝，略過下載。
    set OLLAMA_INSTALLED=1
    goto :choose_model
)

:: 下載安裝程式
echo.
echo  下載 Ollama 安裝程式...
set OLLAMA_SETUP=%TEMP%\OllamaSetup.exe
curl -L --progress-bar -o "%OLLAMA_SETUP%" https://ollama.com/download/OllamaSetup.exe
if errorlevel 1 (
    echo  [錯誤] 下載失敗，請手動前往 https://ollama.com 安裝。
    goto :skip_ollama
)

echo  執行安裝程式（安裝完後請回到此視窗繼續）...
start /wait "Ollama Setup" "%OLLAMA_SETUP%"
del "%OLLAMA_SETUP%" >nul 2>&1

ollama --version >nul 2>&1
if not errorlevel 1 (
    echo  OK  Ollama 安裝完成
    set OLLAMA_INSTALLED=1
    goto :choose_model
)
echo  [警告] 安裝後仍無法偵測到 Ollama，請重新開機後再試。
goto :skip_ollama

:: 選擇下載模型
:choose_model
echo.
echo  ── 下載 Ollama 模型 ───────────────────────────────────
echo   1) qwen2.5:7b   中文優秀，推薦首選  (約 4.7 GB)
echo   2) qwen2.5:3b   輕量版，低顯存      (約 2.0 GB)
echo   3) gemma3:4b    Google 多語言       (約 2.5 GB)
echo   4) llama3.2     Meta 英文強         (約 2.0 GB)
echo   5) 輸入自訂模型名稱
echo   6) 略過
echo.

set OLLAMA_MODEL=
set /p OLLAMA_MODEL=  選擇 [1-6] (Enter=6): 
if "%OLLAMA_MODEL%"=="" set OLLAMA_MODEL=6

if "%OLLAMA_MODEL%"=="1" ollama pull qwen2.5:7b
if "%OLLAMA_MODEL%"=="2" ollama pull qwen2.5:3b
if "%OLLAMA_MODEL%"=="3" ollama pull gemma3:4b
if "%OLLAMA_MODEL%"=="4" ollama pull llama3.2
if "%OLLAMA_MODEL%"=="5" goto :custom_model
if "%OLLAMA_MODEL%"=="6" echo  略過模型下載。
goto :skip_ollama

:custom_model
set CUSTOM_MODEL=
set /p CUSTOM_MODEL=  輸入模型名稱 (例: mistral:7b): 
if "%CUSTOM_MODEL%"=="" (
    echo  未輸入，略過。
    goto :skip_ollama
)
ollama pull %CUSTOM_MODEL%

:skip_ollama
echo.

:: ════════════════════════════════════════════════
:: 安裝摘要
:: ════════════════════════════════════════════════
echo.
echo  ================================================
echo   安裝完成！
echo  ================================================
echo.
echo   主要套件         OK  已安裝

if "%FW_OK%"=="1" (
    echo   faster-whisper   OK  本機 GPU 轉寫
) else (
    echo   faster-whisper   --  未安裝
)
if "%OLLAMA_INSTALLED%"=="1" (
    echo   Ollama           OK  本機 LLM
) else (
    echo   Ollama           --  未安裝
)

echo.
echo  執行 start.cmd 啟動程式
echo  ================================================
echo.

set START_NOW=
set /p START_NOW=  立即啟動？ [Y/N] (Enter=Y): 
if "%START_NOW%"=="" set START_NOW=Y
if /i "%START_NOW%"=="Y" python main.py

echo.
pause
