@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

set "LOCAL_TMP=%CD%\backups\tmp"
if not exist "%LOCAL_TMP%" mkdir "%LOCAL_TMP%" >nul 2>&1
set "TMP=%LOCAL_TMP%"
set "TEMP=%LOCAL_TMP%"

set "PY_SPEC="
set "PY_EXE="
set "PY_MAJOR="
set "PY_MINOR="
set "PY_PATCH="
set "TEMP_PY=%TEMP%\_svw_py_exe.txt"
set "TEMP_PY_VER=%TEMP%\_svw_py_ver.txt"
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

echo.
echo ==================================================
echo  Steven's Voice Workspace - Install
echo  (Qwen3-ASR + Qwen3-TTS)
echo ==================================================
echo.

REM -----------------------------
REM 1) Select Python interpreter
REM -----------------------------
echo [1/7] Selecting Python interpreter...

for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PY_SPEC (
    py -%%V -c "import sys; print('%%d %%d %%d' %% (sys.version_info[0], sys.version_info[1], sys.version_info[2]))" > "%TEMP_PY_VER%" 2>nul
    if not errorlevel 1 (
      set "PY_SPEC=-%%V"
    )
  )
)

if not defined PY_SPEC goto :no_python

for /f "tokens=1,2,3" %%a in ('type "%TEMP_PY_VER%"') do (
  set "PY_MAJOR=%%a"
  set "PY_MINOR=%%b"
  set "PY_PATCH=%%c"
)
del "%TEMP_PY_VER%" >nul 2>&1

if "%PY_MAJOR%"=="" goto :no_python
if %PY_MAJOR% LSS 3 goto :py_old
if %PY_MAJOR% EQU 3 if %PY_MINOR% LSS 10 goto :py_old
if %PY_MAJOR% EQU 3 if %PY_MINOR% GTR 13 goto :py_new

py %PY_SPEC% -c "import sys; print(sys.executable)" > "%TEMP_PY%" 2>nul
if not errorlevel 1 set /p PY_EXE=<"%TEMP_PY%"
del "%TEMP_PY%" >nul 2>&1

echo   OK  Using Python %PY_MAJOR%.%PY_MINOR%.%PY_PATCH%
if defined PY_EXE echo       %PY_EXE%
echo.

REM -----------------------------
REM 2) Create/Reuse venv + install deps
REM -----------------------------
echo [2/7] Preparing virtual environment...
if not exist "%VENV_PY%" (
  py %PY_SPEC% -m venv .venv
  if errorlevel 1 goto :venv_failed
)

"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
  echo   [Info] pip not found in venv, attempting ensurepip recovery...
  "%VENV_PY%" -m ensurepip --upgrade --default-pip >nul 2>&1
)

"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
  echo   [Info] ensurepip recovery failed, bootstrapping pip via py launcher...
  py %PY_SPEC% -m pip --python "%VENV_PY%" install pip >nul 2>&1
)

"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 goto :venv_pip_failed

echo   OK  venv ready: %VENV_PY%
echo.

echo [2/7] Installing requirements...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :pip_failed

"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed

"%VENV_PY%" -m pip install -r requirements-qwen.txt
if errorlevel 1 goto :pip_failed

REM --- Ensure transformers >= 4.58.0 (required for Qwen3-ASR architecture) ---
"%VENV_PY%" -c "import sys,importlib.metadata as m; v=m.version('transformers'); p=list(map(int,v.split('.')[:2])); sys.exit(0 if (p[0],p[1])>=(4,58) else 1)" >nul 2>&1
if errorlevel 1 (
  echo   transformers ^< 4.58.0 detected - installing from source for Qwen3-ASR support...
  "%VENV_PY%" -m pip install "git+https://github.com/huggingface/transformers.git"
  if errorlevel 1 (
    echo   [Warning] Failed to install transformers from source. Qwen3-ASR may not work.
  ) else (
    echo   OK  transformers installed from source.
  )
) else (
  echo   OK  transformers version OK.
)

REM --- Install CUDA PyTorch if GPU is present ---
nvidia-smi >nul 2>&1
if not errorlevel 1 (
  echo   GPU detected. Installing PyTorch with CUDA 12.6 support...
  "%VENV_PY%" -m pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126
  if errorlevel 1 (
    echo   [Warning] CUDA PyTorch install failed. Keeping CPU version.
  ) else (
    echo   OK  PyTorch CUDA installed.
  )
) else (
  echo   No GPU detected. Using CPU PyTorch.
)

echo   OK  Requirements installed.
echo.

REM -----------------------------
REM 3) Runtime import check
REM -----------------------------
echo [3/7] Verifying Qwen runtime packages...
"%VENV_PY%" -c "import transformers, torch, huggingface_hub, yt_dlp, openai; print('import check ok')" >nul 2>&1
if errorlevel 1 (
  echo   [Warning] Some runtime packages failed import.
  echo             Please run: .venv\Scripts\python -m pip install -r requirements.txt
) else (
  echo   OK  Runtime imports verified.
)

"%VENV_PY%" -c "from qwen_tts import Qwen3TTSModel; print('qwen_tts ok')" >nul 2>&1
if errorlevel 1 (
  echo   [Warning] qwen_tts package not importable.
  echo             Please run: .venv\Scripts\python -m pip install -r requirements-qwen.txt
) else (
  echo   OK  qwen_tts import verified.
)
echo.

REM -----------------------------
REM 4) Optional Ollama setup
REM -----------------------------
echo [4/7] Optional: Install/check Ollama for local translation model
set "INSTALL_OLLAMA="
set /p INSTALL_OLLAMA=  Install/check Ollama now? [Y/N] Enter=N:
if "%INSTALL_OLLAMA%"=="" set "INSTALL_OLLAMA=N"
if /i not "%INSTALL_OLLAMA%"=="Y" goto :skip_ollama

ollama --version >nul 2>&1
if not errorlevel 1 (
  echo   OK  Ollama already installed.
  goto :ollama_pull
)

echo   Downloading Ollama installer...
set "OLLAMA_SETUP=%TEMP%\OllamaSetup.exe"
curl -L --progress-bar -o "%OLLAMA_SETUP%" https://ollama.com/download/OllamaSetup.exe
if errorlevel 1 (
  echo   [Warning] Failed to download Ollama installer.
  goto :skip_ollama
)

start /wait "Ollama Setup" "%OLLAMA_SETUP%"
del "%OLLAMA_SETUP%" >nul 2>&1

ollama --version >nul 2>&1
if errorlevel 1 (
  echo   [Warning] Ollama install not detected after setup.
  goto :skip_ollama
)
echo   OK  Ollama installed.

:ollama_pull
echo.
echo   Optional: pull local translation model
set "PULL_MODEL="
set /p PULL_MODEL=  Pull qwen2.5:3b now? [Y/N] Enter=N:
if "%PULL_MODEL%"=="" set "PULL_MODEL=N"
if /i "%PULL_MODEL%"=="Y" (
  ollama pull qwen2.5:3b
)

:skip_ollama
echo.

REM -----------------------------
REM 5) Download Qwen models
REM -----------------------------
echo [5/7] Model pre-download
set "PREDL="
set /p PREDL=  Download ALL Qwen3 ASR/TTS models now? [Y/N] Enter=Y:
if "%PREDL%"=="" set "PREDL=Y"

if /i "%PREDL%"=="Y" (
  "%VENV_PY%" -c "from app.core.model_downloader import download_all_models; download_all_models(status_callback=lambda m: print(m, flush=True))"
  if errorlevel 1 (
    echo   [Warning] Model pre-download failed. You can retry later in app settings.
  ) else (
    echo   OK  Model pre-download completed.
  )
) else (
  echo   Skip model pre-download.
)
echo.

REM -----------------------------
REM 6) Optional cleanup non-Qwen model cache
REM -----------------------------
echo [6/7] Optional: Cleanup non-Qwen HuggingFace model cache
set "CLEAN_NON_QWEN="
set /p CLEAN_NON_QWEN=  Cleanup non-Qwen ASR/TTS model cache now? [Y/N] Enter=N:
if "%CLEAN_NON_QWEN%"=="" set "CLEAN_NON_QWEN=N"
if /i not "%CLEAN_NON_QWEN%"=="Y" goto :skip_cleanup

"%VENV_PY%" -c "from app.core.model_cache_cleanup import cleanup_non_qwen_model_cache; r=cleanup_non_qwen_model_cache(); print(f'removed={len(r.removed_dirs)} kept_qwen={len(r.kept_qwen_dirs)} failed={len(r.failed_dirs)} missing_root={r.missing_cache_root}'); import sys; sys.exit(0 if not r.failed_dirs else 1)"
if errorlevel 1 (
  echo   [Warning] Cache cleanup finished with failures. Please inspect manually.
) else (
  echo   OK  Cache cleanup completed.
)

:skip_cleanup
echo.

REM -----------------------------
REM 7) Quick startup check
REM -----------------------------
echo [7/7] Startup smoke check...
"%VENV_PY%" -c "import main; print('startup import ok')" >nul 2>&1
if errorlevel 1 (
  echo   [Warning] Startup import failed. Please run start.cmd and inspect traceback.
) else (
  echo   OK  App import check passed.
)

echo.
echo ==================================================
echo  Install finished.
echo ==================================================
echo.

set "START_NOW="
set /p START_NOW=  Start app now? [Y/N] Enter=Y:
if "%START_NOW%"=="" set "START_NOW=Y"
if /i "%START_NOW%"=="Y" (
  "%VENV_PY%" main.py
)
goto :eof

:no_python
echo [Error] Python 3.10-3.13 not found via py launcher.
echo         Install Python 3.10-3.13 and ensure "py" command is available.
pause
exit /b 1

:py_old
echo [Error] Python %PY_MAJOR%.%PY_MINOR% is too old. Require 3.10+.
pause
exit /b 1

:py_new
echo [Error] Python %PY_MAJOR%.%PY_MINOR% is not yet supported. Require 3.10-3.13.
pause
exit /b 1

:venv_failed
echo [Error] Failed to create virtual environment.
pause
exit /b 1

:venv_pip_failed
echo [Error] venv created but pip is unavailable.
echo         Please check Python installation integrity, then rerun install.cmd.
pause
exit /b 1

:pip_failed
echo [Error] Failed to install dependencies.
echo         Please check network/proxy and run again.
pause
exit /b 1
