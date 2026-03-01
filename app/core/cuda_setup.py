"""
CUDA DLL 預載入 — 必須在 main.py 最開頭呼叫，早於所有其他 import。

ctranslate2（faster-whisper 的底層）在 Windows 用自己的 LoadLibrary 機制，
光靠 PATH 或 os.add_dll_directory() 不可靠。
解法：用 ctypes 先把 DLL 強制載入到 process，ctranslate2 之後就能直接用。
"""
import os
import sys
import ctypes
import site


def setup() -> None:
    if sys.platform != "win32":
        return

    dll_dirs = _find_nvidia_bin_dirs()
    if not dll_dirs:
        return

    for dll_dir in dll_dirs:
        # 1. os.add_dll_directory（現代 DLL 搜尋）
        try:
            os.add_dll_directory(dll_dir)
        except OSError:
            pass

        # 2. PATH（舊式 LoadLibrary 搜尋）
        if dll_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

        # 3. ctypes 強制預載入（最可靠，讓 DLL 進入 process 快取）
        for fname in os.listdir(dll_dir):
            if fname.endswith(".dll"):
                try:
                    ctypes.WinDLL(os.path.join(dll_dir, fname))
                except OSError:
                    pass


def _find_nvidia_bin_dirs() -> list[str]:
    """回傳所有 nvidia/*/bin 目錄，含系統和使用者 site-packages。"""
    search_roots: list[str] = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site and user_site not in search_roots:
        search_roots.append(user_site)

    found: list[str] = []
    for root in search_roots:
        nvidia_root = os.path.join(root, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for pkg in os.listdir(nvidia_root):
            bin_dir = os.path.join(nvidia_root, pkg, "bin")
            if os.path.isdir(bin_dir):
                found.append(bin_dir)
    return found
