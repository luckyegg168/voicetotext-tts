"""Ollama availability/model helpers."""
import json
import subprocess
import urllib.request
from collections.abc import Callable
from typing import Any


def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check whether Ollama API is reachable."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """List installed Ollama model names."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as response:
            payload = json.loads(response.read())
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    models = payload.get("models", [])
    if not isinstance(models, list):
        return []

    names: list[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def is_ollama_model_available(model_name: str, base_url: str = "http://localhost:11434") -> bool:
    """Check if model exists locally (supports tag prefix match)."""
    if not model_name:
        return False
    normalized = model_name.strip()
    if not normalized:
        return False

    installed = list_ollama_models(base_url)
    normalized_prefix = normalized.split(":")[0]
    for installed_name in installed:
        if installed_name == normalized or installed_name.startswith(normalized_prefix):
            return True
    return False


def _safe_status_callback(status_callback: Callable[[str], None] | None, message: str) -> None:
    if not status_callback:
        return
    try:
        status_callback(message)
    except Exception:
        pass


def _to_readable_process_error(result: subprocess.CompletedProcess[str]) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    if stderr:
        return stderr
    if stdout:
        return stdout
    return f"ollama pull exited with code {result.returncode}."


def _normalize_timeout_seconds(timeout_seconds: Any) -> int:
    if isinstance(timeout_seconds, bool):
        return 600
    if isinstance(timeout_seconds, (int, float)):
        timeout = int(timeout_seconds)
        return timeout if timeout > 0 else 600
    return 600


def pull_ollama_model(
    model_name: str,
    status_callback: Callable[[str], None] | None = None,
    timeout_seconds: int = 600,
) -> None:
    """
    Pull an Ollama model with timeout and readable errors.

    status_callback(msg: str) can be used by UI to show progress.
    """
    normalized_model = model_name.strip() if isinstance(model_name, str) else ""
    if not normalized_model:
        raise ValueError("Model name is required for ollama pull.")

    timeout = _normalize_timeout_seconds(timeout_seconds)
    _safe_status_callback(status_callback, f"Pulling Ollama model: {normalized_model} ...")

    try:
        result = subprocess.run(
            ["ollama", "pull", normalized_model],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Cannot find `ollama` command. Please install Ollama and check PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        if details:
            details = f" Details: {details}"
        raise RuntimeError(
            f"Ollama pull timed out after {timeout} seconds for model '{normalized_model}'.{details}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to execute ollama pull for '{normalized_model}': {exc}") from exc

    if result.returncode != 0:
        readable_error = _to_readable_process_error(result)
        raise RuntimeError(f"Failed to pull Ollama model '{normalized_model}': {readable_error}")

    _safe_status_callback(status_callback, f"Model pull completed: {normalized_model}")
