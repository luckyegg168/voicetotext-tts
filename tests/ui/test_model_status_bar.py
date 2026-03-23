# tests/ui/test_model_status_bar.py
"""Verify HomePage.update_model_status() calls configure() correctly."""
from unittest.mock import MagicMock

from app.ui.main_window import HomePage


def test_update_model_status_configures_label():
    """update_model_status() must call configure() on the label with correct args."""
    label_mock = MagicMock()
    page = object.__new__(HomePage)  # skip __init__ (no tkinter root needed)
    page._model_status_label = label_mock

    page.update_model_status("✅ 模型就緒", "#2ecc71")

    label_mock.configure.assert_called_once_with(text="✅ 模型就緒", text_color="#2ecc71")


def test_update_model_status_clears_on_empty_string():
    """Passing empty string resets the label text."""
    label_mock = MagicMock()
    page = object.__new__(HomePage)
    page._model_status_label = label_mock

    page.update_model_status("", "gray")

    label_mock.configure.assert_called_once_with(text="", text_color="gray")
