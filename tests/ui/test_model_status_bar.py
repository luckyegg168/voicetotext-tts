# tests/ui/test_model_status_bar.py
"""Verify update_model_status() method contract."""
from unittest.mock import MagicMock


def test_update_model_status_configures_label():
    """update_model_status() must call configure() on the label with correct args."""
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
