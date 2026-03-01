"""Audio recorder and recording state definitions."""
import io
import threading
from enum import Enum

import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd

SAMPLE_RATE = 16000


class RecordingState(Enum):
    IDLE = "待機中"
    RECORDING = "錄音中"
    TRANSCRIBING = "轉寫中"
    POLISHING = "潤稿中"
    DONE = "完成"
    ERROR = "錯誤"


class RecorderStartError(RuntimeError):
    """Raised when microphone stream creation/start fails."""


class AudioRecorder:
    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

    def start(self) -> None:
        with self._lock:
            if self._recording:
                raise RecorderStartError("Recorder is already recording.")
            self._frames = []
            self._recording = True

        def _callback(indata, frames, time_info, status):
            del frames, time_info, status
            with self._lock:
                if self._recording:
                    self._frames.append(indata.copy())

        stream: sd.InputStream | None = None
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                callback=_callback,
            )
            stream.start()
        except Exception as exc:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            with self._lock:
                self._recording = False
                self._stream = None
                self._frames = []
            raise RecorderStartError(
                "Unable to start recording. Check microphone availability and permissions."
            ) from exc

        with self._lock:
            self._stream = stream

    def stop(self) -> bytes:
        """Stop recording and return WAV bytes."""
        with self._lock:
            self._recording = False
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

        with self._lock:
            frames = self._frames
            self._frames = []

        if not frames:
            return b""

        audio = np.concatenate(frames, axis=0)
        audio_int16 = (audio * 32767).astype(np.int16)

        buf = io.BytesIO()
        wav.write(buf, SAMPLE_RATE, audio_int16)
        return buf.getvalue()

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording
