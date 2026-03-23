---
name: qwen3-asr
description: |
  Complete reference for Qwen3-ASR speech recognition models (Qwen/Qwen3-ASR-0.6B, Qwen/Qwen3-ASR-1.7B)
  and Qwen3-ForcedAligner-0.6B timestamp alignment. Use this skill whenever working with:
  - Qwen3-ASR installation, setup, or dependency issues (especially the qwen-asr package)
  - ASR transcription code using qwen_asr.Qwen3ASRModel
  - Forced alignment / timestamp extraction with Qwen3ForcedAligner
  - Troubleshooting "model type qwen3_asr not recognized" or transformers version conflicts
  - Choosing between 0.6B vs 1.7B model, transformers vs vLLM backend
  - Language/dialect support questions (52 languages, 22 Chinese dialects)
  - Integrating Qwen3-ASR into apps (batch inference, streaming, API serving)
  - This project's app/core/qwen3_asr.py implementation
---

# Qwen3-ASR Skill

## Critical: Installation

**Always install `qwen-asr` from PyPI** — the `qwen3_asr` architecture is NOT in the main
`transformers` repo (PR #43838 is still WIP as of March 2026).

```bash
pip install -U qwen-asr           # transformers backend
pip install -U qwen-asr[vllm]    # + vLLM for streaming/production
```

`qwen-asr` installs `transformers==4.57.6` and provides the `qwen3_asr` architecture internally.

**Known conflict:** `qwen-tts 0.1.1` pins `transformers==4.57.3`. The 4.57.3 vs 4.57.6
difference is minor and both work in practice — ignore the pip resolver warning.

**DO NOT:**
- `pip install git+https://github.com/huggingface/transformers.git` (doesn't have qwen3_asr)
- Use `transformers.pipeline()` for Qwen3-ASR (will fail — no auto_map in config)

---

## Model Selection

| Model | Params | Use Case |
|-------|--------|----------|
| `Qwen/Qwen3-ASR-0.6B` | 0.6B | Edge / high-concurrency (2000x throughput @ batch 128) |
| `Qwen/Qwen3-ASR-1.7B` | 1.7B | Best accuracy, production quality |
| `Qwen/Qwen3-ForcedAligner-0.6B` | 0.6B | Timestamp/alignment only (requires ASR result) |

---

## Basic Usage

### Transcription (transformers backend)

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",       # or 1.7B
    dtype=torch.float16,           # bfloat16 recommended if GPU supports it
    device_map="cuda:0",           # or "cpu"
    # attn_implementation="flash_attention_2",  # optional speedup
    max_inference_batch_size=32,   # lower to avoid OOM
    max_new_tokens=256,            # increase for long audio
)

# Single file
results = model.transcribe(
    audio="path/to/audio.wav",    # path, URL, base64, or (np.ndarray, sample_rate)
    language=None,                 # None = auto-detect; or "Chinese", "English", etc.
)
print(results[0].language)
print(results[0].text)
```

### Batch inference

```python
results = model.transcribe(
    audio=["audio1.wav", "audio2.wav"],
    language=["Chinese", "English"],   # list must match audio list
)
for r in results:
    print(f"{r.language}: {r.text}")
```

### NumPy array input (used in this project)

```python
import numpy as np

audio_np = np.zeros(16000, dtype=np.float32)   # float32, shape (N,)
results = model.transcribe(
    audio=(audio_np, 16000),    # (ndarray, sample_rate)
    language=None,
)
```

---

## Timestamps with ForcedAligner

### Embed aligner in ASR model

```python
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_new_tokens=256,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ),
)

results = model.transcribe(
    audio="audio.wav",
    language="Chinese",
    return_time_stamps=True,
)
for r in results:
    print(r.text)
    print(r.time_stamps[0])    # first token's timestamps
```

### Standalone aligner

```python
from qwen_asr import Qwen3ForcedAligner

aligner = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

results = aligner.align(
    audio="audio.wav",
    text="甚至出现交易几乎停滞的情况。",
    language="Chinese",
)
# results[0] = list of token-level objects
for token in results[0]:
    print(token.text, token.start_time, token.end_time)
```

**ForcedAligner limits:** max 5 min audio; only 11 languages (Chinese, English, Cantonese,
French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish).

---

## API Parameters Reference

### `Qwen3ASRModel.from_pretrained()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` (1st arg) | str | `"Qwen/Qwen3-ASR-0.6B"` or `"Qwen/Qwen3-ASR-1.7B"` |
| `dtype` | torch.dtype | `torch.float16` / `torch.bfloat16` / `torch.float32` |
| `device_map` | str | `"cuda:0"`, `"cuda:1"`, `"cpu"` |
| `attn_implementation` | str | `"flash_attention_2"` (optional speedup, needs flash-attn) |
| `max_inference_batch_size` | int | Batch limit; -1 = unlimited; lower = less OOM risk |
| `max_new_tokens` | int | 256 default; increase for long audio (up to 4096) |
| `forced_aligner` | str | `"Qwen/Qwen3-ForcedAligner-0.6B"` to enable timestamps |
| `forced_aligner_kwargs` | dict | Init args for the aligner (dtype, device_map, etc.) |

### `model.transcribe()`

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | str / list / tuple | Path, URL, base64, `(ndarray, sr)`, or list of these |
| `language` | str / list / None | Language name(s) or `None` for auto-detect |
| `return_time_stamps` | bool | `True` requires `forced_aligner` to be set at load time |

### Result object fields

| Field | Type | Description |
|-------|------|-------------|
| `result.text` | str | Transcription text |
| `result.language` | str | Detected/specified language |
| `result.time_stamps` | list | Token-level timestamps (only if `return_time_stamps=True`) |

---

## vLLM Backend (Production / Streaming)

```python
# Must be inside if __name__ == '__main__':
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.LLM(
    model="Qwen/Qwen3-ASR-1.7B",
    gpu_memory_utilization=0.7,    # 0.7–0.9 typical
    max_inference_batch_size=128,
    max_new_tokens=4096,
)

results = model.transcribe(audio=["a.wav", "b.wav"], language=["Chinese", "English"])
```

### vLLM serve (OpenAI-compatible API)

```bash
vllm serve Qwen/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000
# Multi-GPU:
vllm serve Qwen/Qwen3-ASR-1.7B --tensor-parallel-size 2
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.audio.transcriptions.create(
    model="Qwen/Qwen3-ASR-1.7B",
    file=open("audio.wav", "rb"),
)
print(response.text)
```

---

## Supported Languages

**30 languages:** Chinese (zh), English (en), Cantonese (yue), Arabic (ar), German (de),
French (fr), Spanish (es), Portuguese (pt), Indonesian (id), Italian (it), Korean (ko),
Russian (ru), Thai (th), Vietnamese (vi), Japanese (ja), Turkish (tr), Hindi (hi),
Malay (ms), Dutch (nl), Swedish (sv), Danish (da), Finnish (fi), Polish (pl), Czech (cs),
Filipino (fil), Persian (fa), Greek (el), Hungarian (hu), Macedonian (mk), Romanian (ro)

**22 Chinese dialects:** Anhui, Dongbei, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei,
Hunan, Jiangxi, Ningxia, Shandong, Shaanxi, Shanxi, Sichuan, Tianjin, Yunnan, Zhejiang,
Cantonese (Hong Kong), Cantonese (Guangdong), Wu, Minnan

---

## This Project: app/core/qwen3_asr.py

The project's `qwen3_asr.py` wraps `Qwen3ASRModel`:

- `_get_model(model_id, device)` — loads and caches the model (limit=1 in-memory)
- `transcribe(audio_input, model_id, language, device, return_segments)` — main entry point
- Audio input: `bytes` (WAV) or `np.ndarray` (float32, mono) at 16000 Hz
- Cache eviction: LRU, evicts previous model + calls `gc.collect()` + `cuda.empty_cache()`

```python
from app.core.qwen3_asr import transcribe

text, chunks = transcribe(
    audio_input=wav_bytes,          # bytes or np.ndarray
    model_id="Qwen/Qwen3-ASR-0.6B",
    language="auto",                # or "Chinese", "English", etc.
    device="cuda",                  # or "cpu"
    return_segments=False,
)
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `model type qwen3_asr not recognized` | Using raw transformers without qwen-asr | `pip install qwen-asr` |
| `qwen-tts requires transformers==4.57.3` | Version conflict with qwen-asr | Ignore — 4.57.6 is compatible |
| OOM during load | Too large batch / dtype | Lower `max_inference_batch_size`; use `float16` |
| Slow inference | No FlashAttention | `pip install flash-attn --no-build-isolation` |
| Streaming not working | Transformers backend | Use vLLM backend for streaming |
| Timestamps empty | No forced aligner | Pass `forced_aligner=` at model load time |

---

## Performance Benchmarks (WER ↓ = better)

| Dataset | Qwen3-ASR-1.7B | Whisper-large-v3 |
|---------|---------------|-----------------|
| LibriSpeech (clean) | 1.63 | 1.51 |
| LibriSpeech (other) | 3.38 | 3.97 |
| GigaSpeech | **8.45** | 9.76 |
| WenetSpeech (net) | **4.97** | 9.86 |
| WenetSpeech (meeting) | **5.88** | 19.11 |
| Fleurs (multilingual) | **4.90** | 5.27 |

Forced aligner accuracy: **42.9ms AAS** vs WhisperX 133.2ms.

---

## Resources

- GitHub: https://github.com/QwenLM/Qwen3-ASR
- Paper: arXiv:2601.21337
- HuggingFace: https://huggingface.co/collections/Qwen/qwen3-asr
- License: Apache 2.0
