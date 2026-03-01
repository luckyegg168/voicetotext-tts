"""OpenAI/OpenRouter text polishing and translation helpers."""

from __future__ import annotations

import os

from openai import OpenAI

CHAT_REQUEST_TIMEOUT_SECONDS = float(os.getenv("OPENAI_CHAT_TIMEOUT_SECONDS", "90"))

TEMPLATES: dict[str, str] = {
    "general": (
        "你是中文文字潤稿助理。請將語音轉寫文字整理成可直接使用的自然書面語，"
        "保留原意，不杜撰內容，修正標點與語句流暢度。"
    ),
    "social": (
        "你是社群文案助理。請將輸入整理成適合社群平台發佈的自然語句，"
        "保留原意並加強可讀性。"
    ),
    "meeting": (
        "你是會議紀錄助理。請將輸入整理成清楚、有條理的會議紀錄或重點摘要，"
        "保留事實與關鍵結論。"
    ),
    "email": (
        "你是商務郵件助理。請將輸入整理成禮貌、清楚且可直接寄送的郵件內容，"
        "保留原意並修正語氣與格式。"
    ),
    "video": (
        "你是影音腳本文案助理。請將輸入整理成口語自然、節奏清楚的影片旁白或腳本，"
        "保持重點明確。"
    ),
    "code": (
        "你是技術寫作助理。請將輸入整理成清楚、準確、可執行的技術說明，"
        "必要時保留原始術語。"
    ),
}

OUTPUT_LANG_INSTRUCTIONS: dict[str, str] = {
    "original": "",
    "zh": "請以繁體中文輸出。",
    "en": "Please output in English.",
}

TEMPLATE_LABELS = {
    "general": "通用",
    "social": "社群貼文",
    "meeting": "會議紀錄",
    "email": "商務郵件",
    "video": "影音腳本",
    "code": "技術說明",
}

TRANSLATE_TARGETS = {
    "英文": "Please translate the following text into natural English. Output only the translation.",
    "日文": "次の文章を自然な日本語に翻訳してください。翻訳結果のみを返してください。",
    "韓文": "다음 문장을 자연스러운 한국어로 번역하세요. 번역 결과만 출력하세요.",
    "西班牙文": "Traduce el siguiente texto al español natural. Devuelve solo la traducción.",
    "法文": "Traduisez le texte suivant en français naturel. Retournez uniquement la traduction.",
    "德文": "Übersetzen Sie den folgenden Text ins natürliche Deutsch. Geben Sie nur die Übersetzung aus.",
    "繁體中文": "請將下列文字翻譯成自然的繁體中文，只輸出翻譯結果。",
    "简体中文": "请将下列文字翻译成自然的简体中文，只输出翻译结果。",
}


def translate(
    text: str,
    target_lang: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
) -> str:
    """Translate text to target language with chat completion API."""
    if not text.strip():
        raise ValueError("沒有可翻譯的文字")

    system_prompt = TRANSLATE_TARGETS.get(target_lang, TRANSLATE_TARGETS["英文"])

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        timeout=CHAT_REQUEST_TIMEOUT_SECONDS,
    )
    return (response.choices[0].message.content or "").strip()


def polish(
    text: str,
    api_key: str,
    template: str = "general",
    output_language: str = "original",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
) -> str:
    """Polish text via chat completion API."""
    if not text.strip():
        raise ValueError("沒有可潤稿的文字")

    system_prompt = TEMPLATES.get(template, TEMPLATES["general"])
    lang_instruction = OUTPUT_LANG_INSTRUCTIONS.get(output_language, "")
    if lang_instruction:
        system_prompt = system_prompt + "\n" + lang_instruction

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        timeout=CHAT_REQUEST_TIMEOUT_SECONDS,
    )
    return (response.choices[0].message.content or "").strip()
