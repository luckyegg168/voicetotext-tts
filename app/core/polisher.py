"""OpenAI / OpenRouter GPT 文字潤稿"""
from openai import OpenAI

TEMPLATES: dict[str, str] = {
    "general": (
        "你是文字潤稿助手。將以下口語錄音轉寫文字整理成通順的書面文字，"
        "修正標點符號，移除重複詞語，但保留原意。直接輸出整理後文字，不要加任何說明。"
    ),
    "social": (
        "你是社群媒體內容創作助手。將以下文字改寫成適合臉書/Instagram 的貼文，"
        "加入適當換行，語氣自然親切。直接輸出結果，不要加任何說明。"
    ),
    "meeting": (
        "你是會議記錄助手。將以下口語內容整理成條列式會議記錄，"
        "包含重要決議、行動項目。直接輸出結果，不要加任何說明。"
    ),
    "email": (
        "你是專業書信助手。將以下內容改寫成正式電子郵件格式，"
        "語氣專業有禮。直接輸出結果，不要加任何說明。"
    ),
    "video": (
        "你是專業影片腳本編輯。將以下口語內容改寫成適合 YouTube 或短影音的腳本，"
        "段落清晰，開場吸引人，語氣生動有個人風格，加入適當的停頓提示與轉場語。"
        "直接輸出腳本，不要加任何說明。"
    ),
    "code": (
        "你是資深程式設計師。將以下口語描述的程式邏輯或需求，整理成清晰的技術規格說明，"
        "條列重點步驟、函式／變數命名建議、邊界條件與注意事項。"
        "使用繁體中文說明，保留英文技術術語與程式關鍵字。"
        "直接輸出結果，不要加任何說明。"
    ),
}

OUTPUT_LANG_INSTRUCTIONS: dict[str, str] = {
    "original": "",
    "zh": "請用繁體中文輸出。",
    "en": "Please output in English.",
}

TEMPLATE_LABELS = {
    "general": "通用",
    "social":  "社群貼文",
    "meeting": "會議記錄",
    "email":   "信件",
    "video":   "拍片腳本",
    "code":    "程式編寫",
}


TRANSLATE_TARGETS = {
    "英文": "Please translate the following text into natural English. Output only the translation.",
    "日文": "以下のテキストを自然な日本語に翻訳してください。翻訳文のみ出力してください。",
    "韓文": "다음 텍스트를 자연스러운 한국어로 번역해 주세요. 번역문만 출력하세요.",
    "西班牙文": "Traduce el siguiente texto al español natural. Solo muestra la traducción.",
    "法文": "Traduisez le texte suivant en français naturel. Ne retournez que la traduction.",
    "德文": "Übersetzen Sie den folgenden Text ins natürliche Deutsch. Geben Sie nur die Übersetzung aus.",
    "繁體中文": "請將以下文字翻譯成繁體中文。只輸出翻譯結果，不要加任何說明。",
    "簡體中文": "请将以下文字翻译成简体中文。只输出翻译结果，不要加任何说明。",
}


def translate(
    text: str,
    target_lang: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
) -> str:
    """翻譯文字到指定語言"""
    if not text.strip():
        raise ValueError("輸入文字為空")

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
    )
    return response.choices[0].message.content.strip()


def polish(
    text: str,
    api_key: str,
    template: str = "general",
    output_language: str = "original",
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
) -> str:
    """
    使用 GPT/OpenRouter 潤稿文字。
    base_url=None → OpenAI；base_url="https://openrouter.ai/api/v1" → OpenRouter
    """
    if not text.strip():
        raise ValueError("輸入文字為空")

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
    )
    return response.choices[0].message.content.strip()
