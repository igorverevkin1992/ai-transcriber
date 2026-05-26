import os
import re

from backend.config import GEMINI_API_KEY, logger

FILLER_WORDS_RE = re.compile(
    r"\b(эээ|ээ|эм+|ммм+|хм+|ну вот|вот так вот|как бы|типа того|короче говоря)\b",
    re.IGNORECASE,
)
REPEATED_WORDS_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)
MULTI_SPACE_RE = re.compile(r" {2,}")
CAPITALIZE_RE = re.compile(r"([.!?])\s+([а-яa-z])")


def regex_cleanup(text: str) -> str:
    """Базовая чистка текста: filler-слова, повторы, капитализация."""
    text = FILLER_WORDS_RE.sub("", text)
    text = REPEATED_WORDS_RE.sub(r"\1", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = CAPITALIZE_RE.sub(lambda m: m.group(1) + " " + m.group(2).upper(), text)
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _get_gemini_model():
    """Возвращает Gemini-модель для полировки текста."""
    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("google-generativeai не установлен. Gemini-полировка недоступна.")
        return None

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY не задан. Gemini-полировка недоступна.")
        return None

    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.0-flash")


_gemini_model = None


def gemini_polish(text: str) -> str:
    """Полировка текста через Gemini API."""
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = _get_gemini_model()
    if _gemini_model is None:
        return text

    prompt = (
        "Ты — редактор расшифровок телеинтервью. Отредактируй текст расшифровки:\n"
        "- Убери слова-паразиты (эээ, ну, вот, как бы, типа)\n"
        "- Исправь пунктуацию и грамматику\n"
        "- Сохрани смысл дословно, не меняй содержание\n"
        "- Не добавляй ничего от себя\n"
        "- Верни ТОЛЬКО отредактированный текст, без пояснений\n\n"
        f"Текст:\n{text}"
    )

    try:
        response = _gemini_model.generate_content(prompt)
        result = response.text.strip()
        if result:
            return result
    except Exception as e:
        logger.warning("Gemini API ошибка: %s", e)

    return text


def postprocess_segments(segments: list[dict], use_gemini: bool = True) -> list[dict]:
    """Постобработка сегментов: regex-чистка + опционально Gemini-полировка.

    Батчит сегменты для Gemini (~5000 символов за раз) для экономии API-вызовов.
    """
    if not segments:
        return segments

    for seg in segments:
        seg["text"] = regex_cleanup(seg["text"])

    if not use_gemini or not GEMINI_API_KEY:
        return segments

    batch_texts = []
    batch_indices = []
    current_batch = ""
    current_indices = []
    SEPARATOR = "\n---SEGMENT_BREAK---\n"
    MAX_BATCH_CHARS = 5000

    for i, seg in enumerate(segments):
        text = seg["text"]
        if not text:
            continue
        if current_batch and len(current_batch) + len(SEPARATOR) + len(text) > MAX_BATCH_CHARS:
            batch_texts.append(current_batch)
            batch_indices.append(current_indices)
            current_batch = text
            current_indices = [i]
        else:
            if current_batch:
                current_batch += SEPARATOR + text
            else:
                current_batch = text
            current_indices.append(i)

    if current_batch:
        batch_texts.append(current_batch)
        batch_indices.append(current_indices)

    for batch_text, indices in zip(batch_texts, batch_indices):
        polished = gemini_polish(batch_text)
        parts = polished.split("---SEGMENT_BREAK---")
        parts = [p.strip() for p in parts]

        if len(parts) == len(indices):
            for idx, polished_text in zip(indices, parts):
                if polished_text:
                    segments[idx]["text"] = polished_text
        else:
            # batch didn't split cleanly — polish individually
            for idx in indices:
                segments[idx]["text"] = gemini_polish(segments[idx]["text"])

    return segments
