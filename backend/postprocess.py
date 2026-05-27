import re
import time

from backend.config import GEMINI_API_KEY, logger
from backend.metrics import gemini_calls

FILLER_WORDS_RE = re.compile(
    r"\b(эээ|ээ|эм+|ммм+|хм+|ну вот|вот так вот|как бы(?! то ни было)|типа того|короче говоря)\b",
    re.IGNORECASE,
)
REPEATED_WORDS_RE = re.compile(r"\b(\w+)(?:\s+\1\b){2,}", re.IGNORECASE)
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


GEMINI_MAX_RETRIES = 3
GEMINI_BACKOFF = [2, 5, 10]  # seconds


def gemini_polish(text: str) -> str:
    """Полировка текста через Gemini API с retry на rate-limit/5xx."""
    global _gemini_model
    if _gemini_model is None:
        _gemini_model = _get_gemini_model()
    if _gemini_model is None:
        return text

    prompt = (
        "Ты — корректор расшифровок телеинтервью. Правила:\n"
        "1. Убери ТОЛЬКО слова-паразиты: эээ, ээ, ну, вот, ну вот, как бы, типа, короче\n"
        "2. Исправь ТОЛЬКО очевидные ошибки пунктуации (пропущенные точки, запятые)\n"
        "3. НЕ меняй порядок слов, НЕ переформулируй, НЕ добавляй слова\n"
        "4. НЕ трогай имена собственные, числа, даты, аббревиатуры\n"
        "5. Сохрани разговорный стиль речи\n"
        "6. Верни ТОЛЬКО текст, без комментариев\n\n"
        "Пример:\n"
        "Вход: Ну вот эээ мы значит пришли и типа начали работать\n"
        "Выход: Мы пришли и начали работать.\n\n"
        f"Текст:\n{text}"
    )

    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            response = _gemini_model.generate_content(prompt)
            result = response.text.strip()
            gemini_calls.labels(outcome="success").inc()
            if result:
                return result
            return text
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(s in err_str for s in ("rate limit", "429", "503", "500", "timeout", "deadline"))
            if attempt < GEMINI_MAX_RETRIES - 1 and is_retryable:
                delay = GEMINI_BACKOFF[attempt]
                logger.warning("Gemini API ошибка (попытка %d/%d): %s. Retry через %dс",
                               attempt + 1, GEMINI_MAX_RETRIES, e, delay)
                time.sleep(delay)
                continue
            gemini_calls.labels(outcome="error").inc()
            logger.warning("Gemini API окончательная ошибка: %s", e)
            break

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
