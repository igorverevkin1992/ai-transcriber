import re
import threading
import time

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, logger
from backend.metrics import gemini_calls

# «что -то» → «что-то»: пробел ПЕРЕД дефисом, но не после.
# Пробелы с обеих сторон — это тире, его обрабатывает DASH_RE ниже.
HYPHEN_SPACE_RE = re.compile(r"(\w) +-(\w)")
# Эталонные стенограммы почти дословные: «ну», «вот», «как бы», «типа»,
# «короче» в них СОХРАНЯЮТСЯ. Удаляем только нечленораздельные звуки,
# которые человек никогда не переносит в текст. Дефисные варианты («э-э»,
# «м-м-м») — частая запись мычания у Whisper. «Хм» НЕ удаляем: эталон ф5
# сохраняет «Хм…» как осмысленную реплику.
FILLER_WORDS_RE = re.compile(r"\b(э(?:-э)+|м(?:-м)+|а(?:-а)+|э{2,}|эм+|ммм+)\b", re.IGNORECASE)
REPEATED_WORDS_RE = re.compile(r"\b([а-яА-ЯёЁa-zA-Z]+)(?:\s+\1\b){2,}", re.IGNORECASE)
# Запятая, за которой идёт другой знак, — артефакт удаления филлера
# («ну, эээ, давай» → «ну,, давай»). В эталонах ",," и ",." — 0 случаев.
DUP_PUNCT_RE = re.compile(r",\s*([,.;:!?])")
# Em/en-dash без пробелов между БУКВАМИ → " – ". Между цифрами не трогаем:
# диапазоны в эталонах пишутся дефисом без пробелов («49-50», «2002-2003»).
SPACELESS_DASH_RE = re.compile(r"([а-яА-ЯёЁa-zA-Z])[—–]([а-яА-ЯёЁa-zA-Z])")
# Цифровой диапазон через en/em-тире (с пробелами или без) → дефис
DIGIT_RANGE_RE = re.compile(r"(\d) ?[—–] ?(\d)")
# Дефис(ы) или en/em-тире с пробелами вокруг → " – " (en-dash, 146:0 в эталонах)
DASH_RE = re.compile(r"(?<=\S)[ \t]+(?:-+|[—–])[ \t]+(?=\S)")

# Эталоны НИКОГДА не используют «т.е./т.д./т.п.» — всегда полные слова
# (цензус: 0 случаев на 4 файла). Точку не сохраняем: пунктуацию по смыслу
# восстанавливает Gemini-полировка.
_ABBR_EXPANSIONS = [
    (re.compile(r"\b[тТ]\.\s?е\."), "то есть"),
    (re.compile(r"\b[тТ]\.\s?д\."), "так далее"),
    (re.compile(r"\b[тТ]\.\s?п\."), "тому подобное"),
]


def _expand_abbreviations(text: str) -> str:
    for rx, repl in _ABBR_EXPANSIONS:
        text = rx.sub(lambda m, r=repl: r.capitalize() if m.group(0)[0] == "Т" else r, text)
    return text
# Эталоны используют "..." (451 случай), а не "…" (74, один файл)
ELLIPSIS_CHAR_RE = re.compile(r"…")
MANY_DOTS_RE = re.compile(r"\.{4,}")
# ASCII-кавычки → «ёлочки» (в эталонах только «»)
GUILLEMET_RE = re.compile(r'"([^"\n]+)"')
SPACE_BEFORE_PUNCT_RE = re.compile(r" +([,.!?;:])")
LEADING_PUNCT_RE = re.compile(r"^[\s,;:]+")
MULTI_SPACE_RE = re.compile(r" {2,}")
# Капитализация после конца предложения; (?<!\.) исключает "...", после
# которого эталоны продолжают со строчной («Мне это было... это была...»)
_CAPITALIZE_BASE_RE = re.compile(r"(?<!\.)([.!?])\s+([а-яa-z])")

# Сокращения, после точки которых НЕ начинается новое предложение.
# Эталоны всегда капитализируют после «да.», «нет.», «так.», «вот.»
# (89 из 90 случаев в цензусе), поэтому защищаем только однобуквенные
# («г. москва», инициалы) и явный белый список единиц измерения.
_NO_CAPITALIZE_AFTER = {"тыс", "руб", "млн", "млрд", "см", "кг", "км", "гг", "др", "пр", "напр", "ок"}


def _capitalize_after_sentence(m: re.Match) -> str:
    if m.group(1) == ".":
        before = m.string[:m.start()]
        word_match = re.search(r"(\w+)$", before)
        if word_match:
            word = word_match.group(1)
            if word.islower() and (len(word) == 1 or word in _NO_CAPITALIZE_AFTER):
                return m.group(0)
    return m.group(1) + " " + m.group(2).upper()


def regex_cleanup(text: str) -> str:
    """Чистка текста: звуки-паразиты, повторы, русская типографика.

    Первая буква НЕ капитализируется: ASR-сегмент может начинаться
    с середины предложения, а заглавную букву реплике ставит
    backend.turns при склейке.
    """
    text = HYPHEN_SPACE_RE.sub(r"\1-\2", text)
    text = _expand_abbreviations(text)
    text = FILLER_WORDS_RE.sub("", text)
    text = REPEATED_WORDS_RE.sub(r"\1", text)
    # Дубли пунктуации после удаления филлеров; до 2 проходов («, , ,»)
    for _ in range(2):
        text, n = DUP_PUNCT_RE.subn(r"\1", text)
        if not n:
            break
    text = SPACELESS_DASH_RE.sub(r"\1 – \2", text)
    text = DIGIT_RANGE_RE.sub(r"\1-\2", text)
    text = DASH_RE.sub(" – ", text)
    text = ELLIPSIS_CHAR_RE.sub("...", text)
    text = MANY_DOTS_RE.sub("...", text)
    text = GUILLEMET_RE.sub(r"«\1»", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = LEADING_PUNCT_RE.sub("", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = _CAPITALIZE_BASE_RE.sub(_capitalize_after_sentence, text)
    return text.strip()


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
    return genai.GenerativeModel(GEMINI_MODEL)


_gemini_model = None
_gemini_lock = threading.Lock()


GEMINI_MAX_RETRIES = 3
GEMINI_BACKOFF = [2, 5, 10]  # seconds


def gemini_polish(text: str) -> str:
    """Полировка текста через Gemini API с retry на rate-limit/5xx."""
    global _gemini_model
    if _gemini_model is None:
        with _gemini_lock:
            if _gemini_model is None:
                _gemini_model = _get_gemini_model()
    if _gemini_model is None:
        return text

    prompt = (
        "Ты — корректор ДОСЛОВНОЙ стенограммы телеинтервью на русском языке.\n"
        "Правила:\n"
        "1. НЕ удаляй и НЕ добавляй слова. Разговорные «ну», «вот», «как бы», "
        "«типа», «короче», «значит» — часть стенограммы, сохраняй их.\n"
        "2. Убирай только нечленораздельные звуки: «эээ», «ммм», «хм».\n"
        "3. Исправляй явные ошибки распознавания по контексту, особенно имена "
        "собственные: «масс-фильм» → «Мосфильм».\n"
        "4. Имена собственные пиши с заглавной буквы.\n"
        "5. Названия (фильмы, песни, театры, каналы) бери в кавычки-ёлочки «».\n"
        "6. Знаки препинания расставляй по смыслу. Тире — с пробелами: "
        "«слово – слово». Дефисы внутри слов без пробелов: «что -то» → «что-то».\n"
        "7. НЕ меняй порядок слов и НЕ переформулируй.\n"
        "8. НЕ меняй первую букву фрагмента: фрагмент может начинаться "
        "с середины предложения.\n"
        "9. Если текст содержит разделитель ---SEGMENT_BREAK---, "
        "сохрани его без изменений на отдельной строке.\n"
        "10. Годы, возраст, суммы, даты пиши цифрами с наращением через "
        "дефис: «в девяносто третьем» → «в 93-м», «с семи до одиннадцати "
        "лет» → «с 7-ми до 11-ти». Диапазоны — дефис без пробелов: «20-25». "
        "Малые количества (один-десять предметов) оставляй прописью: «два "
        "спектакля». Это ЕДИНСТВЕННОЕ разрешённое изменение формы слов.\n"
        "11. Верни ТОЛЬКО текст, без комментариев.\n\n"
        "Пример:\n"
        "Вход: ну вот мы эээ и поехали на масс -фильм снимать вечную любовь\n"
        "Выход: ну вот мы и поехали на «Мосфильм» снимать «Вечную любовь»\n\n"
        f"Текст:\n{text}"
    )

    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            # temperature=0: корректор должен быть детерминированным
            response = _gemini_model.generate_content(
                prompt,
                generation_config={"temperature": 0.0},
            )
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

    segments = [seg for seg in segments if seg["text"].strip()]

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
