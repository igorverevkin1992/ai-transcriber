import json
import re
import threading
import time

from backend.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GLOSSARY_REPLACEMENTS,
    TECH_MOMENT_DETECTION,
    TRANSCRIPT_GLOSSARY,
    logger,
)
from backend.metrics import gemini_calls
from backend.turns import TECH_BREAK_TEXT, UNCLEAR_TEXT

# Полная скобочная ремарка целиком (как в backend.turns) — такие фрагменты не
# являются речью и не классифицируются как технические моменты.
_FULL_PARENTHETICAL_RE = re.compile(r"^\((?:[^()]|\([^()]*\))*\)[.!?…]*$")


class GeminiPolishError(Exception):
    """Окончательный сбой Gemini-полировки (после всех ретраев)."""


def _parse_glossary_replacements(raw: str) -> list[tuple[re.Pattern, str]]:
    """Разбирает "неверно=>верно,..." в список (скомпилированный regex, замена).

    Для одиночных слов используются границы слова; внутри фраз пробелы матчат
    любой пробельный разрыв. Совпадение регистронезависимо.
    """
    pairs: list[tuple[re.Pattern, str]] = []
    for chunk in raw.split(","):
        if "=>" not in chunk:
            continue
        wrong, _, right = chunk.partition("=>")
        wrong = wrong.strip()
        right = right.strip()
        if not wrong or not right:
            continue
        escaped = re.escape(wrong).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        pairs.append((pattern, right))
    return pairs


GLOSSARY_REPLACEMENT_PAIRS = _parse_glossary_replacements(GLOSSARY_REPLACEMENTS)


def apply_glossary_replacements(text: str) -> str:
    """Детерминированные замены из GLOSSARY_REPLACEMENT_PAIRS.

    Замена (правильное написание имени собственного) подставляется как задано —
    регистр считается авторитетным, потому что это имена/термины.
    """
    for pattern, replacement in GLOSSARY_REPLACEMENT_PAIRS:
        text = pattern.sub(replacement, text)
    return text

# «что -то» → «что-то»: пробел ПЕРЕД дефисом, но не после.
# Пробелы с обеих сторон — это тире, его обрабатывает DASH_RE ниже.
HYPHEN_SPACE_RE = re.compile(r"(\w) +-(\w)")
# Эталонные стенограммы почти дословные: «ну», «вот», «как бы», «типа»,
# «короче» в них СОХРАНЯЮТСЯ. Удаляем только нечленораздельные звуки,
# которые человек никогда не переносит в текст. Дефисные варианты («э-э»,
# «м-м-м») — частая запись мычания у Whisper. «Хм» НЕ удаляем: эталон ф5
# сохраняет «Хм…» как осмысленную реплику.
FILLER_WORDS_RE = re.compile(r"\b(э(?:-э)+|м(?:-м)+|а(?:-а)+|э{2,}|эм+|ммм+)\b", re.IGNORECASE)
_FILLER_WORD_RE = re.compile(r"^(э(?:-э)*|м(?:-м)*|а(?:-а)*|э{2,}|эм+|ммм+)$", re.IGNORECASE)
REPEATED_WORDS_RE = re.compile(r"\b([а-яА-ЯёЁa-zA-Z]+)(?:\s+\1\b){2,}", re.IGNORECASE)
# Whisper-петля: фраза из 2-5 слов, повторённая 3+ раз подряд («и мы пошли
# и мы пошли и мы пошли»). Дефисные повторы («да-да-да») не трогаем —
# эталоны их сохраняют.
REPEATED_PHRASE_RE = re.compile(
    r"\b((?:[а-яА-ЯёЁa-zA-Z]+\s+){1,4}[а-яА-ЯёЁa-zA-Z]+)(?:\s+\1\b){2,}", re.IGNORECASE,
)
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


_POST_ABBR_CAP_RE = re.compile(
    r"(то есть|так далее|тому подобное)\s+([А-ЯЁ])([а-яё])",
)


def _expand_abbreviations(text: str) -> str:
    for rx, repl in _ABBR_EXPANSIONS:
        text = rx.sub(lambda m, r=repl: r.capitalize() if m.group(0)[0] == "Т" else r, text)
    text = _POST_ABBR_CAP_RE.sub(lambda m: f"{m.group(1)} {m.group(2).lower()}{m.group(3)}", text)
    return text
# Эталоны используют "..." (451 случай), а не "…" (74, один файл)
ELLIPSIS_CHAR_RE = re.compile(r"…")
MANY_DOTS_RE = re.compile(r"\.{4,}")
LOW9_QUOTE_RE = re.compile("\u201e([^\u201e\u201c\u201d\"\n]+)[\u201c\u201d\"]")
CURLY_QUOTE_RE = re.compile("\u201c([^\u201c\u201e\"\n]+)\u201d")
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
            # «я» — не сокращение, а частый конец предложения («это был я.»)
            if word.islower() and ((len(word) == 1 and word != "я") or word in _NO_CAPITALIZE_AFTER):
                return m.group(0)
    return m.group(1) + " " + m.group(2).upper()


def regex_cleanup(text: str) -> str:
    """Чистка текста: звуки-паразиты, повторы, русская типографика.

    Первая буква НЕ капитализируется: ASR-сегмент может начинаться
    с середины предложения, а заглавную букву реплике ставит
    backend.turns при склейке.
    """
    text = apply_glossary_replacements(text)
    text = HYPHEN_SPACE_RE.sub(r"\1-\2", text)
    text = _expand_abbreviations(text)
    text = FILLER_WORDS_RE.sub("", text)
    text = REPEATED_WORDS_RE.sub(r"\1", text)
    text = REPEATED_PHRASE_RE.sub(r"\1", text)
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
    text = LOW9_QUOTE_RE.sub(r"«\1»", text)
    text = CURLY_QUOTE_RE.sub(r"«\1»", text)
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

# LLM изредка добавляет преамбулу вопреки правилу «верни ТОЛЬКО текст»
_GEMINI_PREAMBLE_RE = re.compile(
    r"^(?:вот\s+)?(?:исправленный|откорректированный|итоговый|готовый)\s+(?:текст|вариант):?\s*",
    re.IGNORECASE,
)


def _clean_gemini_response(result: str) -> str:
    """Срезает markdown-обёртку и преамбулы из ответа Gemini."""
    result = result.strip()
    if result.startswith("```"):
        result = re.sub(r"^```[a-zа-яё]*\s*\n?", "", result)
        result = re.sub(r"\n?```\s*$", "", result)
    result = _GEMINI_PREAMBLE_RE.sub("", result)
    return result.strip()


def _gemini_call(prompt: str) -> str | None:
    """Вызов Gemini с ретраями на rate-limit/5xx. Общее ядро для полировки и
    классификации технических моментов.

    Возвращает очищенный текст ответа (может быть пустым), ``None`` — если модель
    недоступна (нет пакета/ключа). Окончательный сбой после ретраев →
    ``GeminiPolishError``.
    """
    global _gemini_model
    if _gemini_model is None:
        with _gemini_lock:
            if _gemini_model is None:
                _gemini_model = _get_gemini_model()
    if _gemini_model is None:
        return None

    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            # temperature=0: корректор/классификатор должен быть детерминированным
            response = _gemini_model.generate_content(
                prompt,
                generation_config={"temperature": 0.0},
            )
            result = _clean_gemini_response(response.text)
            gemini_calls.labels(outcome="success").inc()
            return result
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
            raise GeminiPolishError(str(e)) from e
    return None


def gemini_polish(text: str) -> str:
    """Полировка текста через Gemini API с retry на rate-limit/5xx."""
    glossary_rule = ""
    if TRANSCRIPT_GLOSSARY:
        glossary_rule = (
            f"\nГлоссарий правильных написаний (имена, термины, названия) — "
            f"приводи распознанные варианты к ним: {TRANSCRIPT_GLOSSARY}.\n"
        )

    prompt = (
        "Ты — корректор ДОСЛОВНОЙ стенограммы телеинтервью на русском языке.\n"
        f"{glossary_rule}"
        "Правила:\n"
        "1. НЕ удаляй и НЕ добавляй слова. Разговорные «ну», «вот», «как бы», "
        "«типа», «короче», «значит», «хм» — часть стенограммы, сохраняй их. "
        "Нечленораздельные звуки уже удалены до тебя.\n"
        "2. Исправляй явные ошибки распознавания по контексту, особенно имена "
        "собственные: «масс-фильм» → «Мосфильм». Иностранные термины пиши "
        "латиницей, если так принято: «старкволити» → «star quality».\n"
        "3. Имена собственные пиши с заглавной буквы.\n"
        "4. Названия (фильмы, песни, театры, каналы) бери в кавычки-ёлочки «».\n"
        "5. Знаки препинания расставляй по смыслу. Тире — с пробелами: "
        "«слово – слово». Дефисы внутри слов без пробелов: «что -то» → «что-то».\n"
        "6. НЕ меняй порядок слов и НЕ переформулируй.\n"
        "7. НЕ меняй первую букву фрагмента: фрагмент может начинаться "
        "с середины предложения.\n"
        "8. Если текст содержит разделитель ---SEGMENT_BREAK---, "
        "сохрани его без изменений на отдельной строке.\n"
        "9. Годы, возраст, суммы, даты пиши цифрами с наращением через "
        "дефис: «в девяносто третьем» → «в 93-м», «с семи до одиннадцати "
        "лет» → «с 7-ми до 11-ти». Диапазоны — дефис без пробелов: «20-25». "
        "Малые количества (один-десять предметов) оставляй прописью: «два "
        "спектакля». Это ЕДИНСТВЕННОЕ разрешённое изменение формы слов.\n"
        "10. Верни ТОЛЬКО текст, без комментариев.\n\n"
        "Пример:\n"
        "Вход: ну вот мы и поехали на масс -фильм снимать вечную любовь\n"
        "Выход: ну вот мы и поехали на «Мосфильм» снимать «Вечную любовь»\n\n"
        f"Текст:\n{text}"
    )

    result = _gemini_call(prompt)
    if result is None:
        return text
    return result or text


_TECH_MOMENT_PROMPT = (
    "Ты — редактор стенограммы телеинтервью на русском языке. Ниже —\n"
    "пронумерованные фрагменты речи из записи.\n"
    "Интервью состоит из вопросов ведущего (голос за кадром) и ответов гостя\n"
    "по темам беседы.\n"
    "Найди фрагменты, которые ЯВНО НЕ относятся к интервью: команды съёмочной\n"
    "группы, перезапуск/настройка камеры, проверка микрофона, технические\n"
    "указания, обращения к оператору или режиссёру, отсчёт, хлопушка.\n"
    "ВАЖНО: при ЛЮБОМ сомнении НЕ помечай фрагмент — лучше оставить лишнюю\n"
    "фразу, чем потерять реплику интервью. Большинство фрагментов — НЕ\n"
    "технические.\n"
    "Верни ТОЛЬКО номера явно технических фрагментов через запятую (например:\n"
    "2, 5). Если таких нет — верни одно слово: НЕТ.\n\n"
    "Фрагменты:\n"
)

_TECH_NUMBERS_RE = re.compile(r"\d+")
_TECH_BATCH_CHARS = 5000


def detect_technical_segments(
    segments: list[dict],
    warnings: list[str] | None = None,
) -> list[dict]:
    """Консервативно помечает реплики съёмочной группы маркером тех. момента.

    Через Gemini находит ЯВНО не-интервью фрагменты (перезапуск камеры,
    проверка микрофона, команды группе) и заменяет их текст на ``TECH_BREAK_TEXT``
    — далее ``build_turns`` выведет их как курсивную ремарку. При сомнении
    фрагмент остаётся без изменений. При сбое Gemini — предупреждение в
    ``warnings``, текст не меняется.
    """
    if not segments:
        return segments

    candidates = [
        i for i, seg in enumerate(segments)
        if seg["text"].strip()
        and seg["text"] != UNCLEAR_TEXT
        and not _FULL_PARENTHETICAL_RE.match(seg["text"])
    ]
    if not candidates:
        return segments

    # Батчинг по символам с нумерацией фрагментов внутри батча.
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0
    for i in candidates:
        t = segments[i]["text"]
        if cur and cur_len + len(t) > _TECH_BATCH_CHARS:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(i)
        cur_len += len(t) + 8  # запас на нумерацию/переносы
    if cur:
        batches.append(cur)

    failed = False
    for batch in batches:
        numbered = "\n".join(
            f"{n}. {segments[idx]['text']}" for n, idx in enumerate(batch, 1)
        )
        try:
            result = _gemini_call(_TECH_MOMENT_PROMPT + numbered)
        except GeminiPolishError:
            failed = True
            continue
        if not result:
            continue
        flagged = {int(m) for m in _TECH_NUMBERS_RE.findall(result)}
        for n, idx in enumerate(batch, 1):
            if n in flagged:
                segments[idx]["text"] = TECH_BREAK_TEXT

    if warnings is not None and failed:
        warnings.append(
            "Определение технических моментов не выполнено (сбой Gemini) — "
            "реплики съёмочной группы могли остаться в тексте."
        )
    return segments


# --- Авто-определение имён гостей через Gemini ---

# Принимаем только имя-отчество (2 слова, второе — патроним): высокая точность
# под текущий кейс. Прочее (одно слово, фамилия, мусор) отбрасываем.
_NAME_PATRONYMIC_RE = re.compile(
    r"^[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]*(?:ович|евич|инична|ична|евна|овна|ьич|ич)$"
)
# Ограничение размера транскрипта в промпте (символы): большинство интервью
# умещаются; обрезка идёт с конца.
_NAME_INFER_MAX_CHARS = 40000


def gemini_infer_speaker_names(
    segments: list[dict],
    *,
    interviewer_id: str | None,
    guest_ids: list[str],
) -> dict[str, str] | None:
    """Определить имя-отчество гостей через Gemini по тому, как к ним обращаются.

    Возвращает ``{speaker_id: "Имя Отчество"}`` только для уверенно опознанных
    гостей из ``guest_ids`` (интервьюер исключён). ``None`` — если Gemini
    недоступен или произошёл сбой (тогда вызывающий код берёт эвристику-фолбэк).
    """
    guest_set = {str(g) for g in guest_ids}
    if not guest_set:
        return None

    lines: list[str] = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text or text in (UNCLEAR_TEXT, TECH_BREAK_TEXT):
            continue
        lines.append(f"[{seg['speaker']}] {text}")
    if not lines:
        return None
    transcript = "\n".join(lines)[:_NAME_INFER_MAX_CHARS]

    interviewer_line = ""
    if interviewer_id is not None:
        interviewer_line = (
            f"Говорящий [{interviewer_id}] — интервьюер за кадром (задаёт вопросы), "
            "его НЕ именуй.\n"
        )
    prompt = (
        "Это стенограмма телеинтервью на русском языке. Каждая реплика помечена "
        "ID говорящего в квадратных скобках, например [0], [1].\n"
        f"{interviewer_line}"
        "Определи имя и отчество говорящих ПО ТОМУ, КАК К НИМ ОБРАЩАЮТСЯ по "
        "имени-отчеству в репликах (именно прямое обращение, а не упоминание "
        f"третьих лиц). Нужны имена для ID: {', '.join(sorted(guest_set))}.\n"
        "Верни СТРОГО JSON-объект вида {\"<id>\": \"Имя Отчество\"}. Если для "
        "говорящего нельзя уверенно определить имя-отчество — поставь null. "
        "Никакого текста кроме JSON.\n\n"
        f"Стенограмма:\n{transcript}"
    )

    try:
        result = _gemini_call(prompt)
    except GeminiPolishError:
        return None
    if not result:
        return None

    match = re.search(r"\{.*\}", result, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    out: dict[str, str] = {}
    for sp, name in data.items():
        sp = str(sp)
        if sp not in guest_set or not isinstance(name, str):
            continue
        name = name.strip()
        if _NAME_PATRONYMIC_RE.match(name):
            out[sp] = name
    return out or None


def postprocess_segments(
    segments: list[dict],
    use_gemini: bool = True,
    warnings: list[str] | None = None,
) -> list[dict]:
    """Постобработка сегментов: regex-чистка + опционально Gemini-полировка.

    Батчит сегменты для Gemini (~5000 символов за раз) для экономии API-вызовов.
    Если передан ``warnings``, при сбоях Gemini туда добавляется предупреждение
    (текст остаётся без AI-коррекции, но glossary/regex уже применены).
    """
    if not segments:
        return segments

    for seg in segments:
        seg["text"] = regex_cleanup(seg["text"])
        if seg.get("words"):
            seg["words"] = [w for w in seg["words"] if not _FILLER_WORD_RE.match(w.get("text", "").strip())]

    segments = [seg for seg in segments if seg["text"].strip()]

    if not use_gemini or not GEMINI_API_KEY:
        return segments

    # Консервативно сворачиваем крон-чаттер в маркер тех. момента ДО полировки.
    if TECH_MOMENT_DETECTION:
        detect_technical_segments(segments, warnings=warnings)

    batch_texts = []
    batch_indices = []
    current_batch = ""
    current_indices = []
    SEPARATOR = "\n---SEGMENT_BREAK---\n"
    MAX_BATCH_CHARS = 5000

    for i, seg in enumerate(segments):
        text = seg["text"]
        # Служебные маркеры (неразборчиво / тех. момент) Gemini не полирует
        if not text or text in (UNCLEAR_TEXT, TECH_BREAK_TEXT):
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

    failed_batches = 0
    for batch_text, indices in zip(batch_texts, batch_indices):
        try:
            polished = gemini_polish(batch_text)
        except GeminiPolishError:
            failed_batches += 1
            continue  # текст остаётся без AI-коррекции
        parts = polished.split("---SEGMENT_BREAK---")
        parts = [p.strip() for p in parts]

        if len(parts) == len(indices):
            for idx, polished_text in zip(indices, parts):
                if polished_text:
                    segments[idx]["text"] = polished_text
        else:
            # batch didn't split cleanly — polish individually
            for idx in indices:
                try:
                    polished_text = gemini_polish(segments[idx]["text"])
                except GeminiPolishError:
                    failed_batches += 1
                    break
                if polished_text:
                    segments[idx]["text"] = polished_text

    if warnings is not None and failed_batches > 0:
        warnings.append(
            f"Gemini-полировка не выполнена для {failed_batches} фрагментов — "
            "возможны ошибки в именах собственных и пунктуации."
        )

    return segments
