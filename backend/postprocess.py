import json
import re
import threading
import time

from backend.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MODEL_SMART,
    GEMINI_TIMEOUT_SECONDS,
    GLOSSARY_REPLACEMENTS,
    TECH_MOMENT_AGGRESSIVE,
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
# Whisper-петля в дефисной форме: «оп-оп-оп-…»×100. Живую речь («да-да-да»,
# 3 повтора) эталоны сохраняют, поэтому схлопываем только 5+ повторов — до трёх.
HYPHEN_LOOP_RE = re.compile(r"\b([а-яА-ЯёЁa-zA-Z]+)(?:-\1){4,}\b", re.IGNORECASE)
# ASR-артефакт: серия «Участник 2. Участник 3. …» (метки говорящих, просочившиеся
# в текст) — удаляем только ПОВТОРЯЮЩИЕСЯ подряд (одиночное «участник» — речь).
PARTICIPANT_LOOP_RE = re.compile(r"(?:\bУчастник \d+(?:-\d+)?[.,]?\s*){2,}", re.IGNORECASE)
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
    text = HYPHEN_LOOP_RE.sub(lambda m: f"{m.group(1)}-{m.group(1).lower()}-{m.group(1).lower()}", text)
    text = PARTICIPANT_LOOP_RE.sub("", text)
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


def _get_gemini_client():
    """Возвращает клиент Gemini (новый SDK ``google-genai``).

    Легаси-пакет ``google-generativeai`` объявлен deprecated; здесь используется
    унифицированный ``google-genai`` с ``genai.Client``. Модель передаётся не при
    создании клиента, а в каждом вызове ``generate_content`` (см. ``_gemini_call``).
    """
    try:
        from google import genai
    except ImportError:
        logger.warning("google-genai не установлен. Gemini-полировка недоступна.")
        return None

    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY не задан. Gemini-полировка недоступна.")
        return None

    try:
        # timeout — в миллисекундах на КАЖДЫЙ HTTP-запрос (включая vision):
        # зависший сокет больше не держит постобработку неограниченно.
        return genai.Client(
            api_key=GEMINI_API_KEY,
            http_options={"timeout": GEMINI_TIMEOUT_SECONDS * 1000},
        )
    except Exception as e:
        # Сбой создания клиента (например, неподдерживаемый httpx прокси вида
        # socks4://, кривой ключ) НЕ должен ронять всю задачу — полировка
        # опциональна. Отключаем её и продолжаем без Gemini.
        logger.warning(
            "Не удалось создать клиент Gemini (%s). Полировка отключена. "
            "Частая причина — прокси: httpx не поддерживает socks4, используйте "
            "socks5:// (плюс пакет httpx[socks]) или http-прокси.",
            e,
        )
        return None


_gemini_client = None
_gemini_lock = threading.Lock()
# Модели, вернувшие 404 (не существуют для этого ключа/версии API): все
# последующие вызовы с ними сразу идут на GEMINI_MODEL, без повторных 404.
_broken_models: set[str] = set()


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


def _gemini_ready() -> bool:
    """Доступен ли Gemini-клиент (лениво создаёт и кэширует).

    False = ключ не задан / пакет отсутствует / клиент не создался (прокси и
    т.п.) — в этом случае все смысловые пассы деградируют, и вызывающий код
    обязан сделать это ВИДИМЫМ (warning в UI/лог), а не молчать.
    """
    global _gemini_client
    if _gemini_client is None:
        with _gemini_lock:
            if _gemini_client is None:
                _gemini_client = _get_gemini_client()
    return _gemini_client is not None


def _list_available_models(limit: int = 20) -> str:
    """Имена доступных gemini-моделей (для подсказки при неверном ID в конфиге)."""
    try:
        names = []
        for m in _gemini_client.models.list():
            name = (getattr(m, "name", "") or "").removeprefix("models/")
            if "gemini" in name:
                names.append(name)
            if len(names) >= limit:
                break
        return ", ".join(names) or "список пуст"
    except Exception as e:
        return f"не удалось получить список ({e})"


def gemini_health_check() -> tuple[bool, str | None]:
    """Однократная проверка работоспособности Gemini (для старта сервера).

    Возвращает (ok, причина-если-нет). Реальный минимальный вызов API: ловит и
    ошибку создания клиента (ключ/прокси), и сетевые/авторизационные сбои.
    Дополнительно проверяет GEMINI_MODEL_SMART (если отличается): её 404 не
    делает ok=False (база работает, пассы откатятся сами), но громко логируется
    вместе со списком доступных моделей.
    """
    if not _gemini_ready():
        return False, ("клиент не создан — проверьте GEMINI_API_KEY и настройки "
                       "прокси (подробности выше в логе)")
    try:
        result = _gemini_call("Ответь одним словом: да")
    except GeminiPolishError as e:
        return False, str(e)
    if result is None:
        return False, "клиент недоступен"

    if GEMINI_MODEL_SMART != GEMINI_MODEL and GEMINI_MODEL_SMART not in _broken_models:
        try:
            _gemini_call("Ответь одним словом: да", model=GEMINI_MODEL_SMART)
        except GeminiPolishError as e:
            logger.error("GEMINI_MODEL_SMART='%s': ошибка проверки (%s)",
                         GEMINI_MODEL_SMART, e)
        if GEMINI_MODEL_SMART in _broken_models:
            logger.error(
                "GEMINI_MODEL_SMART='%s' не существует для этого ключа — умные "
                "пассы будут работать на '%s'. Доступные модели: %s",
                GEMINI_MODEL_SMART, GEMINI_MODEL, _list_available_models(),
            )
    return True, None


def _gemini_call(prompt: str, model: str | None = None) -> str | None:
    """Вызов Gemini с ретраями на rate-limit/5xx. Общее ядро для полировки и
    классификации технических моментов.

    ``model`` — переопределение модели (умные пассы используют
    ``GEMINI_MODEL_SMART``); по умолчанию ``GEMINI_MODEL``. Несуществующая
    override-модель (404 NOT_FOUND — неверный ID для этого ключа) НЕ роняет пасс:
    вызов автоматически откатывается на ``GEMINI_MODEL``, модель запоминается в
    ``_broken_models`` и дальше не пробуется.
    Возвращает очищенный текст ответа (может быть пустым), ``None`` — если модель
    недоступна (нет пакета/ключа). Окончательный сбой после ретраев →
    ``GeminiPolishError``.
    """
    if not _gemini_ready():
        return None

    use_model = model or GEMINI_MODEL
    if use_model in _broken_models:
        use_model = GEMINI_MODEL

    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            # temperature=0: корректор/классификатор должен быть детерминированным
            response = _gemini_client.models.generate_content(
                model=use_model,
                contents=prompt,
                config={"temperature": 0.0},
            )
            result = _clean_gemini_response(response.text)
            gemini_calls.labels(outcome="success").inc()
            return result
        except Exception as e:
            err_str = str(e).lower()
            is_404 = "not_found" in err_str or "is not found" in err_str or "404" in err_str
            if is_404 and use_model != GEMINI_MODEL:
                # Неверный ID умной модели: откат на базовую и запоминаем,
                # чтобы не ловить 404 на каждом батче.
                logger.warning(
                    "Модель Gemini '%s' не найдена (404) — откат на '%s' для этого "
                    "и последующих вызовов. Список доступных моделей печатается "
                    "при старте сервера.", use_model, GEMINI_MODEL,
                )
                _broken_models.add(use_model)
                use_model = GEMINI_MODEL
                continue
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


def gemini_polish(text: str, context: str | None = None) -> str:
    """Полировка текста через Gemini API с retry на rate-limit/5xx.

    ``context`` — описание съёмки из паспорта: даёт корректору тему (правильные
    имена/термины по смыслу) и включает правило про иностранную речь — участники
    кулинарных/национальных съёмок часто переходят на другой язык, который ASR
    транслитерирует кириллицей в бессмыслицу.
    """
    glossary_rule = ""
    if TRANSCRIPT_GLOSSARY:
        glossary_rule = (
            f"\nГлоссарий правильных написаний (имена, термины, названия) — "
            f"приводи распознанные варианты к ним: {TRANSCRIPT_GLOSSARY}.\n"
        )
    context_rule = ""
    if context and context.strip():
        context_rule = (
            f"\nКонтекст съёмки: {context.strip()}\n"
            "Если участник по контексту говорит на иностранном языке, а фраза "
            "выглядит как кириллическая транслитерация иностранной речи — "
            "восстанови её на языке оригинала латиницей, ТОЛЬКО если уверен "
            "(например «гварда» → «guarda», «алора» → «allora»). Если ты УВЕРЕН, "
            "что это иностранная речь, но восстановить её нельзя (бессвязный "
            "транслит) — замени фразу целиком на «(говорит по-итальянски)» (или "
            "соответствующий язык). Если не уверен — оставь без изменений.\n"
        )

    prompt = (
        "Ты — корректор ДОСЛОВНОЙ стенограммы телеинтервью на русском языке.\n"
        f"{glossary_rule}"
        f"{context_rule}"
        "Правила:\n"
        "1. НЕ удаляй и НЕ добавляй слова. Разговорные «ну», «вот», «как бы», "
        "«типа», «короче», «значит», «хм» — часть стенограммы, сохраняй их. "
        "Нечленораздельные звуки уже удалены до тебя.\n"
        "2. Исправляй явные ошибки распознавания по контексту, особенно имена "
        "собственные: «масс-фильм» → «Мосфильм». Иностранные термины пиши "
        "латиницей, если так принято: «старкволити» → «star quality». "
        "СЛУЧАЙНОЕ одиночное английское слово посреди русской фразы — ошибка "
        "распознавания, замени русским по смыслу: «many разных судеб» → «много "
        "разных судеб»; осмысленные англ. фразы и названия (цитаты, песни) "
        "не трогай.\n"
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
    "указания, обращения к оператору или режиссёру, отсчёт, хлопушка,\n"
    "обсуждение ракурсов и планов съёмки («кто общий, кто крупный?», «я общий»,\n"
    "«сначала крупный план», «доски поближе к камере», «не надо снимать»),\n"
    "а также ПЕНИЕ — сам текст исполняемой песни, в том числе на английском\n"
    "(«Happy birthday to you», «I wanna be loved by you»): исполнение песни —\n"
    "постановочный момент съёмки, а не реплика интервью.\n"
    "ВАЖНО: при ЛЮБОМ сомнении НЕ помечай фрагмент — лучше оставить лишнюю\n"
    "фразу, чем потерять реплику интервью. Большинство фрагментов — НЕ\n"
    "технические.\n"
    "Верни ТОЛЬКО номера явно технических фрагментов через запятую (например:\n"
    "2, 5). Если таких нет — верни одно слово: НЕТ.\n\n"
    "Фрагменты:\n"
)

# Агрессивный режим (TECH_MOMENT_AGGRESSIVE): помимо явной техники помечает и
# закадровую организационную болтовню, которую человек обычно вырезает. Всё ещё
# с контрпримерами, чтобы не резать содержательные ответы по теме интервью.
_TECH_MOMENT_PROMPT_AGGRESSIVE = (
    "Ты — редактор стенограммы телеинтервью на русском языке. Ниже —\n"
    "пронумерованные фрагменты речи из записи.\n"
    "Интервью состоит из вопросов ведущего (голос за кадром) и ответов гостя\n"
    "ПО ТЕМЕ беседы.\n"
    "Найди фрагменты, которые НЕ относятся к содержанию интервью:\n"
    "- команды съёмочной группе, перезапуск/настройка камеры, проверка\n"
    "  микрофона, технические указания, обращения к оператору/режиссёру,\n"
    "  отсчёт, хлопушка;\n"
    "- закадровая организация процесса: обращения к посторонним по имени\n"
    "  (оператор, ассистенты), «давай с тебя», «кто-то ещё попробуйте»,\n"
    "  «пойду умоюсь», «подождите», обсуждение, кто и когда играет/снимает;\n"
    "- обсуждение ракурсов и планов съёмки: «кто общий, кто крупный?»,\n"
    "  «я общий», «сначала крупный план», «не надо это снимать»;\n"
    "- ПЕНИЕ: сам текст исполняемой песни, в том числе на английском\n"
    "  («Happy birthday to you», «I wanna be loved by you») — исполнение\n"
    "  и распевка это постановочная часть съёмки, а не интервью;\n"
    "- репетиция и обсуждение дублей ПЕРЕД записью номера: «Ещё раз?»,\n"
    "  «Другое делать? Или не надо?», «Давайте я вам спою эту, и в конце\n"
    "  поворачиваюсь», «А пропустит потом песню в эфир?», обсуждение жестов\n"
    "  и хлопков для камеры;\n"
    "- финальные командные реплики конца съёмки: «Так, ну всё», «Снято»,\n"
    "  «Спасибо, закончили».\n"
    "ОСТАВЛЯЙ как интервью: содержательные ответы и вопросы по теме, даже с\n"
    "разговорными «ну», «вот», «как бы»; реплики про сам предмет беседы.\n"
    "ОСТАВЛЯЙ также короткие ПОСТАНОВОЧНЫЕ фразы героя, произнесённые в кадр\n"
    "как часть съёмки («Hello, Марина», приветствие, обращение в камеру) — это\n"
    "содержимое кадра, а не организационная болтовня (пение — не исключение:\n"
    "тексты песен помечай).\n"
    "При сомнении между «по теме» и «организация» — НЕ помечай.\n"
    "Верни ТОЛЬКО номера технических/организационных фрагментов через запятую\n"
    "(например: 2, 5). Если таких нет — верни одно слово: НЕТ.\n\n"
    "Фрагменты:\n"
)

_TECH_NUMBERS_RE = re.compile(r"\d+")
_TECH_BATCH_CHARS = 5000


def detect_technical_segments(
    segments: list[dict],
    warnings: list[str] | None = None,
    crew_names: list[str] | None = None,
) -> list[dict]:
    """Консервативно помечает реплики съёмочной группы маркером тех. момента.

    Через Gemini находит ЯВНО не-интервью фрагменты (перезапуск камеры,
    проверка микрофона, команды группе) и заменяет их текст на ``TECH_BREAK_TEXT``
    — далее ``build_turns`` выведет их как курсивную ремарку. При сомнении
    фрагмент остаётся без изменений. При сбое Gemini — предупреждение в
    ``warnings``, текст не меняется.

    ``crew_names`` (из «паспорта съёмки») — известные члены съёмочной группы; их
    реплики и обращения к ним по имени надёжнее помечаются техническими.
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

    prompt = _TECH_MOMENT_PROMPT_AGGRESSIVE if TECH_MOMENT_AGGRESSIVE else _TECH_MOMENT_PROMPT
    crew = [str(c).strip() for c in (crew_names or []) if str(c).strip()]
    if crew:
        prompt += (
            "\nИзвестные члены съёмочной группы (НЕ участники интервью): "
            f"{', '.join(crew)}. Их реплики и обращения к ним по имени помечай "
            "техническими моментами.\n"
        )
    failed = False
    total_flagged = 0
    for batch in batches:
        numbered = "\n".join(
            f"{n}. {segments[idx]['text']}" for n, idx in enumerate(batch, 1)
        )
        try:
            result = _gemini_call(prompt + numbered, model=GEMINI_MODEL_SMART)
        except GeminiPolishError:
            failed = True
            continue
        if not result:
            continue
        flagged = {int(m) for m in _TECH_NUMBERS_RE.findall(result)}
        for n, idx in enumerate(batch, 1):
            if n in flagged:
                segments[idx]["text"] = TECH_BREAK_TEXT
                total_flagged += 1

    if warnings is not None and failed:
        warnings.append(
            "Определение технических моментов не выполнено (сбой Gemini) — "
            "реплики съёмочной группы могли остаться в тексте."
        )
    logger.info("Техмоменты (Gemini): помечено %d фрагментов%s",
                total_flagged, " (были сбои батчей)" if failed else "")
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
        result = _gemini_call(prompt, model=GEMINI_MODEL_SMART)
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


# --- Извлечение «паспорта съёмки» через Gemini (фолбэк к детерминированному парсеру) ---

_PASSPORT_MAX_CHARS = 20000


def gemini_extract_passport(text: str) -> dict | None:
    """Извлечь из сырого текста паспорта структуру через Gemini (фолбэк).

    Возвращает `{"speakers": [имена], "num_heroes": int, "description": str}` или
    `None` (нет Gemini / сбой / пусто). Используется, когда детерминированный
    разбор формы (`backend.passport.parse_passport`) не распознал поля.
    """
    text = (text or "").strip()
    if not text:
        return None
    prompt = (
        "Это «паспорт съёмки» — заполненная ассистентом форма о видеосъёмке "
        "интервью. Извлеки СТРОГО JSON-объект вида "
        '{"heroes": ["Имя ..."], "num_heroes": N, "host": true, '
        '"crew": ["Имя ..."], "description": "..."}.\n'
        "heroes — имена героев (гостей) в кадре; num_heroes — их число (целое); "
        "host — есть ли закадровый ведущий/автор/корреспондент (true/false); "
        "crew — имена съёмочной группы (оператор, инженер, продюсер, ассистенты), "
        "НЕ гости; description — краткое описание/тема съёмки. Ведущего и группу в "
        'heroes НЕ включай. Если поля нет — [] / 0 / true / "". '
        "Никакого текста кроме JSON.\n\n"
        f"Паспорт:\n{text[:_PASSPORT_MAX_CHARS]}"
    )
    try:
        result = _gemini_call(prompt, model=GEMINI_MODEL_SMART)
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

    def _names(key):
        val = data.get(key) or []
        if not isinstance(val, list):
            return []
        return [str(x).strip() for x in val if str(x).strip()]

    heroes = _names("heroes")
    crew = _names("crew")
    num = data.get("num_heroes")
    num = int(num) if isinstance(num, (int, float)) and num > 0 else len(heroes)
    has_host = data.get("host")
    has_host = False if has_host is False else True  # дефолт True
    desc = str(data.get("description") or "").strip()
    if not heroes and num <= 0 and not desc and not crew:
        return None
    return {"speakers": heroes, "num_heroes": num, "has_host": has_host,
            "crew": crew, "description": desc}


# --- Gemini-правка границ спикеров ---

_BOUNDARY_MAX_CHARS = 40000

# Разрез текста на предложения (для split_after в правке границ).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def _split_sentences(text: str) -> list[str]:
    return [p for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p]


def _is_marker_text(text: str) -> bool:
    """Реплика-ремарка (техмомент / неразборчиво / целиком в скобках) — не речь."""
    text = (text or "").strip()
    return (not text) or text in (UNCLEAR_TEXT, TECH_BREAK_TEXT) or bool(_FULL_PARENTHETICAL_RE.match(text))


def _merge_adjacent_same_speaker(segments: list[dict]) -> list[dict]:
    """Склеить соседние одно-спикерные речевые реплики (после переназначения
    метки соседи могли стать одним говорящим). Ремарки не склеиваются."""
    out: list[dict] = []
    for seg in segments:
        if (out and not _is_marker_text(seg.get("text", "")) and not _is_marker_text(out[-1].get("text", ""))
                and str(out[-1].get("speaker")) == str(seg.get("speaker"))):
            joined = (out[-1]["text"].rstrip() + " " + seg["text"].strip()).strip()
            out[-1] = {**out[-1], "text": joined}
        else:
            out.append(seg)
    return out


def correct_speaker_boundaries(
    segments: list[dict],
    *,
    speaker_labels: dict[str, str],
    interviewer_id: str | None,
    description: str = "",
    warnings: list[str] | None = None,
    merge_adjacent: bool = True,
) -> list[dict]:
    """Переназначить спикера у реплик, которые диаризация отнесла ЦЕЛИКОМ не тому
    говорящему (ответ гостя в абзаце ведущего и наоборот), через Gemini.

    ``segments`` — список dict с полями ``speaker``/``text``: либо склеенные
    реплики после ``build_turns``, либо сырые ASR-события ДО склейки (мелкая
    гранулярность ≈ предложение — переназначение целого события фактически
    разрезает склейку мульти-спикерной реплики). Для событий передавайте
    ``merge_adjacent=False`` — склейкой займётся последующий ``build_turns``.
    ``speaker_labels`` — ``{speaker_id: отображаемое имя}`` (интервьюер → АЗК,
    гости → имена, прочие → «Спикер N») — только контекст промпта; переназначение
    возвращается в исходных id. При недоступности Gemini / сбое / некорректном
    ответе — возвращается ИСХОДНЫЙ список без изменений (как прочие Gemini-функции).
    """
    indexed = [(i, seg) for i, seg in enumerate(segments) if not _is_marker_text(seg.get("text", ""))]
    if len(indexed) < 2:
        return segments

    valid_ids = {str(seg.get("speaker")) for _, seg in indexed}
    if len(valid_ids) < 2:
        return segments  # один говорящий — переназначать нечего

    n_to_idx = {n: idx for n, (idx, _) in enumerate(indexed)}
    lines = []
    for n, (_, seg) in enumerate(indexed):
        sid = str(seg.get("speaker"))
        label = speaker_labels.get(sid, sid)
        lines.append(f"{n}\t[{sid}: {label}] {seg['text'].strip()}")
    transcript = "\n".join(lines)[:_BOUNDARY_MAX_CHARS]

    roles = []
    if interviewer_id is not None:
        roles.append(f"[{interviewer_id}] — интервьюер (АЗК): задаёт вопросы из-за кадра.")
    for sid, label in speaker_labels.items():
        if sid != interviewer_id:
            roles.append(f"[{sid}] — гость: {label}.")
    roles_block = "\n".join(roles)
    desc_block = f"Тема съёмки: {description.strip()}\n" if description and description.strip() else ""

    prompt = (
        "Это стенограмма телеинтервью на русском. Каждая строка — реплика с "
        "НОМЕРОМ, метка спикера в скобках, например `12\t[0: АЗК] текст`.\n"
        f"{desc_block}"
        "Роли говорящих:\n"
        f"{roles_block}\n\n"
        "Автоматическая диаризация ИЗРЕДКА присваивает реплику целиком не тому "
        "спикеру. Найди реплики с ЯВНО неверной меткой по логике диалога: вопрос/"
        "побуждение/реакция ведущего — это интервьюер; содержательный ответ или "
        "монолог по теме — гость; реплика, продолжающая предложение предыдущего "
        "говорящего, принадлежит ЕМУ. Исправляй ТОЛЬКО когда уверен; сомневаешься "
        "— не трогай.\n"
        "Два вида исправлений:\n"
        '1) вся реплика не того спикера: {"id": НОМЕР, "speaker": "<id>"};\n'
        "2) внутри реплики после какого-то предложения говорящий СМЕНИЛСЯ "
        "(склейка двух голосов): {\"id\": НОМЕР, \"split_after\": K, "
        '"tail_speaker": "<id>"} — после K-го предложения (счёт с 1) хвост '
        "принадлежит tail_speaker. Пример: реплика гостя «…хранить буррату "
        "тёплой. Кстати, вы говорили про моцареллу.» — последнее предложение "
        "начинает НОВУЮ мысль другого голоса → split_after на границе. "
        "Типичная склейка: короткая реакция или риторический вопрос ВЕДУЩЕГО, "
        "приклеенный к КОНЦУ длинного ответа гостя («…я ответила спасибо, мы "
        "расстались. Когда же ещё с таким праздником поздравят при жизни?» — "
        "последняя фраза явно принадлежит интервьюеру) → split_after.\n"
        "Верни СТРОГО JSON-массив исправлений. Если исправлять нечего — верни []. "
        "Никакого текста кроме JSON.\n\n"
        f"Стенограмма:\n{transcript}"
    )

    def _fail(reason: str) -> list[dict]:
        logger.warning("Правка границ спикеров НЕ выполнена: %s", reason)
        if warnings is not None:
            warnings.append(f"Правка границ спикеров не выполнена ({reason}).")
        return segments

    try:
        result = _gemini_call(prompt, model=GEMINI_MODEL_SMART)
    except GeminiPolishError as e:
        return _fail(f"сбой Gemini: {e}")
    if result is None:
        return _fail("Gemini недоступен — ключ/прокси")
    if not result:
        return _fail("пустой ответ Gemini")
    match = re.search(r"\[.*\]", result, re.DOTALL)
    if not match:
        return _fail("нераспознанный ответ Gemini")
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return _fail("некорректный JSON в ответе Gemini")
    if not isinstance(data, list):
        return _fail("некорректный JSON в ответе Gemini")

    out = [dict(s) for s in segments]
    reassigned = 0
    splits: dict[int, tuple[int, str]] = {}  # idx → (split_after, tail_speaker)
    for item in data:
        if not isinstance(item, dict):
            continue
        n = item.get("id")
        if not isinstance(n, int) or n not in n_to_idx:
            continue
        idx = n_to_idx[n]
        if "split_after" in item:
            k = item.get("split_after")
            tail_sid = str(item.get("tail_speaker"))
            if (isinstance(k, int) and k >= 1 and tail_sid in valid_ids
                    and tail_sid != str(out[idx].get("speaker"))):
                splits[idx] = (k, tail_sid)
            continue
        new_sid = str(item.get("speaker"))
        if new_sid not in valid_ids:
            continue
        if str(out[idx].get("speaker")) != new_sid:
            out[idx]["speaker"] = new_sid
            reassigned += 1

    split_count = 0
    if splits:
        rebuilt: list[dict] = []
        for i, seg in enumerate(out):
            if i in splits:
                k, tail_sid = splits[i]
                sentences = _split_sentences(seg.get("text", ""))
                if 1 <= k < len(sentences):
                    head = " ".join(sentences[:k]).strip()
                    tail = " ".join(sentences[k:]).strip()
                    if head and tail:
                        seg_head = {**seg, "text": head}
                        seg_tail = {**seg, "text": tail, "speaker": tail_sid}
                        # События несут start_s/end_s — точку разреза
                        # интерполируем по доле длины текста.
                        if "start_s" in seg and "end_s" in seg:
                            frac = len(head) / max(1, len(seg["text"]))
                            cut = seg["start_s"] + (seg["end_s"] - seg["start_s"]) * frac
                            seg_head["end_s"] = cut
                            seg_tail["start_s"] = cut
                        rebuilt.extend([seg_head, seg_tail])
                        split_count += 1
                        continue
            rebuilt.append(seg)
        out = rebuilt

    logger.info("Gemini-правка границ: переназначено реплик: %d, разрезано склеек: %d",
                reassigned, split_count)
    if not reassigned and not split_count:
        return segments
    return _merge_adjacent_same_speaker(out) if merge_adjacent else out


def postprocess_segments(
    segments: list[dict],
    use_gemini: bool = True,
    warnings: list[str] | None = None,
    crew_names: list[str] | None = None,
    description: str | None = None,
) -> list[dict]:
    """Постобработка сегментов: regex-чистка + опционально Gemini-полировка.

    Батчит сегменты для Gemini (~5000 символов за раз) для экономии API-вызовов.
    Если передан ``warnings``, при сбоях Gemini туда добавляется предупреждение
    (текст остаётся без AI-коррекции, но glossary/regex уже применены).
    ``crew_names`` (из паспорта) передаётся в детекцию технических моментов;
    ``description`` (тема съёмки из паспорта) — контекст для полировки.
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

    # Недоступный клиент (ключ/прокси) раньше деградировал МОЛЧА — все пассы
    # тихо превращались в no-op. Делаем сбой видимым в UI и прекращаем сразу.
    if not _gemini_ready():
        if warnings is not None:
            warnings.append(
                "Gemini недоступен (ключ/прокси — см. лог сервера): полировка, "
                "техмоменты и правка границ спикеров НЕ выполнены."
            )
        return segments

    # Консервативно сворачиваем крон-чаттер в маркер тех. момента ДО полировки.
    if TECH_MOMENT_DETECTION:
        detect_technical_segments(segments, warnings=warnings, crew_names=crew_names)

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
            polished = gemini_polish(batch_text, context=description)
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
                    polished_text = gemini_polish(segments[idx]["text"], context=description)
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
    logger.info("Gemini-полировка: %d/%d батчей успешно",
                len(batch_texts) - failed_batches, len(batch_texts))

    return segments
