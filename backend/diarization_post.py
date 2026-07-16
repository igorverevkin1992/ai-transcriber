"""Постобработка диаризации: определение интервьюера (АЗК) и порядок появления.

Работает на «событиях» (events) — сырых ASR-сегментах с полями
``speaker`` / ``start_s`` / ``end_s``, ещё до склейки в абзацы (``build_turns``).
"""
from __future__ import annotations

import math
import re
from collections import Counter

# Имя + отчество в именительном падеже: «Олег Александрович», «Галина Васильевна».
# Отчество — слово с патронимическим суффиксом. Падежные формы упоминаний
# («у Галины Васильевны») имеют другие окончания и сюда НЕ попадают — это
# естественно отсекает упоминания третьих лиц от обращений.
_PATRONYMIC = r"(?:ович|евич|инична|ична|евна|овна|ьич|ич)"
_VOCATIVE_RE = re.compile(
    rf"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]*{_PATRONYMIC})"
)
_SENTENCE_END = ".?!…"

# Одиночное имя-обращение («Светлана, …», «…, Ян, …»). Капитализированных
# НЕ-имён в этой позиции много — стоп-лист вводных/местоимений/междометий.
_SINGLE_NAME_RE = re.compile(r"(?<![А-ЯЁа-яё-])([А-ЯЁ][а-яё]+)(?![а-яё-])")
_NAME_STOPWORDS = {
    "господи", "боже", "ой", "ну", "вот", "ладно", "хорошо", "спасибо",
    "слушай", "слушайте", "знаете", "знаешь", "понимаете", "понимаешь",
    "скажите", "скажи", "смотрите", "смотри", "простите", "извините",
    "здравствуйте", "привет", "давай", "давайте", "подожди", "подождите",
    "он", "она", "оно", "они", "мы", "вы", "ты", "я", "это", "тут", "там",
    "мама", "папа", "мам", "пап", "сынок", "дочка", "девочки", "ребята",
    "друзья", "коллеги", "секунду", "минутку", "стоп", "всё", "все", "да",
    "нет", "конечно", "наверное", "например", "кстати", "правда", "честно",
}


# Слово сразу после «Имя, …» — маркер АППОЗИТИВА (пояснения), а не обращения:
# «Лена Николаевна, дочка Веры Ефремовны, вышла…» (ф14). Родство/роли.
_APPOSITIVE_WORDS = {
    "дочка", "дочь", "сын", "сынок", "мама", "мать", "отец", "папа",
    "жена", "муж", "брат", "сестра", "бабушка", "дедушка", "внучка", "внук",
    "тренер", "ученица", "ученик", "подруга", "друг", "коллега", "директор",
    "руководитель", "наставник", "педагог", "врач", "автор", "ведущая",
    "ведущий", "актриса", "актёр", "певица", "певец", "чемпионка", "чемпион",
}


def _is_appositive(text: str, end_pos: int) -> bool:
    """Похоже ли «Имя, слово…» на пояснение в 3-м лице, а не на обращение."""
    tail = text[end_pos:].lstrip()
    if not tail.startswith(","):
        return False
    next_word = tail[1:].strip().split()[:1]
    return bool(next_word) and next_word[0].strip(".,!?…").lower() in _APPOSITIVE_WORDS


def _find_vocatives(text: str) -> list[str]:
    """Прямые обращения «Имя Отчество», выделенные запятой (без дублей, по порядку).

    Обращение опознаём по запятой сразу ПОСЛЕ отчества («Олег Александрович, …»)
    или по запятой перед именем и концу предложения после («…, как играл, Олег
    Александрович?»). Это отсекает упоминания третьих лиц в 3-м лице («…а Галина
    Васильевна как опытный тренер нас рассудит»), из-за которых имя уходило не
    тому спикеру. Аппозитивы («Имя, дочка …, вышла») отсекаются отдельно.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _VOCATIVE_RE.finditer(text):
        after = text[m.end():].lstrip()[:1]
        before = text[:m.start()].rstrip()[-1:]
        is_address = after == "," or (before == "," and after in _SENTENCE_END)
        if not is_address or _is_appositive(text, m.end()):
            continue
        name = f"{m.group(1)} {m.group(2)}"
        if name not in seen_set:
            seen_set.add(name)
            seen.append(name)
    return seen


# Глагольные окончания: капитализированное слово в НАЧАЛЕ предложения с
# запятой после — часто глагол («Сыграем, а…», «Спрошу, пожалуй»), не имя.
_VERBLIKE_ENDINGS = ("те", "ем", "ём", "ешь", "ишь", "ет", "ит", "ут", "ют", "у", "ю")


def _find_single_name_vocatives(text: str) -> list[str]:
    """Одиночные имена-обращения, выделенные запятой (без дублей, по порядку).

    Правила позиции те же, что у «Имя Отчество»: запятая сразу ПОСЛЕ имени
    («Светлана, мы сейчас…») или запятая перед именем и конец предложения /
    запятая после («…я переехала, Ян, уже…»). Стоп-лист отсекает вводные и
    местоимения; в начале предложения дополнительно отсекаются глагольные
    формы по окончанию. Короткие усечённые формы («Ян») тоже проходят.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _SINGLE_NAME_RE.finditer(text):
        word = m.group(1)
        if word.lower() in _NAME_STOPWORDS or len(word) < 2:
            continue
        after = text[m.end():].lstrip()[:1]
        before = text[:m.start()].rstrip()[-1:]
        sentence_initial = before in ("", *_SENTENCE_END)
        is_address = (after == "," and (before == "," or sentence_initial)) or \
                     (before == "," and (after in _SENTENCE_END or after == ","))
        if not is_address or _is_appositive(text, m.end()):
            continue
        if sentence_initial and word.lower().endswith(_VERBLIKE_ENDINGS):
            continue
        if word not in seen_set:
            seen_set.add(word)
            seen.append(word)
    return seen


def _name_matches(candidate: str, occurrence: str) -> bool:
    """Совпадение имени с учётом усечённых форм («Ян» ~ «Яна», «Свет» ~ «Света»)."""
    c = candidate.lower()
    o = occurrence.lower()
    if c == o:
        return True
    shorter, longer = (c, o) if len(c) <= len(o) else (o, c)
    return len(shorter) >= 2 and longer.startswith(shorter) and len(longer) - len(shorter) <= 2


def is_direct_address(segments: list[dict], name: str, speaker_id: str) -> bool:
    """Есть ли в диалоге ПРЯМОЕ обращение по имени ``name`` к спикеру ``speaker_id``.

    Обращение = вокатив (запятая-выделение, «Имя Отчество» или одиночное имя)
    в реплике ДРУГОГО спикера, после которой ``speaker_id`` говорит следующим.
    Валидирует результат Gemini-вывода имён: имя, встречающееся только в 3-м
    лице («Лена Николаевна, дочка Веры Ефремовны, вышла…» — ф14), не проходит.
    """
    sid = str(speaker_id)
    first_word = name.split()[0] if name.split() else name
    for i, seg in enumerate(segments):
        cur = str(seg.get("speaker"))
        if cur == sid:
            continue
        text = seg.get("text", "")
        vocatives = _find_vocatives(text) + _find_single_name_vocatives(text)
        if not any(_name_matches(name, v) or _name_matches(first_word, v.split()[0])
                   for v in vocatives):
            continue
        # Адресат — следующий ОТЛИЧНЫЙ говорящий.
        for j in range(i + 1, len(segments)):
            nxt = str(segments[j].get("speaker"))
            if nxt == cur:
                continue
            if nxt == sid:
                return True
            break
    return False


def infer_name_for_speaker(segments: list[dict], speaker_id: str) -> str | None:
    """Детерминированно вывести имя спикера по обращениям к нему в диалоге.

    Самая частая (и самая полная при равенстве) форма вокатива из чужих реплик,
    после которых ``speaker_id`` отвечает. Используется для именования
    ИНТЕРВЬЮЕРА (ф14: гостья обращается «Яна, …» — эталон именует ведущую),
    которого гостевой вывод имён по построению не покрывает.
    """
    sid = str(speaker_id)
    votes: Counter[str] = Counter()
    for i, seg in enumerate(segments):
        cur = str(seg.get("speaker"))
        if cur == sid:
            continue
        text = seg.get("text", "")
        names = _find_vocatives(text) + _find_single_name_vocatives(text)
        if not names:
            continue
        for j in range(i + 1, len(segments)):
            nxt = str(segments[j].get("speaker"))
            if nxt == cur:
                continue
            if nxt == sid:
                for n in names:
                    votes[n] += 1
            break
    if not votes:
        return None
    # Сливаем усечённые формы в самую длинную («Ян» + «Яна» → «Яна»).
    merged: Counter[str] = Counter()
    for name, cnt in votes.items():
        canon = name
        for other in votes:
            if other != name and _name_matches(name, other) and len(other) > len(canon):
                canon = other
        merged[canon] += cnt
    best, _ = max(merged.items(), key=lambda kv: (kv[1], len(kv[0])))
    return best


def infer_speaker_names_by_vocative(
    segments: list[dict],
    *,
    interviewer_id: str | None,
    guest_ids: list[str],
) -> dict[str, str]:
    """Определить имя-отчество гостей по обращениям в диалоге (детерминированно).

    Если в реплике звучит ПРЯМОЕ обращение «Имя Отчество, …», имя засчитывается
    СЛЕДУЮЩЕМУ говорящему-гостю (тому, кто отвечает на обращение). Достаточно
    одного такого обращения; но если одно и то же имя оказывается главным сразу
    у нескольких спикеров (неоднозначность), его не присваиваем. Консервативно:
    при неуверенности спикер остаётся без имени.
    """
    guest_set = {str(g) for g in guest_ids}
    if not guest_set or not segments:
        return {}

    votes: dict[tuple[str, str], int] = {}
    for i, seg in enumerate(segments):
        text = seg.get("text", "")
        # «Имя Отчество» приоритетнее; одиночные имена («Светлана, …») тоже
        # засчитываются — ф14-эталон именует спикеров по имени без отчества.
        names = _find_vocatives(text) or _find_single_name_vocatives(text)
        if not names:
            continue
        cur = str(seg["speaker"])
        # Следующий ОТЛИЧНЫЙ говорящий — предполагаемый адресат обращения.
        for j in range(i + 1, len(segments)):
            nxt = str(segments[j]["speaker"])
            if nxt == cur:
                continue
            if nxt in guest_set:
                for name in names:
                    votes[(nxt, name)] = votes.get((nxt, name), 0) + 1
            break

    by_speaker: dict[str, list[tuple[str, int]]] = {}
    for (sp, name), cnt in votes.items():
        by_speaker.setdefault(sp, []).append((name, cnt))

    # Лучшее имя для каждого спикера; имя присваиваем, только если оно главное
    # ровно у одного спикера (иначе — неоднозначность, пропускаем).
    tops = {sp: max(lst, key=lambda x: x[1]) for sp, lst in by_speaker.items()}
    name_top_count = Counter(name for name, _ in tops.values())
    return {
        sp: name
        for sp, (name, cnt) in tops.items()
        if cnt >= 1 and name_top_count[name] == 1
    }


def build_speaker_sequence(events: list[dict]) -> list[str]:
    """Последовательность speaker_id подряд идущих реплик.

    События сортируются по времени начала, затем повторяющиеся подряд одинаковые
    спикеры схлопываются в один элемент (одна реплика = один элемент).
    """
    ordered = sorted(events, key=lambda e: e["start_s"])
    sequence: list[str] = []
    for ev in ordered:
        sp = str(ev["speaker"])
        if not sequence or sequence[-1] != sp:
            sequence.append(sp)
    return sequence


def first_appearance_order(events: list[dict]) -> list[str]:
    """Спикеры в порядке первого появления (по времени начала)."""
    ordered = sorted(events, key=lambda e: e["start_s"])
    seen: list[str] = []
    seen_set: set[str] = set()
    for ev in ordered:
        sp = str(ev["speaker"])
        if sp not in seen_set:
            seen_set.add(sp)
            seen.append(sp)
    return seen


def detect_interviewer(
    sequence: list[str],
    durations: dict[str, float],
    *,
    min_distinct_guests: int = 2,
    majority_ratio: float = 0.5,
    label_single_guest: bool = False,
    question_shares: dict[str, float] | None = None,
) -> str | None:
    """Определить интервьюера (АЗК) по чередованию реплик.

    Эвристика: интервьюер чередуется с РАЗНЫМИ гостями, а гость — почти только с
    интервьюером. Кандидат = спикер с наибольшим числом различных соседей в
    последовательности реплик.

    Для интервью 1-на-1 (ровно 2 спикера) чередование симметрично и не различает
    роли — там решают два НЕЗАВИСИМЫХ сигнала (``question_shares`` — доля
    событий спикера с «?»): интервьюер задаёт вопросы И говорит меньше гостя.
    Оба должны сойтись, иначе None (ф4: имя из файла уходило ведущей, а гость
    становился «Спикер 2»).

    Возвращает speaker_id интервьюера или ``None``, если уверенности нет (тогда
    вызывающий код откатывается к обычной нумерации/именам).
    """
    speakers = set(sequence) | set(durations)
    total = len(speakers)
    if total < 2:
        return None

    # Множество различных соседей в последовательности реплик.
    neighbors: dict[str, set[str]] = {s: set() for s in speakers}
    for a, b in zip(sequence, sequence[1:]):
        if a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)

    # Число реплик у каждого спикера (интервьюер обычно вставляет много коротких).
    turn_count = {s: sequence.count(s) for s in speakers}

    # Особый случай: ровно 2 спикера (интервью один на один). Только когда оба
    # сигнала согласны: кандидат заметно чаще спрашивает И говорит меньше.
    if total == 2:
        if not label_single_guest or not question_shares:
            return None
        a, b = sorted(speakers, key=lambda s: question_shares.get(s, 0.0), reverse=True)
        cand_share = question_shares.get(a, 0.0)
        other_share = question_shares.get(b, 0.0)
        if (cand_share >= 0.4
                and cand_share >= 2 * other_share
                and durations.get(a, 0.0) <= 0.9 * durations.get(b, 0.0)):
            return a
        return None

    ranked = sorted(speakers, key=lambda s: len(neighbors[s]), reverse=True)
    top = ranked[0]
    top_n = len(neighbors[top])
    second_n = len(neighbors[ranked[1]])

    # Кандидат должен чередоваться с большинством спикеров и с достаточным числом
    # разных гостей — иначе это не интервьюер.
    required = max(min_distinct_guests, math.ceil((total - 1) * majority_ratio))
    if top_n < required:
        return None

    # Однозначный лидер по числу различных соседей.
    if top_n > second_n:
        return top

    # Ничья по соседям: разрешаем по числу реплик (интервьюер активнее); при
    # полном равенстве не угадываем.
    tied = [s for s in speakers if len(neighbors[s]) == top_n]
    tied.sort(key=lambda s: turn_count[s], reverse=True)
    if turn_count[tied[0]] > turn_count[tied[1]]:
        return tied[0]
    return None
