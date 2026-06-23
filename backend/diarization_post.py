"""Постобработка диаризации: определение интервьюера (АЗК) и порядок появления.

Работает на «событиях» (events) — сырых ASR-сегментах с полями
``speaker`` / ``start_s`` / ``end_s``, ещё до склейки в абзацы (``build_turns``).
"""
from __future__ import annotations

import math
import re

# Имя + отчество в именительном падеже: «Олег Александрович», «Галина Васильевна».
# Отчество — слово с патронимическим суффиксом. Падежные формы упоминаний
# («у Галины Васильевны») имеют другие окончания и сюда НЕ попадают — это
# естественно отсекает упоминания третьих лиц от обращений.
_PATRONYMIC = r"(?:ович|евич|инична|ична|евна|овна|ьич|ич)"
_VOCATIVE_RE = re.compile(
    rf"\b([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]*{_PATRONYMIC})\b"
)


def _find_vocatives(text: str) -> list[str]:
    """Все обращения вида «Имя Отчество» в тексте (без дублей, порядок сохранён)."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _VOCATIVE_RE.finditer(text):
        name = f"{m.group(1)} {m.group(2)}"
        if name not in seen_set:
            seen_set.add(name)
            seen.append(name)
    return seen


def infer_speaker_names_by_vocative(
    segments: list[dict],
    *,
    interviewer_id: str | None,
    guest_ids: list[str],
) -> dict[str, str]:
    """Определить имя-отчество гостей по обращениям в диалоге (детерминированно).

    Если в реплике звучит обращение «Имя Отчество», имя засчитывается
    СЛЕДУЮЩЕМУ говорящему-гостю (тому, кто отвечает на обращение). Имя
    присваивается спикеру при ≥2 совпадениях — это отсекает разовые упоминания.
    Консервативно: при неуверенности спикер остаётся без имени.
    """
    guest_set = {str(g) for g in guest_ids}
    if not guest_set or not segments:
        return {}

    votes: dict[tuple[str, str], int] = {}
    for i, seg in enumerate(segments):
        names = _find_vocatives(seg.get("text", ""))
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

    result: dict[str, str] = {}
    for sp, lst in by_speaker.items():
        name, cnt = max(lst, key=lambda x: x[1])
        if cnt >= 2:
            result[sp] = name
    return result


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
) -> str | None:
    """Определить интервьюера (АЗК) по чередованию реплик.

    Эвристика: интервьюер чередуется с РАЗНЫМИ гостями, а гость — почти только с
    интервьюером. Кандидат = спикер с наибольшим числом различных соседей в
    последовательности реплик.

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

    # Особый случай: ровно 2 спикера (интервью один на один) — однозначно
    # определить интервьюера нельзя.
    if total == 2:
        if not label_single_guest:
            return None
        # Помечаем АЗК того, кто говорит меньше (интервьюер обычно короче).
        return min(speakers, key=lambda s: durations.get(s, 0.0))

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
