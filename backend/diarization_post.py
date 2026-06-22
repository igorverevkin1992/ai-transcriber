"""Постобработка диаризации: определение интервьюера (АЗК) и порядок появления.

Работает на «событиях» (events) — сырых ASR-сегментах с полями
``speaker`` / ``start_s`` / ``end_s``, ещё до склейки в абзацы (``build_turns``).
"""
from __future__ import annotations

import math


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
