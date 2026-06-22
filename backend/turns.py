"""Сборка реплик (turns) из ASR-сегментов в стиле эталонных стенограмм.

Правила выведены из четырёх эталонных DOCX:
- одна реплика спикера = один абзац;
- внутри длинной реплики раз в ~60 секунд вставляется инлайн-таймкод,
  строго после конца предложения (все 33 инлайн-таймкода эталона f7
  стоят после ". " или "? ");
- пауза в записи >= порога даёт отдельную ремарку "(Технические моменты)."
  — в том числе перед первой репликой, если речь начинается заметно позже
  стартового таймкода файла; метка спикера после ремарки повторяется;
- незавершённая (прерванная) реплика заканчивается "...".

Ограничения: сегменты без конечной пунктуации склеиваются без искусственной
точки (пунктуацию по смыслу добавляет Gemini-полировка); ремарки, вставленные
человеком при паузах меньше порога, алгоритм не воспроизводит.
"""

import re

from backend.utils import offset_tc

TECH_BREAK_TEXT = "(Технические моменты)."
# Замена неуверенных ASR-сегментов; эталоны пишут её ВНУТРИ реплики
# («...текст (неразборчиво) текст»), поэтому склеивается как обычная речь.
UNCLEAR_TEXT = "(неразборчиво)"

# Конечная пунктуация предложения — строго для инлайн-таймкодов и
# капитализации следующего сегмента. Цензус: все 71 инлайн-таймкодов
# эталонов стоят после .!?, НИКОГДА после »)". Кавычка/скобка сами по себе
# не закрывают предложение («на «Мосфильме» для...» — середина фразы).
_SENTENCE_END_PUNCT = tuple('.!?…')
# Символы, после которых реплика визуально завершена и НЕ получает "...".
# Шире, чем _SENTENCE_END_PUNCT: закрывающая кавычка/скобка завершают
# реплику типа «Это «Щука».» (точка попадает в _SENTENCE_END_PUNCT, но и
# одиночная » не должна тянуть искусственное многоточие).
_TURN_COMPLETE_CHARS = tuple('.!?…»)"')
# Хвост, отбрасываемый перед добавлением "..." прерванной реплики.
_TRAILING_TRIM = " ,;:–—-"
# Неперекрывающиеся альтернативы — линейный матчинг без бэктрекинга.
_FULL_PARENTHETICAL_RE = re.compile(r"^\((?:[^()]|\([^()]*\))*\)[.!?…]*$")
_POLITE_VY = {"Вы", "Вас", "Вам", "Вами", "Ваш", "Ваша", "Ваше", "Ваши"}

# Слова, у которых безопасно понизить регистр в СЕРЕДИНЕ склеенной реплики
# (когда предыдущий сегмент не закончил предложение): союзы, частицы,
# местоимения, наречия — практически никогда не имена собственные. ASR
# капитализирует начало каждого сегмента, и при склейке появляются ложные
# заглавные («…зрителя И заставляли…», «…больше Он уже…»). Имена собственные
# (Нонна, Москва, Простоквашино) в набор не входят и заглавную сохраняют.
# «Вы»-формы исключены намеренно — вежливое «Вы» эталоны пишут с заглавной.
_LOWERCASE_CONTINUATION = {
    "и", "а", "но", "да", "или", "что", "чтобы", "как", "так", "то",
    "он", "она", "оно", "они", "это", "эта", "этот", "эти",
    "потому", "поэтому", "который", "которая", "которое", "которые",
    "вот", "ну", "же", "бы", "ли", "не", "ни", "если", "когда",
    "тоже", "также", "там", "тут", "здесь", "уже", "ещё", "еще",
    "его", "её", "ее", "их", "им", "ему", "ей", "мы", "ты",
    "потом", "затем", "значит", "вообще", "просто",
}


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith(_SENTENCE_END_PUNCT)


def _lower_continuation(part: str) -> str:
    """Понижает регистр первой буквы продолжения реплики, если её первое слово —
    безопасное служебное слово (см. ``_LOWERCASE_CONTINUATION``), а не имя.

    Вызывается только когда предыдущий сегмент НЕ закончил предложение, то есть
    продолжение действительно идёт в середине фразы.
    """
    if not part or not part[0].isupper():
        return part
    first = part.split(maxsplit=1)[0]
    bare = first.strip(",.!?…:;»«\"'()-").lower()
    if bare in _LOWERCASE_CONTINUATION:
        return part[0].lower() + part[1:]
    return part


def _finalize_turn_text(text: str, resumed: bool = False) -> str:
    """Финализирует реплику: заглавная первая буква, "..." для прерванных.

    resumed=True — возобновление прерванной реплики того же спикера после
    тех-паузы: эталоны начинают её с «... » и строчной («М: ... организации.»).
    """
    text = text.strip()
    if not text:
        return text
    # Чисто неразборчивый сегмент: эталоны пишут «(неразборчиво).» с точкой
    # (цензус: все 6 standalone-случаев с точкой). Inline-случаи сюда не
    # попадают — они склеены в более длинную реплику.
    if text == UNCLEAR_TEXT:
        return text + "."
    if resumed:
        first_word = text.split()[0] if text.split() else ""
        if text[0].isupper() and first_word not in _POLITE_VY and not (len(text) > 1 and text[1].isalpha() and text[1].isupper()):
            text = text[0].lower() + text[1:]
        text = "... " + text
    elif text[0].islower():
        text = text[0].upper() + text[1:]
    if not text.endswith(_TURN_COMPLETE_CHARS):
        text = text.rstrip(_TRAILING_TRIM) + "..."
    return text


def build_turns(
    events: list[dict],
    start_frames: int,
    fps: int,
    *,
    inline_tc_seconds: float = 60.0,
    tech_break_gap_seconds: float = 30.0,
) -> list[dict]:
    """Склеивает события [{speaker, text, start_s, end_s}] в реплики.

    Возвращает сегменты {timecode, speaker, text}: по одному на реплику,
    плюс ремарки технических пауз. В середине склейки регистр понижается
    только у безопасных служебных слов (``_LOWERCASE_CONTINUATION``) — имена
    собственные в середине реплики заглавную сохраняют.
    """
    out: list[dict] = []
    events = sorted(events, key=lambda e: e["start_s"])
    if not events:
        return out

    def tc(seconds: float) -> str:
        return offset_tc(start_frames, seconds, fps)

    cur: dict | None = None
    # Спикер, чья реплика была прервана тех-паузой посреди предложения, —
    # его следующая реплика начинается с «... » (эталон: «М: ... организации.»)
    resume_speaker: str | None = None
    # (speaker, index_in_out) последней прерванной реплики (текст "...")
    last_unfinished: tuple[str, int] | None = None

    def close_turn() -> dict | None:
        nonlocal cur, last_unfinished
        closed = None
        if cur is not None:
            text = _finalize_turn_text(" ".join(cur["parts"]), resumed=cur.get("resumed", False))
            if text:
                closed = {
                    "timecode": tc(cur["start_s"]),
                    "speaker": cur["speaker"],
                    "text": text,
                }
                out.append(closed)
                if text.endswith("..."):
                    last_unfinished = (cur["speaker"], len(out) - 1)
            cur = None
        return closed

    # Речь начинается заметно позже стартового таймкода файла —
    # эталоны открываются ремаркой с таймкодом начала записи.
    if events[0]["start_s"] >= tech_break_gap_seconds:
        out.append({
            "timecode": offset_tc(start_frames, 0.0, fps),
            "speaker": events[0]["speaker"],
            "text": TECH_BREAK_TEXT,
        })

    prev_end = None
    for ev in events:
        # Длинная пауза закрывает реплику даже у того же спикера;
        # таймкод ремарки = конец предыдущей речи.
        if prev_end is not None and ev["start_s"] - prev_end >= tech_break_gap_seconds:
            closed = close_turn()
            out.append({
                "timecode": tc(prev_end),
                "speaker": ev["speaker"],
                "text": TECH_BREAK_TEXT,
            })
            if closed is not None and closed["text"].endswith("...") and closed["speaker"] == ev["speaker"]:
                resume_speaker = ev["speaker"]
        prev_end = ev["end_s"]

        # "(неразборчиво)" — НЕ отдельная ремарка: склеивается в реплику ниже.
        if _FULL_PARENTHETICAL_RE.match(ev["text"]) and ev["text"] != UNCLEAR_TEXT:
            close_turn()
            # Подряд идущие одинаковые ремарки не дублируем
            if not (out and out[-1]["text"] == ev["text"]):
                out.append({
                    "timecode": tc(ev["start_s"]),
                    "speaker": ev["speaker"],
                    "text": ev["text"],
                })
            continue

        if cur is not None and ev["speaker"] != cur["speaker"]:
            close_turn()

        if cur is None:
            is_resumed = resume_speaker == ev["speaker"]
            if not is_resumed and last_unfinished is not None:
                uf_speaker, uf_idx = last_unfinished
                if (uf_speaker == ev["speaker"]
                        and len(out) - uf_idx == 2
                        and out[uf_idx + 1]["text"] != TECH_BREAK_TEXT):
                    is_resumed = True
            cur = {
                "speaker": ev["speaker"],
                "start_s": ev["start_s"],
                "parts": [ev["text"]],
                "last_tc_s": ev["start_s"],
                "resumed": is_resumed,
            }
            resume_speaker = None
            continue

        # Продолжение текущей реплики.
        part = ev["text"]
        # Серия неуверенных сегментов → одно "(неразборчиво)" в реплике
        if part == UNCLEAR_TEXT and cur["parts"][-1].endswith(UNCLEAR_TEXT):
            continue
        tail_final = _ends_sentence(cur["parts"][-1])
        if tail_final and part and part[0].islower():
            part = part[0].upper() + part[1:]
        elif not tail_final:
            # Середина фразы: снимаем ложную заглавную у служебных слов,
            # которую ASR поставил в начале своего сегмента.
            part = _lower_continuation(part)
        # Инлайн-таймкод только на границе предложений; если хвост на запятой,
        # вставка откладывается до следующей границы (last_tc_s не двигаем).
        if tail_final and ev["start_s"] - cur["last_tc_s"] >= inline_tc_seconds:
            part = f"{tc(ev['start_s'])} {part}"
            cur["last_tc_s"] = ev["start_s"]
        cur["parts"].append(part)

    close_turn()
    return out
