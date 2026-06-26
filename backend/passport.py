"""Парсинг «паспорта съёмки» (.docx).

После каждой съёмки ассистент заполняет структурированную форму в Word: описание
съёмки, кто герои (имена) и сколько их. Эти достоверные данные подаются в пайплайн
как приоритетный источник `meta` (имена гостей, число говорящих, тема-контекст для
LLM-правки границ) — точнее, чем разбор имени файла и угадывание по диалогу.

Глоссария/терминов в паспорте НЕТ (термины остаются на env `GLOSSARY_REPLACEMENTS`).

Детерминированный разбор по меткам формы (таблица или «Поле: значение»). Если
разметка не распознана — вызывающий код может прибегнуть к фолбэку через Gemini
(`gemini_extract_passport`), передав ему сырой текст (`read_passport_text`).
"""

from pathlib import Path

from backend.config import logger

# Подписи полей формы (нормализованные к нижнему регистру; матч по вхождению).
# Подгоняется под реальный шаблон паспорта.
_HEROES_LABELS = ("имена героев", "герои", "гость", "гости", "участники", "спикеры")
_COUNT_LABELS = ("количество героев", "число героев", "кол-во героев", "количество", "кол-во")
_DESC_LABELS = ("описание съёмки", "описание съемки", "описание", "синопсис", "тема", "о чём", "о чем")

# Разделители списка имён внутри одного значения.
_NAME_SPLIT = (";", ",", "\n", "•", "—", " - ")


def _norm(s: str) -> str:
    return (s or "").strip().lower().rstrip(":").strip()


def _split_names(value: str) -> list[str]:
    """Разбить значение поля «герои» на отдельные имена."""
    if not value:
        return []
    parts = [value]
    for sep in _NAME_SPLIT:
        parts = [p for chunk in parts for p in chunk.split(sep)]
    out = []
    for p in parts:
        name = p.strip(" \t.-–—•").strip()
        # Отсекаем нумерацию списков «1) Иванов», «2. Петров».
        name = name.lstrip("0123456789).").strip()
        if name and name not in out:
            out.append(name)
    return out


def _collect_fields(doc) -> dict[str, str]:
    """Собрать {нормализованная_подпись: значение} из таблиц и абзацев вида
    «Подпись: значение». Первое непустое значение выигрывает."""
    fields: dict[str, str] = {}

    def _put(label: str, value: str):
        label = _norm(label)
        value = (value or "").strip()
        if label and value and label not in fields:
            fields[label] = value

    for table in getattr(doc, "tables", []):
        for row in table.rows:
            cells = [c.text for c in row.cells]
            if len(cells) >= 2 and cells[0].strip():
                _put(cells[0], " ".join(c for c in cells[1:] if c.strip()))

    for para in doc.paragraphs:
        text = para.text
        if ":" in text:
            label, _, value = text.partition(":")
            _put(label, value)
    return fields


def _find(fields: dict[str, str], labels) -> str | None:
    """Значение первого поля, чья подпись содержит одну из меток."""
    for key, value in fields.items():
        if any(lbl in key for lbl in labels):
            return value
    return None


def read_passport_text(path) -> str:
    """Сырой текст паспорта (абзацы + ячейки таблиц) — для Gemini-фолбэка."""
    path = Path(path)
    if not path.exists():
        return ""
    try:
        from docx import Document
    except ImportError:
        return ""
    try:
        doc = Document(str(path))
    except Exception:
        return ""
    lines = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in getattr(doc, "tables", []):
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_passport(path) -> dict | None:
    """Распарсить .docx-паспорт детерминированно по меткам формы.

    Возвращает `{"speakers": [имена героев], "num_heroes": int, "description": str}`
    либо `None`, если файл не открылся / поля не распознаны (тогда вызывающий код
    может прибегнуть к Gemini-фолбэку).
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        from docx import Document
    except ImportError:
        logger.warning("python-docx недоступен — паспорт не распарсен")
        return None
    try:
        doc = Document(str(path))
    except Exception as e:
        logger.warning("Не удалось открыть паспорт %s: %s", path.name, e)
        return None

    fields = _collect_fields(doc)
    heroes = _split_names(_find(fields, _HEROES_LABELS) or "")
    desc = (_find(fields, _DESC_LABELS) or "").strip()

    num = 0
    count_raw = _find(fields, _COUNT_LABELS)
    if count_raw:
        digits = "".join(ch for ch in count_raw if ch.isdigit())
        if digits:
            num = int(digits)
    if num <= 0:
        num = len(heroes)

    if not heroes and num <= 0 and not desc:
        return None
    return {"speakers": heroes, "num_heroes": num, "description": desc}
