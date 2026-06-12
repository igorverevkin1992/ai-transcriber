"""Изготавливает backend/transcript_template.docx из эталонной стенограммы.

Эталоны сделаны в русском MS Word; их пакет (styles.xml с авто-ID a..a6,
тема «Тема Office», fontTable, settings с decimalSymbol ",", header с sdt-
обёрткой PAGE-поля, footnotes/endnotes) невозможно воспроизвести вызовами
python-docx — его дефолтный шаблон выдаёт машинную природу файла.

Поэтому генератор (backend/docx_export.py) открывает не пустой Document(),
а этот шаблон: всё «тело» Word-пакета достаётся из эталона как есть, а
docx_export лишь дописывает абзацы расшифровки.

Скрипт:
1. открывает эталон f7, опустошает w:body (оставляя только sectPr);
2. вычищает авторские метаданные core.xml (их подставляет DOCX_AUTHOR при
   генерации) — чтобы коммитнутый бинарь не нёс имя живого расшифровщика;
3. обнуляет счётчики docProps/app.xml (Words/Pages/...): реальные значения
   docx_export впишет под конкретную расшифровку.

Запуск:  python scripts/make_template.py
"""

import re
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE = REPO_ROOT / "Майданов Денис, 02.02.2026_f7.docx"
TEMPLATE_OUT = REPO_ROOT / "backend" / "transcript_template.docx"

# Счётчики docProps/app.xml, которые надо обнулить в шаблоне.
_APP_COUNTERS = (
    "TotalTime", "Pages", "Words", "Characters",
    "Lines", "Paragraphs", "CharactersWithSpaces",
)


def _empty_body(doc: Document) -> None:
    """Удаляет всё содержимое w:body, кроме завершающего sectPr."""
    body = doc.element.body
    sect_pr = body.find(qn("w:sectPr"))
    for child in list(body):
        if child is not sect_pr:
            body.remove(child)


def _clean_core_properties(doc: Document) -> None:
    """Очищает авторские поля core.xml (их задаёт генератор из DOCX_AUTHOR)."""
    cp = doc.core_properties
    cp.author = ""
    cp.last_modified_by = ""
    cp.revision = 1


def _zero_app_counters(path: Path) -> None:
    """Обнуляет счётчики статистики в docProps/app.xml внутри готового пакета."""
    with zipfile.ZipFile(path) as zin:
        items = [(i, zin.read(i.filename)) for i in zin.infolist()]

    new_items = []
    for info, data in items:
        if info.filename == "docProps/app.xml":
            xml = data.decode("utf-8")
            for tag in _APP_COUNTERS:
                xml = re.sub(
                    rf"<{tag}>[^<]*</{tag}>", f"<{tag}>0</{tag}>", xml
                )
            data = xml.encode("utf-8")
        new_items.append((info, data))

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for info, data in new_items:
            zout.writestr(info, data)


def main() -> None:
    if not REFERENCE.exists():
        raise SystemExit(f"Эталон не найден: {REFERENCE}")
    doc = Document(str(REFERENCE))
    _empty_body(doc)
    _clean_core_properties(doc)
    TEMPLATE_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(TEMPLATE_OUT))
    _zero_app_counters(TEMPLATE_OUT)
    print(f"Шаблон записан: {TEMPLATE_OUT}")


if __name__ == "__main__":
    main()
