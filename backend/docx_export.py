import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Cm, Pt

from backend.utils import strip_extension


PARENTHETICAL_RE = re.compile(r"(\([^)]*\))")


def _configure_paragraph(paragraph):
    """Применяет к абзацу выравнивание JUSTIFY, line_spacing=1.0, space_after=0."""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = paragraph.paragraph_format
    fmt.space_after = Pt(0)
    fmt.space_before = Pt(0)
    fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    fmt.line_spacing = 1.0


def _add_run(paragraph, text: str, italic: bool = False):
    """Добавляет run с заданным текстом и опциональным курсивом."""
    run = paragraph.add_run(text)
    run.font.size = Pt(14)
    if italic:
        run.italic = True
    return run


def _add_text_with_italics(paragraph, text: str):
    """Добавляет текст в абзац, выделяя курсивом фрагменты в скобках (...)."""
    if not text:
        return
    parts = PARENTHETICAL_RE.split(text)
    for part in parts:
        if not part:
            continue
        is_parenthetical = part.startswith("(") and part.endswith(")")
        _add_run(paragraph, part, italic=is_parenthetical)


def generate_docx(project: dict, final_map: dict, abbr_map: dict, output_path: str) -> str:
    """Генерирует DOCX с расшифровкой в формате, идентичном ручному.

    Формат:
        <original_filename>
        <пустая строка>
        <Имя Фамилия> – <АББР>.
        ...
        <пустая строка>
        <таймкод> <АББР>: <текст>
        ...

    Все абзацы: 14pt, JUSTIFY, line_spacing=1.0, space_after=0.
    Фрагменты в (...) — курсивом.
    """
    doc = Document()

    style = doc.styles["Normal"]
    style.font.size = Pt(14)

    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(3)
        section.right_margin = Cm(1.5)

    original_filename = project.get("original_filename", "transcript")

    header_para = doc.add_paragraph()
    _configure_paragraph(header_para)
    _add_run(header_para, original_filename)

    empty1 = doc.add_paragraph()
    _configure_paragraph(empty1)

    speakers_info = project["result"].get("speakers", {})
    for speaker_id, info in speakers_info.items():
        name = final_map.get(speaker_id, info.get("suggested_name", f"Спикер {speaker_id}"))
        abbr = abbr_map.get(speaker_id, "")
        legend_text = f"{name} – {abbr}." if abbr else f"{name}."
        legend_para = doc.add_paragraph()
        _configure_paragraph(legend_para)
        _add_run(legend_para, legend_text)

    empty2 = doc.add_paragraph()
    _configure_paragraph(empty2)

    segments = project["result"]["segments"]
    for seg in segments:
        speaker_id = seg["speaker"]
        abbr = abbr_map.get(speaker_id, "")
        display_name = abbr or final_map.get(speaker_id, f"Спикер {speaker_id}")

        p = doc.add_paragraph()
        _configure_paragraph(p)
        _add_run(p, f"{seg['timecode']} {display_name}: ")
        _add_text_with_italics(p, seg["text"])

    doc.save(output_path)

    download_name = strip_extension(original_filename) + ".docx"
    return download_name
