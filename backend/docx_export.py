import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Cm, Mm, Pt

from backend.utils import strip_extension

# Parenthetical remark, optionally followed by sentence punctuation that the
# human reference also italicizes (e.g. the whole "(...)." including the dot).
PARENTHETICAL_RE = re.compile("(\\((?:[^()]*|\\([^()]*\\))*\\)[.!?…]*)")


def _configure_paragraph(paragraph):
    """Применяет к абзацу выравнивание JUSTIFY, line_spacing=1.0, space_after=0."""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = paragraph.paragraph_format
    fmt.space_after = Pt(0)
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
        is_parenthetical = part.startswith("(") and ")" in part
        _add_run(paragraph, part, italic=is_parenthetical)


def generate_docx(
    project: dict,
    final_map: dict,
    abbr_map: dict,
    output_path: str,
    legend_exclude: set[str] | None = None,
    segments_override: list[dict] | None = None,
) -> str:
    """Генерирует DOCX с расшифровкой в формате, идентичном ручному."""
    doc = Document()

    # The human reference renders in Calibri (theme minor font), while
    # python-docx's default template resolves the minor font to Cambria.
    # Pin Normal to Calibri so body text and blank lines match the reference.
    # We deliberately do NOT set a Normal size: the 11pt document default is
    # kept (blank separator lines are 11pt); 14pt is applied only per run.
    doc.styles["Normal"].font.name = "Calibri"

    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(3)
        section.right_margin = Cm(1.5)

    original_filename = project.get("original_filename", "transcript")
    download_name = strip_extension(original_filename) + ".docx"

    header_para = doc.add_paragraph()
    _configure_paragraph(header_para)
    _add_run(header_para, download_name)

    empty1 = doc.add_paragraph()
    _configure_paragraph(empty1)

    speakers_info = project["result"].get("speakers", {})
    segments = segments_override if segments_override is not None else project["result"]["segments"]
    speakers_in_segments = {seg["speaker"] for seg in segments}
    exclude = legend_exclude or set()

    for speaker_id, info in speakers_info.items():
        if speaker_id not in speakers_in_segments:
            continue
        if speaker_id in exclude:
            continue
        name = final_map.get(speaker_id, info.get("suggested_name", f"Спикер {speaker_id}"))
        abbr = abbr_map.get(speaker_id, "")
        legend_text = f"{name} – {abbr}." if abbr else f"{name}."
        legend_para = doc.add_paragraph()
        _configure_paragraph(legend_para)
        _add_run(legend_para, legend_text)

    empty2 = doc.add_paragraph()
    _configure_paragraph(empty2)

    for seg in segments:
        speaker_id = seg["speaker"]
        abbr = abbr_map.get(speaker_id, "")
        display_name = abbr or final_map.get(speaker_id, f"Спикер {speaker_id}")
        text = seg["text"]

        p = doc.add_paragraph()
        _configure_paragraph(p)

        if text.startswith("("):
            _add_run(p, f"{seg['timecode']} ")
            _add_text_with_italics(p, text)
        else:
            _add_run(p, f"{seg['timecode']} {display_name}: ")
            _add_text_with_italics(p, text)

    doc.save(output_path)
    return download_name
