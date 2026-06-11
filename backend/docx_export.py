import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt

from backend.turns import UNCLEAR_TEXT
from backend.utils import strip_extension

# Parenthetical remark, optionally followed by sentence punctuation that the
# human reference also italicizes (e.g. the whole "(...)." including the dot).
PARENTHETICAL_RE = re.compile("(\\((?:[^()]*|\\([^()]*\\))*\\)[.!?…]*)")

# Служебные и эпизодические говорящие не входят в легенду эталонов:
# АЗК/ГЗК (автор/голос за кадром) и безымянные метки М1, М2, М3, Ж, ММ
# (цензус f8: реплики «М1:», «Ж:» есть, в легенде — только именованные).
_LEGEND_EXCLUDE_NAME_RE = re.compile(r"^([А-ЯЁ]ЗК|[МЖ]{1,2}\d*|ЖЕНЩИНА|МУЖЧИНА)$")


def is_legend_excluded_name(name: str) -> bool:
    """True, если имя спикера — служебная метка, не входящая в легенду."""
    return bool(_LEGEND_EXCLUDE_NAME_RE.match(name.strip().upper()))


def _configure_paragraph(paragraph):
    """Применяет к абзацу выравнивание JUSTIFY, line_spacing=1.0, space_after=0."""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = paragraph.paragraph_format
    fmt.space_after = Pt(0)
    fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    fmt.line_spacing = 1.0


def _add_page_number_header(section):
    """Добавляет номер страницы в верхний колонтитул (вправо), как в эталонах."""
    header = section.header
    header.is_linked_to_previous = False
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT

    run1 = OxmlElement("w:r")
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    run1.append(fld_begin)
    p._element.append(run1)

    run2 = OxmlElement("w:r")
    instr = OxmlElement("w:instrText")
    instr.text = "PAGE   \\* MERGEFORMAT"
    run2.append(instr)
    p._element.append(run2)

    run3 = OxmlElement("w:r")
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    run3.append(fld_sep)
    p._element.append(run3)

    run4 = OxmlElement("w:r")
    t = OxmlElement("w:t")
    t.text = "1"
    run4.append(t)
    p._element.append(run4)

    run5 = OxmlElement("w:r")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run5.append(fld_end)
    p._element.append(run5)


def _add_run(paragraph, text: str, italic: bool = False):
    """Добавляет run с заданным текстом и опциональным курсивом."""
    run = paragraph.add_run(text)
    run.font.size = Pt(14)
    run._element.find(qn("w:rPr")).append(
        OxmlElement("w:szCs", {qn("w:val"): "28"})
    )
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
        is_parenthetical = part.startswith("(") and ")" in part and not part.startswith(UNCLEAR_TEXT)
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

    # Язык документа: эталоны — ru-RU; python-docx default — en-US.
    styles_el = doc.styles.element
    doc_defaults = styles_el.find(qn("w:docDefaults"))
    if doc_defaults is not None:
        rpr_default_el = doc_defaults.find(qn("w:rPrDefault"))
        if rpr_default_el is not None:
            rpr = rpr_default_el.find(qn("w:rPr"))
            if rpr is not None:
                lang = rpr.find(qn("w:lang"))
                if lang is not None:
                    lang.set(qn("w:val"), "ru-RU")

    # settings.xml: themeFontLang
    settings = doc.settings.element
    tfl = settings.find(qn("w:themeFontLang"))
    if tfl is not None:
        tfl.set(qn("w:val"), "ru-RU")

    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(3)
        section.right_margin = Cm(1.5)
        # Эталоны: расстояние до колонтитулов 449580 EMU (~1.25 см);
        # дефолт python-docx 457200 EMU (1.27 см).
        section.header_distance = 449580
        section.footer_distance = 449580
        _add_page_number_header(section)
        # cols space: эталоны 708, python-docx default 720
        cols = section._sectPr.find(qn("w:cols"))
        if cols is not None:
            cols.set(qn("w:space"), "708")

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

    # Эталоны упорядочивают легенду по ПЕРВОМУ появлению в стенограмме
    # (ф5: ИНТЕРВЬЮЕР раньше АНТИПЕНКО, хотя А говорит в 2,5 раза больше),
    # а не по длительности речи, в которой отсортирован speakers_info.
    appearance_order: list = []
    for seg in segments:
        if seg["speaker"] not in appearance_order:
            appearance_order.append(seg["speaker"])
    ordered_ids = [sid for sid in appearance_order if sid in speakers_info]
    ordered_ids += [sid for sid in speakers_info if sid not in appearance_order]

    for speaker_id in ordered_ids:
        info = speakers_info[speaker_id]
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

        # Без префикса спикера — только ЦЕЛИКОМ скобочная ремарка;
        # реплика, начинающаяся со скобки («(пауза) и потом...»), — речь.
        if PARENTHETICAL_RE.fullmatch(text) and not text.startswith(UNCLEAR_TEXT):
            _add_run(p, f"{seg['timecode']} ")
            _add_text_with_italics(p, text)
        else:
            _add_run(p, f"{seg['timecode']} {display_name}: ")
            _add_text_with_italics(p, text)

    # Эталоны заканчиваются пустым абзацем после последней реплики
    # (все 4 файла: 1-2 завершающих пустых абзаца).
    trailing = doc.add_paragraph()
    _configure_paragraph(trailing)

    doc.save(output_path)
    return download_name
