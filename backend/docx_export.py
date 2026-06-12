import re

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt

from backend.config import DOCX_AUTHOR, DOCX_TEMPLATE_PATH
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


def _set_paragraph_mark_rpr(paragraph):
    """Свойства знака абзаца (14pt, cstheme) — как у всех абзацев эталона.

    В эталонах даже ПУСТЫЕ абзацы-разделители несут rPr знака абзаца с
    sz=28: без него высота пустой строки падает до 11pt и вертикальная
    раскладка/разбиение на страницы расходятся с ручным оригиналом.
    """
    p_pr = paragraph._p.get_or_add_pPr()
    rpr = p_pr.find(qn("w:rPr"))
    if rpr is None:
        # rPr знака абзаца идёт последним среди свойств абзаца (после jc).
        rpr = OxmlElement("w:rPr")
        p_pr.append(rpr)
    rpr.append(OxmlElement("w:rFonts", {qn("w:cstheme"): "minorHAnsi"}))
    rpr.append(OxmlElement("w:sz", {qn("w:val"): "28"}))
    rpr.append(OxmlElement("w:szCs", {qn("w:val"): "28"}))


def _configure_paragraph(paragraph):
    """Применяет к абзацу выравнивание JUSTIFY, line_spacing=1.0, space_after=0."""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    fmt = paragraph.paragraph_format
    fmt.space_after = Pt(0)
    fmt.line_spacing_rule = WD_LINE_SPACING.SINGLE
    fmt.line_spacing = 1.0
    _set_paragraph_mark_rpr(paragraph)


def _add_run(paragraph, text: str, italic: bool = False):
    """Добавляет run с заданным текстом и опциональным курсивом."""
    run = paragraph.add_run(text)
    run.font.size = Pt(14)
    rpr = run._element.find(qn("w:rPr"))
    rpr.append(OxmlElement("w:szCs", {qn("w:val"): "28"}))
    rpr.insert(0, OxmlElement("w:rFonts", {qn("w:cstheme"): "minorHAnsi"}))
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
    """Генерирует DOCX с расшифровкой в формате, идентичном ручному.

    Документ строится из backend/transcript_template.docx — пакета,
    извлечённого из эталонной стенограммы (styles/theme/fontTable/settings,
    header с номером страницы, footnotes/endnotes). Это снимает все отпечатки
    дефолтного шаблона python-docx разом; здесь дописываются только абзацы.
    """
    doc = Document(str(DOCX_TEMPLATE_PATH))

    doc.core_properties.author = DOCX_AUTHOR
    doc.core_properties.last_modified_by = DOCX_AUTHOR

    # Размер страницы, поля и расстояние до колонтитулов уже заданы в sectPr
    # шаблона (A4, поля 2/3/2/1.5 см, header/footer 708 twips). Дублируем явно
    # как страховку на случай смены шаблона — значения совпадают, диффа нет.
    for section in doc.sections:
        section.page_width = Mm(210)
        section.page_height = Mm(297)
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(3)
        section.right_margin = Cm(1.5)
        section.header_distance = 449580
        section.footer_distance = 449580

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
