from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from backend.utils import strip_extension


def generate_docx(project: dict, final_map: dict, abbr_map: dict, output_path: str) -> str:
    """Генерирует DOCX с расшифровкой. Возвращает имя файла для скачивания."""
    doc = Document()

    # Настройка стилей
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Times New Roman"
    font.size = Pt(12)

    # Настройка полей страницы
    for section in doc.sections:
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
        section.left_margin = Cm(3)
        section.right_margin = Cm(1.5)

    original_filename = project.get("original_filename", "transcript")

    # Заголовок
    header_para = doc.add_paragraph()
    header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    header_run = header_para.add_run(f"ИСХОДНИК: {original_filename}")
    header_run.bold = True
    header_run.font.size = Pt(14)
    doc.add_paragraph()

    # Список спикеров с аббревиатурами
    speakers_info = project["result"].get("speakers", {})
    speakers_para = doc.add_paragraph()
    speakers_para.add_run("УЧАСТНИКИ:").bold = True
    for speaker_id, info in speakers_info.items():
        name = final_map.get(speaker_id, info.get("suggested_name", f"Спикер {speaker_id}"))
        abbr = abbr_map.get(speaker_id, "")
        line = f"\n{name.upper()}"
        if abbr:
            line += f" ({abbr})"
        speakers_para.add_run(line)
    doc.add_paragraph()

    # Разделительная линия
    separator = doc.add_paragraph()
    separator.alignment = WD_ALIGN_PARAGRAPH.CENTER
    separator.add_run("— " * 20)
    doc.add_paragraph()

    # Сегменты расшифровки
    segments = project["result"]["segments"]
    for seg in segments:
        speaker_name = final_map.get(seg["speaker"], f"Спикер {seg['speaker']}")
        abbr = abbr_map.get(seg["speaker"], "")
        display_name = abbr if abbr else speaker_name

        p = doc.add_paragraph()
        tc_run = p.add_run(f"{seg['timecode']} ")
        tc_run.font.size = Pt(10)
        tc_run.font.color.rgb = None

        name_run = p.add_run(f"{display_name}: ")
        name_run.bold = True

        p.add_run(seg["text"])

    # Номера страниц
    for section in doc.sections:
        footer = section.footer
        footer.is_linked_to_previous = False
        footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.save(output_path)

    download_name = strip_extension(original_filename) + ".docx"
    return download_name
