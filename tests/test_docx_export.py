from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from backend.docx_export import generate_docx


class TestGenerateDocx:
    def test_header_is_download_name_with_docx_extension(self, tmp_path, sample_project):
        out = tmp_path / "out.docx"
        generate_docx(
            sample_project,
            final_map={"0": "Денис Майданов", "1": "Григорий Антипенко"},
            abbr_map={"0": "М", "1": "А"},
            output_path=str(out),
        )
        doc = Document(str(out))
        first_para = doc.paragraphs[0]
        assert first_para.text == "test_interview.docx"
        for run in first_para.runs:
            assert not run.bold
        assert first_para.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY

    def test_legend_only_for_speakers_in_segments(self, tmp_path, sample_project):
        sample_project["result"]["speakers"]["99"] = {
            "duration_sec": 5.0, "suggested_name": "Призрак",
        }
        out = tmp_path / "out.docx"
        generate_docx(
            sample_project,
            final_map={"0": "Денис", "1": "Григорий", "99": "Призрак"},
            abbr_map={"0": "М", "1": "А", "99": "П"},
            output_path=str(out),
        )
        doc = Document(str(out))
        text = "\n".join(p.text for p in doc.paragraphs)
        assert "Призрак" not in text
        assert "Денис" in text
        assert "Григорий" in text

    def test_legend_exclude_hides_speaker_from_legend(self, tmp_path, sample_project):
        sample_project["result"]["speakers"]["2"] = {
            "duration_sec": 10.0, "suggested_name": "АЗК",
        }
        sample_project["result"]["segments"].append(
            {"timecode": "11:04:30:00", "speaker": "2", "text": "Ещё раз."}
        )
        out = tmp_path / "out.docx"
        generate_docx(
            sample_project,
            final_map={"0": "Денис", "1": "Григорий", "2": "АЗК"},
            abbr_map={"0": "М", "1": "А", "2": "АЗК"},
            output_path=str(out),
            legend_exclude={"2"},
        )
        doc = Document(str(out))
        legend_texts = [p.text for p in doc.paragraphs if "–" in p.text]
        assert not any("АЗК" in t for t in legend_texts)
        segment_texts = [p.text for p in doc.paragraphs if "11:04:30:00" in p.text]
        assert len(segment_texts) == 1
        assert "АЗК:" in segment_texts[0]

    def test_parenthetical_segment_no_speaker_prefix(self, tmp_path, sample_project):
        out = tmp_path / "out.docx"
        generate_docx(sample_project, {"0": "Д", "1": "Г"}, {"0": "М", "1": "А"}, str(out))
        doc = Document(str(out))
        tech_paras = [p for p in doc.paragraphs if "(Технические моменты.)" in p.text]
        assert len(tech_paras) == 1
        assert "М:" not in tech_paras[0].text
        assert tech_paras[0].text.startswith("11:04:15:00 (")

    def test_parenthetical_is_italic(self, tmp_path, sample_project):
        out = tmp_path / "out.docx"
        generate_docx(sample_project, {"0": "Д", "1": "Г"}, {"0": "М", "1": "А"}, str(out))
        doc = Document(str(out))
        found_italic = False
        for para in doc.paragraphs:
            for run in para.runs:
                if run.italic and "(" in run.text:
                    found_italic = True
        assert found_italic

    def test_no_page_numbers_in_footer(self, tmp_path, sample_project):
        out = tmp_path / "out.docx"
        generate_docx(sample_project, {"0": "Д", "1": "Г"}, {"0": "М", "1": "А"}, str(out))
        doc = Document(str(out))
        for section in doc.sections:
            footer_text = "".join(p.text for p in section.footer.paragraphs).strip()
            assert footer_text == ""

    def test_margins(self, tmp_path, sample_project):
        out = tmp_path / "out.docx"
        generate_docx(sample_project, {"0": "Д", "1": "Г"}, {"0": "М", "1": "А"}, str(out))
        doc = Document(str(out))
        section = doc.sections[0]
        assert abs(section.top_margin - Cm(2)) < 1000
        assert abs(section.left_margin - Cm(3)) < 1000
        assert abs(section.right_margin - Cm(1.5)) < 1000

    def test_segment_format_inline_timecode(self, tmp_path, sample_project):
        out = tmp_path / "out.docx"
        generate_docx(sample_project, {"0": "Д", "1": "Г"}, {"0": "М", "1": "А"}, str(out))
        doc = Document(str(out))
        segment_paragraphs = [p for p in doc.paragraphs if "11:04:" in p.text]
        assert len(segment_paragraphs) == 3
        for para in segment_paragraphs:
            for run in para.runs:
                if run.text.startswith("11:04:"):
                    assert run.font.size == Pt(14) or run.font.size is None

    def test_returns_correct_download_name(self, tmp_path, sample_project):
        out = tmp_path / "out.docx"
        name = generate_docx(sample_project, {"0": "Д"}, {"0": "М"}, str(out))
        assert name == "test_interview.docx"
