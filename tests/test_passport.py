"""Тесты парсинга «паспорта съёмки» (.docx)."""

from docx import Document

from backend.passport import _split_names, parse_passport, read_passport_text


def _make_table_passport(path, rows):
    doc = Document()
    table = doc.add_table(rows=0, cols=2)
    for label, value in rows:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = value
    doc.save(str(path))


def _make_para_passport(path, lines):
    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    doc.save(str(path))


class TestSplitNames:
    def test_comma_separated(self):
        assert _split_names("Олег Александрович, Галина Васильевна") == [
            "Олег Александрович", "Галина Васильевна"]

    def test_semicolon_and_newline(self):
        assert _split_names("Иванов; Петров\nСидоров") == ["Иванов", "Петров", "Сидоров"]

    def test_numbered_list(self):
        assert _split_names("1) Иванов\n2) Петров") == ["Иванов", "Петров"]

    def test_empty(self):
        assert _split_names("") == []


class TestParsePassportTable:
    def test_full_form(self, tmp_path):
        p = tmp_path / "passport.docx"
        _make_table_passport(p, [
            ("Описание съёмки", "Тренировка по бадминтону в вузе"),
            ("Герои", "Олег Александрович, Галина Васильевна"),
            ("Количество героев", "2"),
        ])
        data = parse_passport(p)
        assert data["speakers"] == ["Олег Александрович", "Галина Васильевна"]
        assert data["num_heroes"] == 2
        assert "бадминтон" in data["description"].lower()

    def test_count_from_names_when_absent(self, tmp_path):
        p = tmp_path / "p.docx"
        _make_table_passport(p, [("Гости", "Иванов; Петров; Сидоров")])
        data = parse_passport(p)
        assert data["num_heroes"] == 3
        assert len(data["speakers"]) == 3


class TestParsePassportParagraphs:
    def test_label_value_lines(self, tmp_path):
        p = tmp_path / "p.docx"
        _make_para_passport(p, [
            "Описание: интервью о кино",
            "Герои: Майданов Денис",
            "Количество героев: 1",
        ])
        data = parse_passport(p)
        assert data["speakers"] == ["Майданов Денис"]
        assert data["num_heroes"] == 1
        assert data["description"] == "интервью о кино"


class TestParsePassportEdgeCases:
    def test_missing_file(self, tmp_path):
        assert parse_passport(tmp_path / "nope.docx") is None

    def test_empty_doc_returns_none(self, tmp_path):
        p = tmp_path / "empty.docx"
        Document().save(str(p))
        assert parse_passport(p) is None

    def test_read_passport_text_collects_all(self, tmp_path):
        p = tmp_path / "p.docx"
        _make_table_passport(p, [("Герои", "Иванов")])
        text = read_passport_text(p)
        assert "Герои" in text and "Иванов" in text
