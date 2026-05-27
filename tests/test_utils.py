"""Unit tests for backend utility functions."""

from backend.utils import (
    frames_to_tc,
    parse_filename_metadata,
    sanitize_filename,
    strip_extension,
    tc_to_frames,
    validate_file_extension,
    validate_url,
)


class TestFramesToTc:
    def test_zero(self):
        assert frames_to_tc(0) == "00:00:00:00"

    def test_one_second(self):
        assert frames_to_tc(25) == "00:00:01:00"

    def test_one_minute(self):
        assert frames_to_tc(25 * 60) == "00:01:00:00"

    def test_one_hour(self):
        assert frames_to_tc(25 * 3600) == "01:00:00:00"

    def test_mixed(self):
        # 1h 23m 45s 12f
        frames = (1 * 3600 + 23 * 60 + 45) * 25 + 12
        assert frames_to_tc(frames) == "01:23:45:12"

    def test_custom_fps_30(self):
        assert frames_to_tc(30, fps=30) == "00:00:01:00"

    def test_frame_wrap(self):
        assert frames_to_tc(24) == "00:00:00:24"
        assert frames_to_tc(25) == "00:00:01:00"


class TestTcToFrames:
    def test_zero(self):
        assert tc_to_frames("00:00:00:00") == 0

    def test_one_second(self):
        assert tc_to_frames("00:00:01:00") == 25

    def test_complex(self):
        assert tc_to_frames("01:23:45:12") == (1 * 3600 + 23 * 60 + 45) * 25 + 12

    def test_invalid_returns_zero(self):
        assert tc_to_frames("invalid") == 0
        assert tc_to_frames("") == 0

    def test_roundtrip(self):
        for f in [0, 1, 24, 25, 100, 1000, 90000]:
            assert tc_to_frames(frames_to_tc(f)) == f


class TestParseFilenameMetadata:
    def test_simple_names(self):
        result = parse_filename_metadata("Довлатова Алла, Павел, 05.11.2025_f8.mp3")
        assert "Довлатова Алла" in result["speakers"]
        assert "Павел" in result["speakers"]
        assert result["start_tc"] == "00:00:00:00"

    def test_with_timecode(self):
        result = parse_filename_metadata("15:40:41:00_test.wav")
        assert result["start_tc"] == "15:40:41:00"

    def test_filters_stop_words(self):
        result = parse_filename_metadata("Имя_лайф_f8.mp4")
        assert "Имя" in result["speakers"]
        assert "лайф" not in result["speakers"]
        assert "f8" not in result["speakers"]

    def test_filters_dates(self):
        result = parse_filename_metadata("Имя, 05.11.2025_f8.mp3")
        dates = [s for s in result["speakers"] if "2025" in s]
        assert len(dates) == 0

    def test_no_names(self):
        result = parse_filename_metadata("f8.mp3")
        assert result["speakers"] == []

    def test_filters_russian_file_codes(self):
        result = parse_filename_metadata("2026.02.03_Антипенко_ф5.mp4")
        assert result["speakers"] == ["Антипенко"]

    def test_filters_f_codes_any_number(self):
        result = parse_filename_metadata("02.02.2026_f7.mp4")
        assert result["speakers"] == []

    def test_filters_multi_file_codes(self):
        result = parse_filename_metadata("Гайнутдинов_2026.01.24_ф14-15.mp4")
        assert result["speakers"] == ["Гайнутдинов"]

    def test_filters_yyyy_mm_dd_dates(self):
        result = parse_filename_metadata("2026.02.03_Антипенко.mp4")
        assert result["speakers"] == ["Антипенко"]

    def test_filters_pure_digits(self):
        result = parse_filename_metadata("Имя_15.mp4")
        assert result["speakers"] == ["Имя"]


class TestStripExtension:
    def test_basic(self):
        assert strip_extension("file.mp3") == "file"

    def test_multiple_dots(self):
        assert strip_extension("my.file.name.docx") == "my.file.name"

    def test_no_extension(self):
        assert strip_extension("noext") == "noext"


class TestValidateUrl:
    def test_valid_yadi_sk(self):
        assert validate_url("https://yadi.sk/d/abc123") is None

    def test_valid_disk_yandex(self):
        assert validate_url("https://disk.yandex.ru/d/abc123") is None

    def test_invalid_host(self):
        error = validate_url("https://evil.com/file")
        assert error is not None
        assert "Яндекс.Диск" in error

    def test_invalid_scheme(self):
        error = validate_url("ftp://yadi.sk/d/abc")
        assert error is not None

    def test_invalid_url(self):
        error = validate_url("not a url at all")
        assert error is not None


class TestSanitizeFilename:
    def test_normal_filename(self):
        assert sanitize_filename("video.mp4") == "video.mp4"

    def test_path_traversal(self):
        assert sanitize_filename("../../etc/passwd") == "passwd"

    def test_backslash_stripped(self):
        result = sanitize_filename("..\\..\\windows\\system32\\cmd.exe")
        assert "\\" not in result
        assert "/" not in result

    def test_null_byte(self):
        assert sanitize_filename("file\x00.mp4") == "file.mp4"

    def test_leading_dots_stripped(self):
        assert sanitize_filename("...hidden") == "hidden"

    def test_empty_becomes_unnamed(self):
        assert sanitize_filename("") == "unnamed_file"
        assert sanitize_filename("...") == "unnamed_file"

    def test_base_dir_check(self, tmp_path):
        safe = sanitize_filename("video.mp4", base_dir=str(tmp_path))
        assert safe == "video.mp4"

    def test_base_dir_escape_blocked(self, tmp_path):
        result = sanitize_filename("../escape.txt", base_dir=str(tmp_path))
        assert result == "escape.txt"


class TestValidateFileExtension:
    def test_valid_extensions(self):
        for ext in [".mp3", ".wav", ".mov", ".mxf", ".mp4", ".wmv"]:
            assert validate_file_extension(f"file{ext}") is None

    def test_invalid_extension(self):
        error = validate_file_extension("file.zip")
        assert error is not None

    def test_case_insensitive(self):
        assert validate_file_extension("FILE.MP3") is None
