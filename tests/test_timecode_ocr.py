"""Tests for backend.timecode_ocr pure logic (no easyocr/models needed)."""

from backend.timecode_ocr import _parse_region, parse_start_timecode_from_ocr


class TestParseStartTimecodeFromOcr:
    def test_two_frames_agree(self):
        # Кадр 25 (1с) показывает 04:41:19:00, кадр 50 (2с) — 04:41:20:00 →
        # старт первого кадра = 04:41:18:00.
        samples = [(25, ["04:41:19:00"]), (50, ["04:41:20:00"])]
        assert parse_start_timecode_from_ocr(samples, 25) == "04:41:18:00"

    def test_garbage_text_filtered(self):
        samples = [
            (25, ["xx 04:41:19:00 xx"]),
            (50, ["99:99:99:99", "04:41:20:00"]),
        ]
        assert parse_start_timecode_from_ocr(samples, 25) == "04:41:18:00"

    def test_picks_consistent_candidate(self):
        samples = [
            (25, ["12:00:00:00", "04:41:19:00"]),
            (50, ["04:41:20:00", "99:99:99:99"]),
        ]
        assert parse_start_timecode_from_ocr(samples, 25) == "04:41:18:00"

    def test_drop_frame_separator(self):
        samples = [(25, ["04:41:19;00"]), (50, ["04:41:20;00"])]
        assert parse_start_timecode_from_ocr(samples, 25) == "04:41:18:00"

    def test_single_frame_returns_none(self):
        assert parse_start_timecode_from_ocr([(25, ["04:41:19:00"])], 25) is None

    def test_disagreement_returns_none(self):
        samples = [(25, ["04:41:19:00"]), (50, ["01:00:00:00"])]
        assert parse_start_timecode_from_ocr(samples, 25) is None

    def test_frame_field_above_fps_rejected(self):
        # f=30 невозможен при 25 fps → кандидаты отбрасываются.
        samples = [(25, ["04:41:19:30"]), (50, ["04:41:20:30"])]
        assert parse_start_timecode_from_ocr(samples, 25) is None

    def test_empty_samples(self):
        assert parse_start_timecode_from_ocr([], 25) is None


class TestParseRegion:
    def test_valid(self):
        assert _parse_region("0.55,0.82,1.0,1.0") == (0.55, 0.82, 1.0, 1.0)

    def test_empty_or_none(self):
        assert _parse_region("") is None
        assert _parse_region(None) is None

    def test_too_few_values(self):
        assert _parse_region("0.5,0.5") is None

    def test_out_of_order_rejected(self):
        assert _parse_region("0.9,0.1,0.2,0.2") is None

    def test_out_of_range_rejected(self):
        assert _parse_region("0.5,0.5,1.5,1.0") is None
