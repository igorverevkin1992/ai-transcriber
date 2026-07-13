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

    def test_separate_digit_tokens_joined(self):
        # Крупные цифры OCR дробит на отдельные токены — склейка должна собрать ТК.
        samples = [(25, ["16", "39", "57", "11"]), (50, ["16", "39", "58", "11"])]
        assert parse_start_timecode_from_ocr(samples, 25) == "16:39:56:00"

    def test_spaced_separators(self):
        samples = [(25, ["16 : 39 : 57 : 11"]), (50, ["16 : 39 : 58 : 11"])]
        assert parse_start_timecode_from_ocr(samples, 25) == "16:39:56:00"

    def test_visible_only_from_later_frames(self):
        # ТК скрыт чёрным лидером в первые секунды, виден с 4-й — всё равно ловим.
        samples = [
            (25, []), (50, []), (75, []),
            (100, ["04:41:21:00"]), (125, ["04:41:22:00"]),
        ]
        assert parse_start_timecode_from_ocr(samples, 25) == "04:41:17:00"


class TestParseRegion:
    def test_valid(self):
        assert _parse_region("0.3,0.7,1.0,1.0") == (0.3, 0.7, 1.0, 1.0)

    def test_empty_or_none(self):
        assert _parse_region("") is None
        assert _parse_region(None) is None

    def test_too_few_values(self):
        assert _parse_region("0.5,0.5") is None

    def test_out_of_order_rejected(self):
        assert _parse_region("0.9,0.1,0.2,0.2") is None

    def test_out_of_range_rejected(self):
        assert _parse_region("0.5,0.5,1.5,1.0") is None


class TestFindTcBox:
    def _strip_with_box(self, w=960, h=240):
        cv2 = __import__("cv2")
        np = __import__("numpy")
        rng = np.random.default_rng(42)
        # Пёстрый светлый фон (кухня) + чёрная лента ТК с белыми цифрами по центру.
        img = rng.integers(120, 220, size=(h, w), dtype=np.uint8)
        x0, y0, bw, bh = 280, 150, 400, 60  # aspect ≈ 6.7
        img[y0:y0 + bh, x0:x0 + bw] = 10
        cv2.putText(img, "11:59:22:04", (x0 + 15, y0 + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 3)
        return img, (x0, y0, bw, bh)

    def test_box_found_and_covers_digits(self):
        import pytest
        pytest.importorskip("cv2")
        from backend.timecode_ocr import _find_tc_box
        img, (x0, y0, bw, bh) = self._strip_with_box()
        box = _find_tc_box(img)
        assert box is not None
        x, y, w, h = box
        # Найденный бокс покрывает ленту ТК (с учётом padding).
        assert x <= x0 and x + w >= x0 + bw
        assert y <= y0 and y + h >= y0 + bh

    def test_no_box_returns_none(self):
        import pytest
        pytest.importorskip("cv2")
        np = __import__("numpy")
        from backend.timecode_ocr import _find_tc_box
        rng = np.random.default_rng(1)
        img = rng.integers(120, 220, size=(240, 960), dtype=np.uint8)
        assert _find_tc_box(img) is None


class TestPreprocessVariants:
    def test_variants_with_box(self, tmp_path):
        import pytest
        cv2 = pytest.importorskip("cv2")
        from backend.timecode_ocr import _preprocess_variants
        img, _ = TestFindTcBox()._strip_with_box()
        p = str(tmp_path / "strip.png")
        cv2.imwrite(p, img)
        variants = _preprocess_variants(p)
        labels = [lbl for lbl, _ in variants]
        assert labels == ["v1", "v2", "v3"]
        np = __import__("numpy")
        v1 = dict(variants)["v1"]
        # v1 бинаризован: только два уровня яркости, цифры тёмные на белом.
        assert set(np.unique(v1)).issubset({0, 255})
        assert v1.mean() > 127

    def test_missing_file_falls_back(self, tmp_path):
        import pytest
        pytest.importorskip("cv2")
        from backend.timecode_ocr import _preprocess_variants
        p = str(tmp_path / "nope.png")
        variants = _preprocess_variants(p)
        assert variants == [("v3", p)]
