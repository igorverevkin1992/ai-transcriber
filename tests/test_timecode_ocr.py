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


class TestHourModeVoting:
    def test_hour_corrected_by_majority(self):
        # v1 (бинаризация) систематически читает «15» вместо «11» и выигрывает
        # голосование стартов; v2/v3 читают «11» верно → мода часов ≥2/3 чинит час.
        samples = [
            (25, ["15:59:22:04", "11:59:22:04", "11:59:22:04"]),
            (50, ["15:59:23:04", "11:59:23:04", "11:59:23:04"]),
        ]
        # Старты: 15:59:21 (2 голоса) и 11:59:21 (2 голоса) — берётся первый по
        # сортировке... часовая мода 11 (4 из 6) исправляет итог на 11:59:21.
        assert parse_start_timecode_from_ocr(samples, 25) == "11:59:21:00"

    def test_no_correction_without_majority(self):
        # Голоса часов 50/50 (легитимный переход часа) — правка не запускается.
        samples = [
            (25, ["11:59:59:04", "12:00:00:04"]),
            (50, ["12:00:00:04", "11:59:59:04"]),
        ]
        result = parse_start_timecode_from_ocr(samples, 25)
        assert result is not None  # какой-то согласованный старт есть

    def test_all_variants_wrong_hour_stays(self):
        # Все чтения дают «15» — мода совпадает с выигравшим часом, правки нет
        # (systematic misread всех вариантов этим механизмом не лечится).
        samples = [(25, ["15:59:22:04"]), (50, ["15:59:23:04"])]
        assert parse_start_timecode_from_ocr(samples, 25) == "15:59:21:00"


class TestColonMisreadRecovery:
    def test_real_f13_log_recovers_true_hour(self):
        # Реальные чтения f13: двоеточие после «11» прочитано как «5»
        # («11:59» → «11559»), сырой разбор давал 15:59:20. Восстановление
        # удалением одной цифры должно вернуть 11:59:20.
        samples = [
            (25, ["179 212"]),
            (50, ["159 22  2"]),
            (75, ["11559 28 22"]),
            (100, ["11559 24 22"]),
            (125, ["1150 25 22"]),
            (150, ["119 20722"]),
            (175, ["11559 27 22"]),
            (200, ["11559 28 22"]),
            (225, ["11759 29 22"]),
            (250, ["1159  30 22"]),
            (300, ["11559  32722"]),
            (375, ["11559  35  22"]),
        ]
        assert parse_start_timecode_from_ocr(samples, 25) == "11:59:20:00"

    def test_clean_readings_unaffected(self):
        # Чистые чтения (без лишних цифр) работают как раньше.
        samples = [(25, ["04:41:19:00"]), (50, ["04:41:20:00"])]
        assert parse_start_timecode_from_ocr(samples, 25) == "04:41:18:00"


class TestFindTcBoxRobustness:
    def test_blurred_box_found(self):
        import pytest
        cv2 = pytest.importorskip("cv2")
        from backend.timecode_ocr import _find_tc_box
        img, (x0, y0, bw, bh) = TestFindTcBox()._strip_with_box()
        img = cv2.GaussianBlur(img, (7, 7), 2.0)  # мыльный исходник
        box = _find_tc_box(img)
        assert box is not None
        x, y, w, h = box
        assert x <= x0 + 10 and x + w >= x0 + bw - 10
        assert y <= y0 + 5 and y + h >= y0 + bh - 5

    def test_dark_distractor_does_not_win(self):
        # Тёмная тень слева (высокая, не «ленточная») не должна перебить бокс ТК.
        import pytest
        pytest.importorskip("cv2")
        from backend.timecode_ocr import _find_tc_box
        img, (x0, y0, bw, bh) = TestFindTcBox()._strip_with_box()
        img[0:240, 0:100] = 20  # вертикальная тень: aspect << 2.5
        box = _find_tc_box(img)
        assert box is not None
        x, y, w, h = box
        assert x <= x0 and x + w >= x0 + bw  # найден именно бокс ТК


class TestReadTcAt:
    def test_two_of_three_agree(self, monkeypatch):
        import backend.timecode_ocr as tco
        monkeypatch.setattr(tco, "_extract_frame_at", lambda vp, sec, reg: f"/fake/{sec}.png")
        # Кадры k=0,1,2: ТК растёт с кадром; третий — мисрид.
        base = 6 * 3600 + 50 * 60 + 0  # 06:50:00
        readings = {0: ["06:50:00:10"], 1: ["06:50:01:10"], 2: ["99:99:99:99"]}
        monkeypatch.setattr(tco, "_ocr_frames",
                            lambda reader, pairs: [(k, readings[k]) for k, _ in pairs])
        monkeypatch.setattr(tco.Path, "unlink", lambda self: None)
        assert tco._read_tc_at(None, "v.wmv", 120.0, 25, None) == base

    def test_disagreement_returns_none(self, monkeypatch):
        import backend.timecode_ocr as tco
        monkeypatch.setattr(tco, "_extract_frame_at", lambda vp, sec, reg: f"/fake/{sec}.png")
        readings = {0: ["06:50:00:10"], 1: ["07:10:00:10"], 2: ["05:00:00:10"]}
        monkeypatch.setattr(tco, "_ocr_frames",
                            lambda reader, pairs: [(k, readings[k]) for k, _ in pairs])
        monkeypatch.setattr(tco.Path, "unlink", lambda self: None)
        assert tco._read_tc_at(None, "v.wmv", 120.0, 25, None) is None


class TestReadTcAnchors:
    def _patch(self, monkeypatch, tc_by_second):
        import backend.timecode_ocr as tco
        monkeypatch.setattr(tco, "_get_reader", lambda: object())
        monkeypatch.setattr(
            tco, "_read_tc_at",
            lambda reader, vp, sec, fps, reg: tc_by_second(sec),
        )
        return tco

    def test_anchors_include_start_and_readings(self, monkeypatch):
        start = 6 * 3600 + 48 * 60  # 06:48:00
        # ТК = старт + медиа (без дрейфа)
        tco = self._patch(monkeypatch, lambda sec: start + round(sec))
        anchors = tco.read_tc_anchors("v.wmv", 25, 400.0, start_tc_s=start, interval_s=120)
        assert anchors[0] == (0.0, start)
        assert (120.0, start + 120) in anchors and (240.0, start + 240) in anchors

    def test_monotonicity_drops_misread(self, monkeypatch):
        start = 1000
        def fake(sec):
            if sec == 120:
                return start + 60  # offset -60: ТК «назад» — мисрид
            return start + round(sec) + 10
        tco = self._patch(monkeypatch, fake)
        anchors = tco.read_tc_anchors("v.wmv", 25, 400.0, start_tc_s=start, interval_s=120)
        assert (120.0, start + 180) not in anchors
        assert all(t - m >= start - 1 for m, t in anchors)

    def test_bisection_localizes_jump(self, monkeypatch):
        start = 1000
        # Скачок ТК +15 c на медиа-секунде 100: до неё offset 0, после +15.
        def fake(sec):
            return start + round(sec) + (15 if sec >= 100 else 0)
        tco = self._patch(monkeypatch, fake)
        anchors = tco.read_tc_anchors("v.wmv", 25, 300.0, start_tc_s=start,
                                      interval_s=120, bisect_depth=4)
        # Бисекция добавила точки между 0 и 120 — скачок локализован точнее 120 c.
        mids = [m for m, _ in anchors if 0 < m < 120]
        assert mids, f"нет бисекционных якорей: {anchors}"
        # Есть якорь с offset 0 ниже 100 и якорь с offset 15 в (100, 120).
        assert any(m < 100 and t - m == start for m, t in anchors)
        assert any(100 <= m < 120 and t - m == start + 15 for m, t in anchors)
