"""Тесты видео-описаний техмоментов (без реального Gemini/ffmpeg)."""

import backend.tech_vision as tv
from backend.tech_vision import (
    _apply_descriptions,
    _clean_description,
    _frame_seconds,
    _marker_intervals,
    annotate_tech_markers,
    gemini_describe_frames,
)

TM = "(Технические моменты)"


def _seg(tc, text, speaker="0"):
    return {"timecode": tc, "speaker": speaker, "text": text}


class TestMarkerIntervals:
    def test_intervals_and_short_filter(self):
        segs = [
            _seg("00:00:00:00", TM),           # 0-20s → интервал 20 c
            _seg("00:00:20:00", "Реплика."),
            _seg("00:00:30:00", TM),           # 30-31s → короче 3 c, пропуск
            _seg("00:00:31:00", "Ответ."),
            _seg("00:00:40:00", TM),           # последний → +10 c
        ]
        out = _marker_intervals(segs, 25, 0, min_gap_s=3, max_markers=30)
        assert out == [(0, 0.0, 20.0), (4, 40.0, 50.0)]

    def test_start_frames_offset(self):
        # Стартовый ТК файла 01:00:00:00 → маркер в 01:00:05:00 = 5-я секунда.
        segs = [_seg("01:00:05:00", TM), _seg("01:00:15:00", "Реплика.")]
        start_frames = 25 * 3600
        out = _marker_intervals(segs, 25, start_frames, min_gap_s=3, max_markers=30)
        assert out == [(0, 5.0, 15.0)]

    def test_max_markers_cap(self):
        segs = []
        for i in range(10):
            segs.append(_seg(f"00:0{i}:00:00", TM))
            segs.append(_seg(f"00:0{i}:30:00", "Реплика."))
        out = _marker_intervals(segs, 25, 0, min_gap_s=3, max_markers=4)
        assert len(out) == 4


class TestFrameSeconds:
    def test_midpoint_and_quarter(self):
        secs = _frame_seconds([(0, 0.0, 10.0), (2, 0.0, 40.0)])
        assert secs[0] == [5.0]
        assert secs[2] == [10.0, 20.0]  # 1/4 и середина длинного интервала


class TestCleanDescription:
    def test_normalizes(self):
        assert _clean_description(' «съемка   нарезки\nпомидоров.» ') == "Съемка нарезки помидоров"

    def test_garbage_empty(self):
        assert _clean_description("   ") == ""
        assert _clean_description("...") == ""

    def test_truncates_long(self):
        long = "съемка " + "очень " * 30 + "длинная"
        out = _clean_description(long)
        assert len(out) <= 80


class TestApplyDescriptions:
    def test_marker_enriched(self):
        segs = [_seg("00:00:00:00", TM), _seg("00:00:20:00", "Реплика.")]
        out = _apply_descriptions(segs, {0: "съемка на кухне ресторана"})
        assert out[0]["text"] == "(Технические моменты. Съемка на кухне ресторана)"
        assert out[1]["text"] == "Реплика."
        assert segs[0]["text"] == TM  # исходник не мутирован

    def test_non_marker_untouched(self):
        segs = [_seg("00:00:00:00", "Реплика.")]
        out = _apply_descriptions(segs, {0: "описание"})
        assert out[0]["text"] == "Реплика."

    def test_garbage_description_skipped(self):
        segs = [_seg("00:00:00:00", TM)]
        out = _apply_descriptions(segs, {0: "  ...  "})
        assert out[0]["text"] == TM


class TestGeminiDescribeFrames:
    def test_batched_json_parsed(self, monkeypatch):
        calls = []

        def fake_generate(contents):
            calls.append(contents)
            return '{"0": "съемка на кухне", "4": "исполнение песни"}'

        monkeypatch.setattr(tv, "_vision_generate", fake_generate)
        monkeypatch.setattr(tv, "_image_part", lambda data, mime: ("img", mime))
        marker_images = [(0, [(b"x", "image/jpeg")]), (4, [(b"y", "image/jpeg")])]
        out = gemini_describe_frames(marker_images, heroes=["Иванов"], description="кухня")
        assert out == {0: "съемка на кухне", 4: "исполнение песни"}
        assert len(calls) == 1  # один батч
        prompt = calls[0][-1]
        assert "Иванов" in prompt and "кухня" in prompt

    def test_unavailable_returns_empty(self, monkeypatch):
        monkeypatch.setattr(tv, "_vision_generate", lambda contents: None)
        monkeypatch.setattr(tv, "_image_part", lambda data, mime: data)
        assert gemini_describe_frames([(0, [(b"x", "image/jpeg")])]) == {}

    def test_invalid_ids_ignored(self, monkeypatch):
        monkeypatch.setattr(tv, "_vision_generate",
                            lambda contents: '{"0": "ок", "99": "чужой", "abc": "мусор"}')
        monkeypatch.setattr(tv, "_image_part", lambda data, mime: data)
        out = gemini_describe_frames([(0, [(b"x", "image/jpeg")])])
        assert out == {0: "ок"}


class TestAnnotateTechMarkers:
    def test_end_to_end_with_mocks(self, monkeypatch, tmp_path):
        png = tmp_path / "frame.png"
        png.write_bytes(b"fakepng")
        segs = [
            _seg("00:00:00:00", TM),
            _seg("00:00:20:00", "Реплика."),
        ]
        monkeypatch.setattr(
            "backend.timecode_ocr._extract_frames",
            lambda path, idxs, region: [(idxs[0], str(png))],
        )
        monkeypatch.setattr(tv, "_encode_jpeg", lambda p: (b"jpg", "image/jpeg"))
        monkeypatch.setattr(tv, "gemini_describe_frames",
                            lambda mi, **kw: {0: "съемка на кухне"})
        out = annotate_tech_markers(segs, "video.wmv", 25, 0)
        assert out[0]["text"] == "(Технические моменты. Съемка на кухне)"

    def test_failure_returns_original(self, monkeypatch):
        segs = [_seg("00:00:00:00", TM), _seg("00:00:20:00", "Реплика.")]

        def boom(path, idxs, region):
            raise RuntimeError("no ffmpeg")

        monkeypatch.setattr("backend.timecode_ocr._extract_frames", boom)
        out = annotate_tech_markers(segs, "video.wmv", 25, 0)
        assert out == segs

    def test_no_markers_noop(self):
        segs = [_seg("00:00:00:00", "Просто реплика.")]
        assert annotate_tech_markers(segs, "video.wmv", 25, 0) == segs


class TestEnrichedMarkerItalics:
    def test_enriched_marker_rendered_italic(self):
        # Обогащённый маркер — полная скобочная ремарка: DOCX курсивит её целиком.
        from docx import Document

        from backend.docx_export import _add_text_with_italics
        doc = Document()
        p = doc.add_paragraph()
        _add_text_with_italics(p, "(Технические моменты. Съемка на кухне ресторана)")
        runs = [(r.text, bool(r.italic)) for r in p.runs if r.text.strip()]
        assert runs and all(it for _, it in runs)


class TestPromptPolish:
    def test_prompt_prefers_syomka_style(self, monkeypatch):
        prompts = []

        def fake_generate(contents):
            prompts.append(contents[-1])
            return "{}"

        monkeypatch.setattr(tv, "_vision_generate", fake_generate)
        monkeypatch.setattr(tv, "_image_part", lambda data, mime: data)
        gemini_describe_frames([(0, [(b"x", "image/jpeg")])])
        assert "Съемка нарезки помидоров" in prompts[0]
        assert prompts[0].index("Съемка") < prompts[0].index("Исполнение")

    def test_speech_context_in_batch(self, monkeypatch):
        batches = []

        def fake_generate(contents):
            batches.append(contents)
            return "{}"

        monkeypatch.setattr(tv, "_vision_generate", fake_generate)
        monkeypatch.setattr(tv, "_image_part", lambda data, mime: data)
        gemini_describe_frames(
            [(3, [(b"x", "image/jpeg")])],
            speech_context={3: ("аллора рагацци", "продолжаем сервировку")},
        )
        joined = " ".join(s for s in batches[0] if isinstance(s, str))
        assert "аллора рагацци" in joined and "продолжаем сервировку" in joined
        assert "иностранном языке" in batches[0][-1]

    def test_speech_context_built_from_segments(self):
        segs = [
            _seg("00:00:00:00", "Хвост предыдущей реплики героя."),
            _seg("00:00:10:00", TM),
            _seg("00:00:30:00", "Начало следующей реплики."),
        ]
        before, after = tv._speech_context(segs, 1)
        assert before.endswith("героя.")
        assert after.startswith("Начало")

    def test_speech_context_skips_markers(self):
        segs = [
            _seg("00:00:00:00", "Реплика."),
            _seg("00:00:10:00", TM),
            _seg("00:00:20:00", TM),
            _seg("00:00:40:00", "Дальше."),
        ]
        before, after = tv._speech_context(segs, 2)
        assert before == "Реплика."
        assert after == "Дальше."


class TestInsideSpeechContext:
    def test_folded_texts_reach_prompt(self, monkeypatch, tmp_path):
        png = tmp_path / "frame.png"
        png.write_bytes(b"fakepng")
        segs = [
            _seg("00:00:00:00", TM),
            _seg("00:01:00:00", "Реплика после паузы."),
        ]
        batches = []

        def fake_generate(contents):
            batches.append(contents)
            return '{"0": "исполнение песни"}'

        monkeypatch.setattr(
            "backend.timecode_ocr._extract_frames",
            lambda path, idxs, region: [(idxs[0], str(png))],
        )
        monkeypatch.setattr(tv, "_encode_jpeg", lambda p: (b"jpg", "image/jpeg"))
        monkeypatch.setattr(tv, "_vision_generate", fake_generate)
        monkeypatch.setattr(tv, "_image_part", lambda data, mime: data)
        out = tv.annotate_tech_markers(
            segs, "video.wmv", 25, 0,
            folded_speech=[(10.0, "Happy birthday to you"), (30.0, "Ещё раз с поцелуем?")],
        )
        joined = " ".join(s for s in batches[0] if isinstance(s, str))
        assert "Happy birthday to you" in joined and "Ещё раз с поцелуем?" in joined
        assert "звучало" in joined
        assert out[0]["text"] == "(Технические моменты. Исполнение песни)"

    def test_no_folded_speech_no_context_line(self, monkeypatch):
        batches = []
        monkeypatch.setattr(tv, "_vision_generate", lambda c: batches.append(c) or "{}")
        monkeypatch.setattr(tv, "_image_part", lambda data, mime: data)
        gemini_describe_frames([(0, [(b"x", "image/jpeg")])])
        joined = " ".join(s for s in batches[0] if isinstance(s, str))
        assert "звучало:" not in joined
