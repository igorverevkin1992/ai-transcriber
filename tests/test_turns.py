"""Тесты сборки реплик (backend.turns) — структура как в эталонных стенограммах."""

from backend.turns import TECH_BREAK_TEXT, build_turns
from backend.utils import tc_to_frames


def _ev(speaker, text, start_s, end_s):
    return {"speaker": speaker, "text": text, "start_s": float(start_s), "end_s": float(end_s)}


def _build(events, start_frames=0, fps=25, **kw):
    return build_turns(events, start_frames, fps, **kw)


class TestMerging:
    def test_merges_consecutive_same_speaker(self):
        out = _build([_ev("0", "Привет.", 0, 2), _ev("0", "Как дела?", 3, 5)])
        assert out == [{"timecode": "00:00:00:00", "speaker": "0", "text": "Привет. Как дела?"}]

    def test_no_merge_across_speakers(self):
        out = _build([
            _ev("0", "Раз.", 0, 2),
            _ev("1", "Два.", 2, 4),
            _ev("0", "Три.", 4, 6),
        ])
        assert [s["timecode"] for s in out] == ["00:00:00:00", "00:00:02:00", "00:00:04:00"]
        assert [s["speaker"] for s in out] == ["0", "1", "0"]

    def test_single_segment_turn_unchanged(self):
        out = _build([_ev("0", "Привет.", 1, 2)])
        assert out == [{"timecode": "00:00:01:00", "speaker": "0", "text": "Привет."}]

    def test_empty_events(self):
        assert _build([]) == []


class TestInlineTimecodes:
    def test_inline_tc_after_interval(self):
        out = _build([
            _ev("0", "А.", 0, 20),
            _ev("0", "Б.", 21, 40),
            _ev("0", "В.", 41, 62),
            _ev("0", "Г.", 63, 80),
        ])
        assert len(out) == 1
        assert out[0]["text"] == "А. Б. В. 00:01:03:00 Г."

    def test_no_inline_tc_under_interval(self):
        out = _build([_ev("0", "Начало.", 0, 30), _ev("0", "Конец.", 45, 50)])
        assert out[0]["text"] == "Начало. Конец."

    def test_inline_tc_defers_to_sentence_boundary(self):
        # Хвост на запятой — таймкод откладывается до следующей границы предложения
        out = _build([
            _ev("0", "Начало истории,", 0, 59),
            _ev("0", "которая продолжается.", 61, 70),
            _ev("0", "Конец.", 75, 80),
        ])
        assert out[0]["text"] == "Начало истории, которая продолжается. 00:01:15:00 Конец."


class TestTechBreaks:
    def test_tech_break_at_start(self):
        out = _build([_ev("0", "Поехали.", 35, 40)])
        assert out[0] == {"timecode": "00:00:00:00", "speaker": "0", "text": TECH_BREAK_TEXT}
        assert out[1] == {"timecode": "00:00:35:00", "speaker": "0", "text": "Поехали."}

    def test_tech_break_at_start_with_file_start_tc(self):
        frames = tc_to_frames("11:26:35:00", 25)
        out = _build([_ev("0", "Поехали.", 35, 40)], start_frames=frames)
        assert out[0]["timecode"] == "11:26:35:00"
        assert out[0]["text"] == TECH_BREAK_TEXT

    def test_no_tech_break_quick_start(self):
        out = _build([_ev("0", "Поехали.", 10, 12)])
        assert out[0]["text"] == "Поехали."

    def test_tech_break_mid_gap_same_speaker(self):
        # Пауза закрывает реплику даже у того же спикера; метка повторяется
        out = _build([_ev("0", "Первая часть.", 0, 10), _ev("0", "Вторая часть.", 55, 60)])
        assert len(out) == 3
        assert out[0] == {"timecode": "00:00:00:00", "speaker": "0", "text": "Первая часть."}
        assert out[1] == {"timecode": "00:00:10:00", "speaker": "0", "text": TECH_BREAK_TEXT}
        assert out[2] == {"timecode": "00:00:55:00", "speaker": "0", "text": "Вторая часть."}

    def test_tech_break_mid_gap_cross_speaker(self):
        out = _build([_ev("0", "Первая.", 0, 10), _ev("1", "Вторая.", 55, 60)])
        assert out[1]["text"] == TECH_BREAK_TEXT
        assert out[1]["speaker"] == "1"


class TestFinalization:
    def test_trailing_ellipsis_unfinished(self):
        assert _build([_ev("0", "и тогда мы решили", 0, 2)])[0]["text"] == "И тогда мы решили..."
        assert _build([_ev("0", "и тогда мы решили,", 0, 2)])[0]["text"] == "И тогда мы решили..."

    def test_finished_turns_untouched(self):
        assert _build([_ev("0", "Всё хорошо.", 0, 2)])[0]["text"] == "Всё хорошо."
        assert _build([_ev("0", "Правда?", 0, 2)])[0]["text"] == "Правда?"
        assert _build([_ev("0", "Это «Щука».", 0, 2)])[0]["text"] == "Это «Щука»."

    def test_capitalize_join_after_sentence(self):
        out = _build([_ev("0", "привет.", 0, 2), _ev("0", "как дела?", 3, 5)])
        assert out[0]["text"] == "Привет. Как дела?"

    def test_mid_sentence_join_keeps_case(self):
        out = _build([_ev("0", "мы поехали в", 0, 2), _ev("0", "Москву.", 3, 5)])
        assert out[0]["text"] == "Мы поехали в Москву."

    def test_parenthetical_not_merged(self):
        out = _build([
            _ev("0", "Первая фраза.", 0, 5),
            _ev("0", "(смех)", 6, 8),
            _ev("0", "Вторая фраза.", 9, 12),
        ])
        assert len(out) == 3
        assert out[1] == {"timecode": "00:00:06:00", "speaker": "0", "text": "(смех)"}
        assert out[2]["timecode"] == "00:00:09:00"
