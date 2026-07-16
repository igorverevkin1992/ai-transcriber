"""Тесты сборки реплик (backend.turns) — структура как в эталонных стенограммах."""

from backend.turns import TECH_BREAK_TEXT, UNCLEAR_TEXT, build_turns
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


class TestFrameZeroing:
    def test_fractional_seconds_produce_zero_frames(self):
        out = _build([_ev("0", "Привет.", 2.52, 4.0)])
        assert out[0]["timecode"] == "00:00:02:00"

    def test_fractional_with_start_tc(self):
        frames = tc_to_frames("11:26:35:00", 25)
        out = _build([_ev("0", "Привет.", 2.52, 4.0)], start_frames=frames)
        assert out[0]["timecode"] == "11:26:37:00"

    def test_embedded_tc_frames_zeroed(self):
        frames = tc_to_frames("11:26:35:12", 25)
        out = _build([_ev("0", "Поехали.", 35, 40)], start_frames=frames)
        assert out[0]["timecode"] == "11:26:35:00"
        assert out[1]["timecode"] == "11:27:10:00"


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


class TestResumedTurns:
    def test_interrupted_same_speaker_resumes_with_ellipsis(self):
        # Эталон: «М: ... организации.» — прерванная реплика возобновляется
        # с «... » и строчной буквы
        out = _build([
            _ev("0", "мы начали обсуждать вопросы", 0, 5),
            _ev("0", "Организации. И поехали дальше.", 50, 55),
        ])
        assert len(out) == 3
        assert out[0]["text"] == "Мы начали обсуждать вопросы..."
        assert out[1]["text"] == TECH_BREAK_TEXT
        assert out[2]["text"] == "... организации. И поехали дальше."

    def test_other_speaker_after_break_no_ellipsis(self):
        out = _build([
            _ev("0", "мы начали обсуждать", 0, 5),
            _ev("1", "новая тема.", 50, 55),
        ])
        assert out[2]["text"] == "Новая тема."

    def test_finished_turn_no_resume_prefix(self):
        # Реплика завершена точкой — после паузы продолжение с заглавной
        out = _build([
            _ev("0", "Мы всё обсудили.", 0, 5),
            _ev("0", "теперь о другом.", 50, 55),
        ])
        assert out[2]["text"] == "Теперь о другом."

    def test_resumed_acronym_not_lowercased(self):
        out = _build([
            _ev("0", "тогда мы пошли в", 0, 5),
            _ev("0", "МХАТ на спектакль.", 50, 55),
        ])
        assert out[2]["text"] == "... МХАТ на спектакль."

    def test_resume_after_one_interruption(self):
        out = _build([
            _ev("0", "мы начали обсуждать", 0, 5),
            _ev("1", "Артефактами?", 6, 8),
            _ev("0", "с артефактами, которые уезжали.", 9, 14),
        ])
        assert out[0]["text"] == "Мы начали обсуждать..."
        assert out[1]["text"] == "Артефактами?"
        assert out[2]["text"] == "... с артефактами, которые уезжали."

    def test_no_resume_if_previous_turn_finished(self):
        out = _build([
            _ev("0", "Мы всё обсудили.", 0, 5),
            _ev("1", "Хорошо.", 6, 8),
            _ev("0", "теперь о другом.", 9, 14),
        ])
        assert out[2]["text"] == "Теперь о другом."

    def test_no_resume_if_two_intervening_turns(self):
        out = _build([
            _ev("0", "мы начали обсуждать", 0, 5),
            _ev("1", "Артефактами?", 6, 8),
            _ev("2", "Нет.", 9, 10),
            _ev("0", "с артефактами.", 11, 14),
        ])
        assert out[3]["text"] == "С артефактами."

    def test_resume_preserves_polite_vy(self):
        out = _build([
            _ev("0", "мы хотели спросить", 0, 5),
            _ev("1", "Конечно.", 6, 8),
            _ev("0", "Вы даже молодых сегодня.", 9, 14),
        ])
        assert out[2]["text"] == "... Вы даже молодых сегодня."


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

    def test_mid_sentence_join_lowercases_function_word(self):
        # ASR капитализирует начало каждого сегмента; при склейке в середине
        # фразы ложная заглавная у служебного слова снимается.
        out = _build([
            _ev("0", "переносить зрителя", 0, 2),
            _ev("0", "И заставляли верить.", 3, 5),
        ])
        assert out[0]["text"] == "Переносить зрителя и заставляли верить."

    def test_mid_sentence_join_preserves_proper_noun(self):
        out = _build([
            _ev("0", "кумиров не было, потому что", 0, 2),
            _ev("0", "Нонна Мордюкова и Гурченко.", 3, 5),
        ])
        assert out[0]["text"] == "Кумиров не было, потому что Нонна Мордюкова и Гурченко."

    def test_mid_sentence_join_preserves_polite_vy(self):
        out = _build([_ev("0", "я думаю, что", 0, 2), _ev("0", "Вы правы.", 3, 5)])
        assert out[0]["text"] == "Я думаю, что Вы правы."

    def test_whitespace_only_events_produce_no_turns(self):
        out = _build([_ev("0", " ", 0, 2), _ev("0", "  ", 3, 5)])
        assert out == []

    def test_partial_parenthetical_merged(self):
        out = _build([_ev("0", "(пауза) и потом он сказал", 0, 5)])
        assert len(out) == 1
        assert "(пауза) и потом он сказал" in out[0]["text"]

    def test_parenthetical_not_merged(self):
        out = _build([
            _ev("0", "Первая фраза.", 0, 5),
            _ev("0", "(смех)", 6, 8),
            _ev("0", "Вторая фраза.", 9, 12),
        ])
        assert len(out) == 3
        assert out[1] == {"timecode": "00:00:06:00", "speaker": "0", "text": "(смех)"}
        assert out[2]["timecode"] == "00:00:09:00"

    def test_nested_parenthetical_recognized(self):
        out = _build([
            _ev("0", "Первая фраза.", 0, 5),
            _ev("0", "(Технические моменты (перерыв)).", 6, 8),
            _ev("0", "Вторая фраза.", 9, 12),
        ])
        assert len(out) == 3
        assert out[1]["text"] == "(Технические моменты (перерыв))."

    def test_consecutive_identical_remarks_deduped(self):
        out = _build([
            _ev("0", "(звучит песня)", 0, 5),
            _ev("0", "(звучит песня)", 6, 10),
            _ev("0", "Дальше речь.", 11, 14),
        ])
        assert len(out) == 2
        assert out[0]["text"] == "(звучит песня)"
        assert out[1]["text"] == "Дальше речь."


class TestUnclearText:
    def test_unclear_merges_inline_into_turn(self):
        # Эталон ф14-15: «(неразборчиво)» — внутри реплики, не отдельный абзац
        out = _build([
            _ev("0", "мы начали говорить", 0, 3),
            _ev("0", UNCLEAR_TEXT, 4, 6),
            _ev("0", "и закончили хорошо.", 7, 10),
        ])
        assert len(out) == 1
        assert UNCLEAR_TEXT in out[0]["text"]
        assert out[0]["text"].startswith("Мы начали")

    def test_consecutive_unclear_collapsed(self):
        out = _build([
            _ev("0", "речь шла", 0, 3),
            _ev("0", UNCLEAR_TEXT, 4, 6),
            _ev("0", UNCLEAR_TEXT, 7, 9),
            _ev("0", UNCLEAR_TEXT, 10, 12),
        ])
        assert len(out) == 1
        assert out[0]["text"].count(UNCLEAR_TEXT) == 1

    def test_unclear_alone_is_own_turn(self):
        # Эталоны: standalone «(неразборчиво).» пишется с точкой (6/6 случаев)
        out = _build([_ev("0", UNCLEAR_TEXT, 0, 3)])
        assert out == [{"timecode": "00:00:00:00", "speaker": "0", "text": UNCLEAR_TEXT + "."}]

    def test_standalone_unclear_gets_period(self):
        out = _build([_ev("0", UNCLEAR_TEXT, 0, 3)])
        assert out[0]["text"] == "(неразборчиво)."

    def test_inline_unclear_no_period(self):
        # Внутри реплики «(неразборчиво)» не получает собственную точку;
        # закрывающая скобка завершает реплику без искусственного «...».
        out = _build([
            _ev("0", "какая это была", 0, 3),
            _ev("0", UNCLEAR_TEXT, 4, 6),
        ])
        assert len(out) == 1
        assert out[0]["text"] == "Какая это была (неразборчиво)"


class TestSentenceBoundary:
    def test_guillemet_ending_not_treated_as_sentence(self):
        # Сегмент, кончающийся «»» без точки — середина фразы: следующий
        # сегмент НЕ капитализируется, инлайн-таймкод НЕ вставляется.
        out = _build([
            _ev("0", "мы поехали на «Мосфильм»", 0, 59),
            _ev("0", "снимать кино.", 61, 70),
        ])
        assert len(out) == 1
        assert out[0]["text"] == "Мы поехали на «Мосфильм» снимать кино."

    def test_guillemet_with_period_is_sentence(self):
        # «...«Щука».» — точка после кавычки закрывает предложение:
        # следующий сегмент капитализируется.
        out = _build([
            _ev("0", "это «Щука».", 0, 2),
            _ev("0", "потом мы ушли.", 3, 5),
        ])
        assert out[0]["text"] == "Это «Щука». Потом мы ушли."

    def test_guillemet_ending_no_artificial_ellipsis(self):
        # Реплика, кончающаяся «»», не получает искусственное «...»
        out = _build([_ev("0", "снимать «Вечную любовь»", 0, 2)])
        assert out[0]["text"] == "Снимать «Вечную любовь»"


class TestParentheticalRegexSafety:
    def test_pathological_input_linear_time(self):
        # Регрессия ReDoS: незакрытая скобка не должна вызывать
        # катастрофический бэктрекинг (старый паттерн зависал уже на ~30 символах).
        import time

        from backend.docx_export import PARENTHETICAL_RE
        from backend.turns import _FULL_PARENTHETICAL_RE

        evil = "(" + "а" * 100_000
        start = time.monotonic()
        assert _FULL_PARENTHETICAL_RE.match(evil) is None
        assert PARENTHETICAL_RE.search(evil) is None
        assert time.monotonic() - start < 1.0

    def test_semantics_unchanged(self):
        from backend.turns import _FULL_PARENTHETICAL_RE

        assert _FULL_PARENTHETICAL_RE.match("(смех)")
        assert _FULL_PARENTHETICAL_RE.match("(Технические моменты (перерыв)).")
        assert _FULL_PARENTHETICAL_RE.match("(пауза)…")
        assert not _FULL_PARENTHETICAL_RE.match("(пауза) и потом")
        assert not _FULL_PARENTHETICAL_RE.match("обычный текст")


class TestTailTechMarker:
    def _ev(self, speaker, text, start_s, end_s):
        return {"speaker": speaker, "text": text, "start_s": start_s, "end_s": end_s}

    def test_tail_marker_when_recording_continues(self):
        from backend.turns import TECH_BREAK_TEXT, build_turns
        out = build_turns(
            [self._ev("0", "Финальная реплика.", 0.0, 5.0)],
            0, 25, tech_break_gap_seconds=30.0, total_duration_s=60.0,
        )
        assert out[-1]["text"] == TECH_BREAK_TEXT
        assert out[-1]["timecode"] == "00:00:05:00"  # конец последней речи

    def test_no_tail_marker_below_threshold(self):
        from backend.turns import TECH_BREAK_TEXT, build_turns
        out = build_turns(
            [self._ev("0", "Финальная реплика.", 0.0, 5.0)],
            0, 25, tech_break_gap_seconds=30.0, total_duration_s=20.0,
        )
        assert out[-1]["text"] != TECH_BREAK_TEXT

    def test_no_duration_no_marker(self):
        from backend.turns import TECH_BREAK_TEXT, build_turns
        out = build_turns(
            [self._ev("0", "Финальная реплика.", 0.0, 5.0)],
            0, 25, tech_break_gap_seconds=30.0,
        )
        assert all(s["text"] != TECH_BREAK_TEXT for s in out)


class TestTcFnOverride:
    def test_tc_fn_used_for_all_timecodes(self):
        from backend.turns import TECH_BREAK_TEXT, build_turns
        calls = []

        def fake_tc(seconds):
            calls.append(seconds)
            return f"TC@{seconds:.0f}"

        events = [
            {"speaker": "0", "text": "Первая.", "start_s": 40.0, "end_s": 42.0},
        ]
        out = build_turns(events, 0, 25, tech_break_gap_seconds=30.0, tc_fn=fake_tc)
        # Стартовый маркер (речь позже порога) и реплика — оба через tc_fn.
        assert out[0]["text"] == TECH_BREAK_TEXT and out[0]["timecode"] == "TC@0"
        assert out[1]["timecode"] == "TC@40"

    def test_none_tc_fn_keeps_linear(self):
        from backend.turns import build_turns
        events = [{"speaker": "0", "text": "Реплика.", "start_s": 2.0, "end_s": 3.0}]
        out = build_turns(events, 25 * 3600, 25)
        assert out[0]["timecode"] == "01:00:02:00"


class TestLeadingMarkerRestamp:
    def test_first_folded_marker_restamped_to_zero(self):
        from backend.turns import TECH_BREAK_TEXT, build_turns
        events = [
            {"speaker": "0", "text": TECH_BREAK_TEXT, "start_s": 25.0, "end_s": 26.0},
            {"speaker": "0", "text": "Первая реплика.", "start_s": 40.0, "end_s": 42.0},
        ]
        out = build_turns(events, 25 * 3600, 25, tech_break_gap_seconds=30.0)
        assert out[0]["text"] == TECH_BREAK_TEXT
        assert out[0]["timecode"] == "01:00:00:00"  # старт записи, не 01:00:25

    def test_leading_gap_marker_not_duplicated(self):
        # Первое событие-маркер ПОЗЖЕ порога: лидирующий маркер уже стоит на
        # tc(0), свёрнутый дедупится — рестамп не создаёт второго.
        from backend.turns import TECH_BREAK_TEXT, build_turns
        events = [
            {"speaker": "0", "text": TECH_BREAK_TEXT, "start_s": 45.0, "end_s": 46.0},
            {"speaker": "0", "text": "Реплика.", "start_s": 50.0, "end_s": 52.0},
        ]
        out = build_turns(events, 0, 25, tech_break_gap_seconds=30.0)
        markers = [s for s in out if s["text"] == TECH_BREAK_TEXT]
        assert len(markers) == 1
        assert markers[0]["timecode"] == "00:00:00:00"

    def test_first_speech_event_unchanged(self):
        from backend.turns import build_turns
        events = [{"speaker": "0", "text": "Реплика.", "start_s": 5.0, "end_s": 7.0}]
        out = build_turns(events, 0, 25)
        assert out[0]["timecode"] == "00:00:05:00"


class TestLeadGapSeconds:
    def test_lead_marker_with_soft_threshold(self):
        from backend.turns import TECH_BREAK_TEXT, build_turns
        events = [{"speaker": "0", "text": "Первая реплика.", "start_s": 20.0, "end_s": 22.0}]
        out = build_turns(events, 0, 25, tech_break_gap_seconds=30.0, lead_gap_seconds=15.0)
        assert out[0]["text"] == TECH_BREAK_TEXT
        assert out[0]["timecode"] == "00:00:00:00"

    def test_no_lead_marker_below_soft_threshold(self):
        from backend.turns import TECH_BREAK_TEXT, build_turns
        events = [{"speaker": "0", "text": "Первая реплика.", "start_s": 10.0, "end_s": 12.0}]
        out = build_turns(events, 0, 25, tech_break_gap_seconds=30.0, lead_gap_seconds=15.0)
        assert out[0]["text"] != TECH_BREAK_TEXT

    def test_default_keeps_gap_threshold(self):
        from backend.turns import TECH_BREAK_TEXT, build_turns
        events = [{"speaker": "0", "text": "Первая реплика.", "start_s": 20.0, "end_s": 22.0}]
        out = build_turns(events, 0, 25, tech_break_gap_seconds=30.0)
        assert out[0]["text"] != TECH_BREAK_TEXT  # 20 < 30, lead не задан
