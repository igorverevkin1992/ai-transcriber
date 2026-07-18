import backend.postprocess as pp
from backend.postprocess import (
    GeminiPolishError,
    correct_speaker_boundaries,
    gemini_extract_passport,
    gemini_infer_speaker_names,
    postprocess_segments,
    regex_cleanup,
)
from backend.turns import TECH_BREAK_TEXT


class TestGeminiInferSpeakerNames:
    @staticmethod
    def _seg(sp, text):
        return {"timecode": "00:00:00:00", "speaker": sp, "text": text}

    def test_parses_and_validates(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt, **kw: '{"1": "Олег Александрович", "2": "Галина Васильевна", "0": null}',
        )
        segs = [self._seg("0", "Вопрос."), self._seg("1", "Ответ."), self._seg("2", "Ответ2.")]
        res = gemini_infer_speaker_names(segs, interviewer_id="0", guest_ids=["1", "2"])
        assert res == {"1": "Олег Александрович", "2": "Галина Васильевна"}

    def test_rejects_non_patronymic(self, monkeypatch):
        # «Олег» — одиночное имя, теперь ПРИНИМАЕТСЯ (ф14: «Яна»/«Светлана» без
        # отчества; точность страхует валидация прямого обращения в services).
        # «Иванов Иван» — второе слово не отчество → по-прежнему отброшено.
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: '{"1": "Олег", "2": "Иванов Иван"}')
        segs = [self._seg("1", "a"), self._seg("2", "b")]
        assert gemini_infer_speaker_names(segs, interviewer_id=None, guest_ids=["1", "2"]) == {"1": "Олег"}

    def test_only_guest_ids(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt, **kw: '{"0": "Олег Александрович", "1": "Галина Васильевна"}',
        )
        segs = [self._seg("0", "a"), self._seg("1", "b")]
        res = gemini_infer_speaker_names(segs, interviewer_id="0", guest_ids=["1"])
        assert res == {"1": "Галина Васильевна"}  # "0" не в guest_ids

    def test_none_on_gemini_failure(self, monkeypatch):
        def boom(prompt, **kw):
            raise GeminiPolishError("fail")
        monkeypatch.setattr(pp, "_gemini_call", boom)
        assert gemini_infer_speaker_names([self._seg("1", "a")], interviewer_id=None, guest_ids=["1"]) is None

    def test_none_when_gemini_unavailable(self, monkeypatch):
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: None)
        assert gemini_infer_speaker_names([self._seg("1", "a")], interviewer_id=None, guest_ids=["1"]) is None

    def test_extracts_json_with_preamble(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt, **kw: 'Вот результат: {"1": "Олег Александрович"} — готово',
        )
        res = gemini_infer_speaker_names([self._seg("1", "a")], interviewer_id=None, guest_ids=["1"])
        assert res == {"1": "Олег Александрович"}

    def test_no_guests_returns_none(self):
        assert gemini_infer_speaker_names([self._seg("0", "a")], interviewer_id="0", guest_ids=[]) is None


class TestRegexCleanup:
    def test_removes_hesitation_sounds(self):
        assert regex_cleanup("эээ давай") == "давай"
        assert regex_cleanup("мммм да") == "да"

    def test_removes_hyphenated_hesitations(self):
        # Whisper часто пишет мычание через дефис: «э-э», «м-м-м»
        assert regex_cleanup("э-э, ну вот") == "ну вот"
        assert regex_cleanup("м-м-м, наверное") == "наверное"
        assert regex_cleanup("а-а-а, понял") == "понял"

    def test_keeps_hm_as_meaningful_reply(self):
        # Эталон ф5 сохраняет «Хм…» как осмысленную реплику
        assert regex_cleanup("хм понятно") == "хм понятно"

    def test_preserves_discourse_words(self):
        # Эталонные стенограммы почти дословные: ну/вот/как бы/короче сохраняются
        assert regex_cleanup("ну вот хорошо") == "ну вот хорошо"
        assert regex_cleanup("это как бы важно") == "это как бы важно"
        assert regex_cleanup("короче говоря, мы поехали") == "короче говоря, мы поехали"
        assert regex_cleanup("типа того") == "типа того"

    def test_collapses_triple_repeated_words(self):
        assert regex_cleanup("слово слово слово") == "слово"

    def test_preserves_double_repeated_words(self):
        assert regex_cleanup("привет привет мир") == "привет привет мир"
        assert regex_cleanup("очень очень важно") == "очень очень важно"

    def test_preserves_kak_by_to_ni_bylo(self):
        assert "как бы то ни было" in regex_cleanup("как бы то ни было")

    def test_capitalizes_after_punctuation(self):
        assert regex_cleanup("привет. как дела") == "привет. Как дела"
        assert regex_cleanup("да? хорошо") == "да? Хорошо"

    def test_no_capitalize_after_ellipsis(self):
        assert regex_cleanup("Мне это было... это была другая история") == \
            "Мне это было... это была другая история"

    def test_no_capitalize_after_abbreviation(self):
        # Однобуквенные сокращения и единицы измерения — строчная сохраняется
        assert regex_cleanup("г. москву") == "г. москву"
        assert regex_cleanup("5 тыс. рублей в месяц") == "5 тыс. рублей в месяц"
        assert regex_cleanup("в 90-х гг. началось") == "в 90-х гг. началось"

    def test_capitalize_after_short_words(self):
        # Эталоны: после «да.», «нет.», «так.», «вот.» — всегда заглавная (89/90)
        assert regex_cleanup("да. потом мы поехали") == "да. Потом мы поехали"
        assert regex_cleanup("нет. но мы решили") == "нет. Но мы решили"
        assert regex_cleanup("так. и вот тогда") == "так. И вот тогда"
        assert regex_cleanup("вот. и началось") == "вот. И началось"

    def test_abbreviations_expanded(self):
        # Эталоны никогда не пишут «т.е./т.д./т.п.» — всегда полные слова
        assert regex_cleanup("т.е. мы поехали") == "то есть мы поехали"
        assert regex_cleanup("Т.е. мы поехали") == "То есть мы поехали"
        assert regex_cleanup("вещи и т.д. собрали") == "вещи и так далее собрали"
        assert regex_cleanup("вещи и т.п. собрали") == "вещи и тому подобное собрали"
        assert regex_cleanup("т. е. мы") == "то есть мы"

    def test_number_repetition_preserved(self):
        assert regex_cleanup("2024 2024 2024 год") == "2024 2024 2024 год"

    def test_repeated_phrase_collapsed(self):
        # Whisper-петля: фраза повторяется 3+ раз подряд
        assert regex_cleanup("и мы пошли и мы пошли и мы пошли") == "и мы пошли"
        assert regex_cleanup("он сказал он сказал он сказал он сказал да") == "он сказал да"

    def test_double_phrase_preserved(self):
        # Двойной повтор — легитимная речь («мы шли, мы шли»)
        assert regex_cleanup("мы шли мы шли и пришли") == "мы шли мы шли и пришли"

    def test_hyphenated_repeats_preserved(self):
        # Эталоны сохраняют «Да-да-да»
        assert regex_cleanup("Да-да-да, был этот") == "Да-да-да, был этот"

    def test_capitalize_after_ya(self):
        # «я» — не сокращение: после «я.» эталоны капитализируют
        assert regex_cleanup("это был я. потом мы пошли") == "это был я. Потом мы пошли"

    def test_no_double_punctuation_after_filler_removal(self):
        # В эталонах «,,» и «,.» — 0 случаев
        assert regex_cleanup("ну, эээ, давай") == "ну, давай"
        assert regex_cleanup("я думаю, эээ. что мы") == "я думаю. Что мы"
        assert regex_cleanup("слушай, эээ, ммм, погоди") == "слушай, погоди"

    def test_preserves_first_letter_case(self):
        # ASR-сегмент может начинаться с середины предложения; заглавную букву
        # реплике ставит backend.turns при склейке.
        assert regex_cleanup("привет мир") == "привет мир"
        assert regex_cleanup("Привет мир") == "Привет мир"

    def test_empty_string(self):
        assert regex_cleanup("") == ""

    def test_clean_text_unchanged(self):
        assert regex_cleanup("Привет, мир.") == "Привет, мир."

    def test_fixes_hyphen_spacing(self):
        assert regex_cleanup("что -то") == "что-то"
        assert regex_cleanup("какой -нибудь") == "какой-нибудь"
        assert regex_cleanup("из -за") == "из-за"

    def test_multi_space_collapse(self):
        assert regex_cleanup("слово    другое") == "слово другое"


class TestTypography:
    def test_spaced_hyphen_becomes_dash(self):
        assert regex_cleanup("мы пришли - и началось") == "мы пришли – и началось"

    def test_double_hyphen_becomes_dash(self):
        assert regex_cleanup("слово -- другое") == "слово – другое"

    def test_em_dash_normalized_to_en_dash(self):
        assert regex_cleanup("театр — это другие деньги") == "театр – это другие деньги"

    def test_en_dash_kept(self):
        assert regex_cleanup("здоровье – это главное") == "здоровье – это главное"

    def test_hyphenated_particles_not_dashed(self):
        assert regex_cleanup("как-то раз") == "как-то раз"
        # частица фиксится ДО обработки тире
        assert regex_cleanup("он что -то сказал - и ушёл") == "он что-то сказал – и ушёл"

    def test_ellipsis_char_to_three_dots(self):
        # Эталоны используют "..." (451 случай против 74 "…")
        assert regex_cleanup("подожди…") == "подожди..."

    def test_many_dots_collapse(self):
        assert regex_cleanup("подожди.....") == "подожди..."

    def test_three_dots_unchanged(self):
        assert regex_cleanup("подожди...") == "подожди..."

    def test_ascii_quotes_to_guillemets(self):
        assert regex_cleanup('передача "Время"') == "передача «Время»"

    def test_low9_quotes_to_guillemets(self):
        assert regex_cleanup('передача „Время"') == "передача «Время»"
        assert regex_cleanup('передача „Время"') == "передача «Время»"

    def test_curly_quotes_to_guillemets(self):
        assert regex_cleanup('передача “Время”') == "передача «Время»"

    def test_no_quotes_unchanged(self):
        assert regex_cleanup("передача Время") == "передача Время"

    def test_space_before_punctuation_removed(self):
        assert regex_cleanup("слово , другое") == "слово, другое"
        assert regex_cleanup("конец .") == "конец."

    def test_leading_comma_after_filler_removal(self):
        assert regex_cleanup("эээ, давай") == "давай"

    def test_triple_dash_becomes_en_dash(self):
        assert regex_cleanup("слово --- другое") == "слово – другое"

    def test_spaceless_em_dash_normalized(self):
        assert regex_cleanup("слово—другое") == "слово – другое"
        assert regex_cleanup("слово–другое") == "слово – другое"

    def test_digit_ranges_become_hyphen(self):
        # Эталоны: диапазоны только дефисом без пробелов («49-50», «2002-2003»)
        assert regex_cleanup("1941–1945 годы") == "1941-1945 годы"
        assert regex_cleanup("2002—2003 год") == "2002-2003 год"
        assert regex_cleanup("20 – 30 лет") == "20-30 лет"
        assert regex_cleanup("песен 20-25 за вечер") == "песен 20-25 за вечер"

    def test_spaceless_hyphen_not_dashed(self):
        assert regex_cleanup("как-то раз") == "как-то раз"
        assert regex_cleanup("кое-что") == "кое-что"

    def test_combined_typography(self):
        result = regex_cleanup('он сказал - смотрите "Время"…')
        assert result == "он сказал – смотрите «Время»..."


class TestCleanGeminiResponse:
    def test_plain_text_unchanged(self):
        from backend.postprocess import _clean_gemini_response
        assert _clean_gemini_response("ну вот мы и поехали") == "ну вот мы и поехали"

    def test_markdown_fence_stripped(self):
        from backend.postprocess import _clean_gemini_response
        assert _clean_gemini_response("```\nтекст реплики\n```") == "текст реплики"
        assert _clean_gemini_response("```text\nтекст\n```") == "текст"

    def test_preamble_stripped(self):
        from backend.postprocess import _clean_gemini_response
        assert _clean_gemini_response("Вот исправленный текст: мы поехали") == "мы поехали"
        assert _clean_gemini_response("Исправленный текст:\nмы поехали") == "мы поехали"

    def test_speech_starting_with_vot_kept(self):
        from backend.postprocess import _clean_gemini_response
        # «вот» как разговорное слово не должно срезаться
        assert _clean_gemini_response("вот мы и поехали") == "вот мы и поехали"


class TestPostprocessSegments:
    def test_applies_regex_to_all_segments(self):
        segs = [
            {"text": "эээ привет", "channel_tag": 0},
            {"text": "ну вот как дела", "channel_tag": 1},
        ]
        result = postprocess_segments(segs, use_gemini=False)
        assert result[0]["text"] == "привет"
        assert result[1]["text"] == "ну вот как дела"

    def test_preserves_metadata(self):
        segs = [{"text": "эээ да", "channel_tag": "SPEAKER_00", "start_ms": 100}]
        result = postprocess_segments(segs, use_gemini=False)
        assert result[0]["channel_tag"] == "SPEAKER_00"
        assert result[0]["start_ms"] == 100

    def test_empty_list(self):
        assert postprocess_segments([], use_gemini=False) == []

    def test_filler_words_removed_from_timing(self):
        segs = [{
            "text": "эээ привет",
            "channel_tag": 0,
            "words": [
                {"text": "эээ", "start_ms": 1000, "end_ms": 1500},
                {"text": "привет", "start_ms": 2000, "end_ms": 2500},
            ],
        }]
        result = postprocess_segments(segs, use_gemini=False)
        assert result[0]["text"] == "привет"
        assert result[0]["words"][0]["text"] == "привет"
        assert result[0]["words"][0]["start_ms"] == 2000

    def test_all_filler_words_fallback(self):
        segs = [{
            "text": "эээ ммм",
            "channel_tag": 0,
            "words": [
                {"text": "эээ", "start_ms": 1000, "end_ms": 1500},
                {"text": "ммм", "start_ms": 2000, "end_ms": 2500},
            ],
        }]
        result = postprocess_segments(segs, use_gemini=False)
        assert result == []


class TestGeminiPrompt:
    def test_gemini_prompt_preserves_hm(self):
        import unittest.mock as mock

        from backend.postprocess import gemini_polish

        fake_response = mock.MagicMock()
        fake_response.text = "хм понятно"

        fake_client = mock.MagicMock()
        fake_client.models.generate_content.return_value = fake_response

        import backend.postprocess as pp
        original_client = pp._gemini_client
        pp._gemini_client = fake_client
        try:
            gemini_polish("хм понятно")
            prompt_text = fake_client.models.generate_content.call_args.kwargs["contents"]
            assert "«хм»" in prompt_text
            assert "сохраняй" in prompt_text.split("«хм»")[1][:50]
            assert "Убирай" not in prompt_text or "«хм»" not in prompt_text.split("Убирай")[1][:50]
        finally:
            pp._gemini_client = original_client


class TestAbbreviationCapitalization:
    def test_abbreviation_expansion_lowercases_next_word(self):
        assert regex_cleanup("и т.д. Потом мы пошли") == "и так далее потом мы пошли"

    def test_abbreviation_expansion_keeps_acronym_after(self):
        assert regex_cleanup("и т.д. МХАТ продолжил") == "и так далее МХАТ продолжил"


class TestTechMomentPromptSelection:
    def _run_and_capture_prompt(self, monkeypatch, aggressive):
        captured = {}

        def fake_call(prompt, **kw):
            captured["prompt"] = prompt
            return "НЕТ"

        monkeypatch.setattr(pp, "_gemini_call", fake_call)
        monkeypatch.setattr(pp, "TECH_MOMENT_AGGRESSIVE", aggressive)
        pp.detect_technical_segments([{"text": "Олег Александрович, расскажите про бадминтон."}])
        return captured["prompt"]

    def test_conservative_default_prompt(self, monkeypatch):
        prompt = self._run_and_capture_prompt(monkeypatch, False)
        assert "при ЛЮБОМ сомнении НЕ помечай" in prompt
        assert "пойду умоюсь" not in prompt

    def test_aggressive_prompt_when_enabled(self, monkeypatch):
        prompt = self._run_and_capture_prompt(monkeypatch, True)
        assert "организация процесса" in prompt
        assert "пойду умоюсь" in prompt


class TestTechBreakDot:
    def test_marker_respects_dot_config(self):
        # Дефолт — с точкой (как f7/f8)
        from backend.turns import TECH_BREAK_TEXT
        assert TECH_BREAK_TEXT == "(Технические моменты)."


class TestGlossaryReplacements:
    def test_word_replacement(self, monkeypatch):
        import backend.postprocess as pp
        pairs = pp._parse_glossary_replacements(
            "Мурдюкова=>Мордюкова,Горченко=>Гурченко,прияды=>плеяда"
        )
        monkeypatch.setattr(pp, "GLOSSARY_REPLACEMENT_PAIRS", pairs)
        assert pp.apply_glossary_replacements("люблю Мурдюкова и Горченко") == "люблю Мордюкова и Гурченко"
        assert pp.apply_glossary_replacements("огромные прияды") == "огромные плеяда"

    def test_phrase_replacement_tolerates_whitespace(self, monkeypatch):
        import backend.postprocess as pp
        pairs = pp._parse_glossary_replacements("просто квашено=>Простоквашино")
        monkeypatch.setattr(pp, "GLOSSARY_REPLACEMENT_PAIRS", pairs)
        assert pp.apply_glossary_replacements("это просто  квашено сегодня") == "это Простоквашино сегодня"

    def test_case_insensitive_match(self, monkeypatch):
        import backend.postprocess as pp
        pairs = pp._parse_glossary_replacements("старполити=>star quality")
        monkeypatch.setattr(pp, "GLOSSARY_REPLACEMENT_PAIRS", pairs)
        assert pp.apply_glossary_replacements("назвать Старполити") == "назвать star quality"

    def test_malformed_entries_skipped(self):
        import backend.postprocess as pp
        pairs = pp._parse_glossary_replacements("badentry,a=>b,=>x,y=>")
        assert len(pairs) == 1

    def test_runs_in_regex_cleanup_without_gemini(self, monkeypatch):
        import backend.postprocess as pp
        pairs = pp._parse_glossary_replacements("Мурдюкова=>Мордюкова")
        monkeypatch.setattr(pp, "GLOSSARY_REPLACEMENT_PAIRS", pairs)
        monkeypatch.setattr(pp, "GEMINI_API_KEY", "")
        out = postprocess_segments([{"text": "вот Мурдюкова", "words": []}], use_gemini=False)
        assert out[0]["text"] == "вот Мордюкова"


class TestGeminiFailureWarning:
    def test_failure_emits_warning_and_keeps_text(self, monkeypatch):
        import backend.postprocess as pp
        monkeypatch.setattr(pp, "GEMINI_API_KEY", "fake-key")
        monkeypatch.setattr(pp, "GLOSSARY_REPLACEMENT_PAIRS", [])
        monkeypatch.setattr(pp, "TECH_MOMENT_DETECTION", False)

        def boom(text, **kw):
            raise pp.GeminiPolishError("rate limit")

        monkeypatch.setattr(pp, "gemini_polish", boom)
        warnings = []
        out = postprocess_segments([{"text": "привет", "words": []}], warnings=warnings)
        assert out[0]["text"] == "привет"
        assert any("Gemini" in w for w in warnings)


class TestTechnicalMoments:
    def test_marks_flagged_segments(self, monkeypatch):
        import backend.postprocess as pp
        from backend.turns import TECH_BREAK_TEXT
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: "2")
        segs = [
            {"text": "Вопрос про кино?"},
            {"text": "Камера, ещё дубль, отойди от микрофона."},
            {"text": "Ответ героя."},
        ]
        out = pp.detect_technical_segments(segs, warnings=[])
        assert out[1]["text"] == TECH_BREAK_TEXT
        assert out[0]["text"] == "Вопрос про кино?"
        assert out[2]["text"] == "Ответ героя."

    def test_conservative_none_no_change(self, monkeypatch):
        import backend.postprocess as pp
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: "НЕТ")
        segs = [{"text": "А."}, {"text": "Б."}]
        out = pp.detect_technical_segments(segs)
        assert [s["text"] for s in out] == ["А.", "Б."]

    def test_gemini_failure_warns_and_keeps(self, monkeypatch):
        import backend.postprocess as pp

        def boom(prompt, **kw):
            raise pp.GeminiPolishError("503")

        monkeypatch.setattr(pp, "_gemini_call", boom)
        segs = [{"text": "Реальная реплика."}]
        warnings = []
        out = pp.detect_technical_segments(segs, warnings=warnings)
        assert out[0]["text"] == "Реальная реплика."
        assert any("технических моментов" in w for w in warnings)

    def test_skips_existing_remarks_in_numbering(self, monkeypatch):
        import backend.postprocess as pp
        from backend.turns import TECH_BREAK_TEXT, UNCLEAR_TEXT
        seen = {}

        def fake(prompt, **kw):
            seen["prompt"] = prompt
            return "НЕТ"

        monkeypatch.setattr(pp, "_gemini_call", fake)
        segs = [{"text": UNCLEAR_TEXT}, {"text": TECH_BREAK_TEXT}, {"text": "Речь."}]
        pp.detect_technical_segments(segs)
        # В нумерацию попадает только речевой фрагмент, не ремарки.
        assert "1. Речь." in seen["prompt"]
        assert UNCLEAR_TEXT not in seen["prompt"]

    def test_disabled_flag_skips_detection(self, monkeypatch):
        import backend.postprocess as pp
        monkeypatch.setattr(pp, "GEMINI_API_KEY", "fake-key")
        monkeypatch.setattr(pp, "TECH_MOMENT_DETECTION", False)
        monkeypatch.setattr(pp, "GLOSSARY_REPLACEMENT_PAIRS", [])
        called = {"n": 0}

        def fake(prompt, **kw):
            called["n"] += 1
            return "1"

        monkeypatch.setattr(pp, "_gemini_call", fake)
        monkeypatch.setattr(pp, "gemini_polish", lambda t, **kw: t)
        out = postprocess_segments([{"text": "Камера дубль.", "words": []}], warnings=[])
        assert out[0]["text"] == "Камера дубль."
        assert called["n"] == 0

    def test_postprocess_marks_and_skips_polish(self, monkeypatch):
        import backend.postprocess as pp
        from backend.turns import TECH_BREAK_TEXT
        monkeypatch.setattr(pp, "GEMINI_API_KEY", "fake-key")
        monkeypatch.setattr(pp, "TECH_MOMENT_DETECTION", True)
        monkeypatch.setattr(pp, "GLOSSARY_REPLACEMENT_PAIRS", [])
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: "1")
        polished = []

        def fake_polish(text, **kw):
            polished.append(text)
            return text + " [P]"

        monkeypatch.setattr(pp, "gemini_polish", fake_polish)
        out = postprocess_segments([{"text": "Камера, дубль.", "words": []}], warnings=[])
        assert out[0]["text"] == TECH_BREAK_TEXT
        assert polished == []  # маркер тех. момента не полируется


class TestCorrectSpeakerBoundaries:
    LABELS = {"0": "АЗК", "1": "Олег Александрович"}

    def _qa(self):
        return [
            {"timecode": "00:00:01:00", "speaker": "0", "text": "Вопрос?"},
            {"timecode": "00:00:05:00", "speaker": "1", "text": "Ответ."},
        ]

    def test_reassigns_and_merges(self, monkeypatch):
        segs = [
            {"timecode": "00:00:01:00", "speaker": "0", "text": "Меня занесло случайно."},
            {"timecode": "00:00:05:00", "speaker": "1", "text": "Я занимался борьбой."},
        ]
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: '[{"id": 0, "speaker": "1"}]')
        out = correct_speaker_boundaries(segs, speaker_labels=self.LABELS, interviewer_id="0")
        assert len(out) == 1
        assert out[0]["speaker"] == "1"
        assert out[0]["text"] == "Меня занесло случайно. Я занимался борьбой."
        assert out[0]["timecode"] == "00:00:01:00"

    def test_no_change_on_empty_corrections(self, monkeypatch):
        segs = self._qa()
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: "[]")
        assert correct_speaker_boundaries(segs, speaker_labels=self.LABELS, interviewer_id="0") == segs

    def test_unchanged_when_gemini_unavailable(self, monkeypatch):
        segs = self._qa()
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: None)
        assert correct_speaker_boundaries(segs, speaker_labels=self.LABELS, interviewer_id="0") == segs

    def test_unchanged_on_gemini_error(self, monkeypatch):
        segs = self._qa()

        def boom(prompt, **kw):
            raise GeminiPolishError("x")

        monkeypatch.setattr(pp, "_gemini_call", boom)
        assert correct_speaker_boundaries(segs, speaker_labels=self.LABELS, interviewer_id="0") == segs

    def test_ignores_invalid_speaker_id(self, monkeypatch):
        segs = self._qa()
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: '[{"id": 0, "speaker": "9"}]')
        assert correct_speaker_boundaries(segs, speaker_labels=self.LABELS, interviewer_id="0") == segs

    def test_skips_tech_markers(self, monkeypatch):
        segs = [
            {"timecode": "00:00:01:00", "speaker": "1", "text": TECH_BREAK_TEXT},
            {"timecode": "00:00:05:00", "speaker": "0", "text": "Вопрос?"},
            {"timecode": "00:00:09:00", "speaker": "1", "text": "Ответ."},
        ]
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: '[{"id": 0, "speaker": "1"}]')
        out = correct_speaker_boundaries(segs, speaker_labels=self.LABELS, interviewer_id="0")
        assert out[0]["text"] == TECH_BREAK_TEXT
        assert any(s["text"] == "Вопрос? Ответ." and s["speaker"] == "1" for s in out)


class TestGeminiExtractPassport:
    def test_parses_json(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt, **kw: '{"heroes": ["Иванов", "Петров"], "num_heroes": 2, "description": "интервью"}',
        )
        data = gemini_extract_passport("любой текст паспорта")
        assert data["speakers"] == ["Иванов", "Петров"]
        assert data["num_heroes"] == 2
        assert data["description"] == "интервью"

    def test_num_from_heroes_when_missing(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt, **kw: '{"heroes": ["А", "Б", "В"], "num_heroes": 0, "description": ""}',
        )
        assert gemini_extract_passport("t")["num_heroes"] == 3

    def test_none_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: None)
        assert gemini_extract_passport("t") is None

    def test_none_on_empty_text(self):
        assert gemini_extract_passport("") is None

    def test_none_on_error(self, monkeypatch):
        def boom(prompt, **kw):
            raise GeminiPolishError("x")

        monkeypatch.setattr(pp, "_gemini_call", boom)
        assert gemini_extract_passport("t") is None


class TestPassportCrewAndHost:
    def test_extracts_host_and_crew(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda p, **kw: '{"heroes":["Иванов"],"num_heroes":1,"host":true,"crew":["Оператор Петя"],"description":"тема"}',
        )
        data = gemini_extract_passport("t")
        assert data["speakers"] == ["Иванов"]
        assert data["crew"] == ["Оператор Петя"]
        assert data["has_host"] is True

    def test_host_false(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda p, **kw: '{"heroes":["Иванов"],"num_heroes":1,"host":false,"crew":[],"description":""}',
        )
        assert gemini_extract_passport("t")["has_host"] is False

    def test_crew_names_injected_into_tech_prompt(self, monkeypatch):
        captured = {}

        def fake_call(prompt, **kw):
            captured["prompt"] = prompt
            return ""  # ничего не помечаем

        monkeypatch.setattr(pp, "_gemini_call", fake_call)
        segs = [{"text": "Обычная реплика интервью.", "words": []}]
        pp.detect_technical_segments(segs, crew_names=["Иван Петров"])
        assert "Иван Петров" in captured["prompt"]


class TestHyphenAndParticipantLoops:
    def test_hyphen_loop_collapsed(self):
        text = "Оп" + "-оп" * 60 + "!"
        assert regex_cleanup(text) == "Оп-оп-оп!"

    def test_triple_hyphen_repeat_preserved(self):
        assert regex_cleanup("Да-да-да, был этот") == "Да-да-да, был этот"

    def test_participant_loop_removed(self):
        out = regex_cleanup("Участник 2. Участник 3. Участник 4. Участник 5.")
        assert "Участник" not in out

    def test_single_participant_preserved(self):
        assert "участник" in regex_cleanup("наш участник 2 пришёл")


class TestGeminiVisibility:
    def test_postprocess_warns_when_client_unavailable(self, monkeypatch):
        monkeypatch.setattr(pp, "GEMINI_API_KEY", "key")
        monkeypatch.setattr(pp, "_gemini_ready", lambda: False)
        warnings = []
        segs = [{"text": "Привет.", "words": []}]
        out = postprocess_segments(segs, warnings=warnings)
        assert out[0]["text"] == "Привет."
        assert any("недоступен" in w for w in warnings)

    def test_boundary_correction_warns_when_unavailable(self, monkeypatch):
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: None)
        warnings = []
        segs = [
            {"timecode": "00:00:01:00", "speaker": "0", "text": "Вопрос?"},
            {"timecode": "00:00:05:00", "speaker": "1", "text": "Ответ."},
        ]
        out = correct_speaker_boundaries(
            segs, speaker_labels={"0": "АЗК", "1": "Гость"},
            interviewer_id="0", warnings=warnings,
        )
        assert out == segs
        assert any("не выполнена" in w for w in warnings)

    def test_health_check_ok(self, monkeypatch):
        monkeypatch.setattr(pp, "_gemini_ready", lambda: True)
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: "да")
        ok, reason = pp.gemini_health_check()
        assert ok is True and reason is None

    def test_health_check_client_missing(self, monkeypatch):
        monkeypatch.setattr(pp, "_gemini_ready", lambda: False)
        ok, reason = pp.gemini_health_check()
        assert ok is False and "клиент" in reason

    def test_health_check_api_error(self, monkeypatch):
        monkeypatch.setattr(pp, "_gemini_ready", lambda: True)

        def boom(prompt, **kw):
            raise GeminiPolishError("403 forbidden")

        monkeypatch.setattr(pp, "_gemini_call", boom)
        ok, reason = pp.gemini_health_check()
        assert ok is False and "403" in reason


class TestSmartModelFallback:
    class _Resp:
        text = "ок"

    class _FakeModels:
        def __init__(self, bad="bad-pro"):
            self.bad = bad
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append(model)
            if model == self.bad:
                raise Exception(
                    "404 NOT_FOUND. models/bad-pro is not found for API version v1beta")
            return TestSmartModelFallback._Resp()

        def list(self):
            class _M:
                def __init__(self, name):
                    self.name = name
            return [_M("models/gemini-3.5-flash"), _M("models/gemini-2.5-pro")]

    def _setup(self, monkeypatch):
        client = type("C", (), {})()
        client.models = self._FakeModels()
        monkeypatch.setattr(pp, "_gemini_client", client)
        monkeypatch.setattr(pp, "_gemini_ready", lambda: True)
        monkeypatch.setattr(pp, "_broken_models", set())
        return client

    def test_404_falls_back_to_default_model(self, monkeypatch):
        client = self._setup(monkeypatch)
        result = pp._gemini_call("привет", model="bad-pro")
        assert result == "ок"
        assert client.models.calls == ["bad-pro", pp.GEMINI_MODEL]
        assert "bad-pro" in pp._broken_models

    def test_broken_model_skipped_on_next_call(self, monkeypatch):
        client = self._setup(monkeypatch)
        pp._broken_models.add("bad-pro")
        result = pp._gemini_call("привет", model="bad-pro")
        assert result == "ок"
        assert client.models.calls == [pp.GEMINI_MODEL]  # 404 не повторяется

    def test_404_on_default_model_still_raises(self, monkeypatch):
        client = self._setup(monkeypatch)
        client.models.bad = pp.GEMINI_MODEL
        import pytest
        with pytest.raises(GeminiPolishError):
            pp._gemini_call("привет")  # база сама не существует — честный сбой
        assert pp.GEMINI_MODEL not in pp._broken_models

    def test_health_check_smart_404_still_ok(self, monkeypatch):
        self._setup(monkeypatch)
        monkeypatch.setattr(pp, "GEMINI_MODEL_SMART", "bad-pro")
        ok, reason = pp.gemini_health_check()
        assert ok is True and reason is None
        assert "bad-pro" in pp._broken_models


class TestBoundaryCorrectionMergeParam:
    LABELS = {"0": "АЗК", "1": "Гость"}

    def test_no_merge_when_disabled(self, monkeypatch):
        # События до склейки: переназначение есть, но соседние одно-спикерные
        # НЕ склеиваются (этим займётся build_turns).
        segs = [
            {"speaker": "0", "text": "Первое.", "start_s": 1.0, "end_s": 2.0},
            {"speaker": "1", "text": "Второе.", "start_s": 3.0, "end_s": 4.0},
            {"speaker": "0", "text": "Третье.", "start_s": 5.0, "end_s": 6.0},
        ]
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt, **kw: '[{"id": 1, "speaker": "0"}]')
        out = correct_speaker_boundaries(
            segs, speaker_labels=self.LABELS, interviewer_id="0", merge_adjacent=False,
        )
        assert len(out) == 3  # не склеено
        assert [s["speaker"] for s in out] == ["0", "0", "0"]
        assert out[1]["start_s"] == 3.0  # поля события сохранены


class TestBoundarySplitInsideSegment:
    LABELS = {"0": "АФ", "1": "МД"}

    def test_split_after_sentence(self, monkeypatch):
        segs = [
            {"speaker": "1", "text": "Буррата должна быть тёплой. Кстати, вы говорили про моцареллу.",
             "start_s": 10.0, "end_s": 20.0},
            {"speaker": "0", "text": "Оказывается, никто не знает об этом.",
             "start_s": 21.0, "end_s": 24.0},
        ]
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt, **kw: '[{"id": 0, "split_after": 1, "tail_speaker": "0"}]',
        )
        out = correct_speaker_boundaries(
            segs, speaker_labels=self.LABELS, interviewer_id=None, merge_adjacent=False,
        )
        assert len(out) == 3
        assert out[0]["speaker"] == "1"
        assert out[0]["text"] == "Буррата должна быть тёплой."
        assert out[1]["speaker"] == "0"
        assert out[1]["text"] == "Кстати, вы говорили про моцареллу."
        # Тайминги монотонны: разрез интерполирован внутри исходного события.
        assert out[0]["start_s"] == 10.0
        assert out[0]["end_s"] == out[1]["start_s"]
        assert 10.0 < out[1]["start_s"] < 20.0
        assert out[1]["end_s"] == 20.0

    def test_invalid_split_ignored(self, monkeypatch):
        segs = [
            {"speaker": "1", "text": "Одно предложение.", "start_s": 1.0, "end_s": 2.0},
            {"speaker": "0", "text": "Другое.", "start_s": 3.0, "end_s": 4.0},
        ]
        # split_after=5 при одном предложении и tail_speaker вне valid_ids.
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt, **kw: '[{"id": 0, "split_after": 5, "tail_speaker": "0"}, '
                                 '{"id": 1, "split_after": 1, "tail_speaker": "9"}]',
        )
        out = correct_speaker_boundaries(
            segs, speaker_labels=self.LABELS, interviewer_id=None, merge_adjacent=False,
        )
        assert out == segs  # ничего не изменилось

    def test_split_and_reassign_combined(self, monkeypatch):
        segs = [
            {"speaker": "1", "text": "Ответ. Новый вопрос?", "start_s": 0.0, "end_s": 4.0},
            {"speaker": "1", "text": "Целиком чужая реплика.", "start_s": 5.0, "end_s": 7.0},
            {"speaker": "0", "text": "Реплика второго голоса.", "start_s": 8.0, "end_s": 9.0},
        ]
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt, **kw: '[{"id": 0, "split_after": 1, "tail_speaker": "0"}, '
                                 '{"id": 1, "speaker": "0"}]',
        )
        out = correct_speaker_boundaries(
            segs, speaker_labels=self.LABELS, interviewer_id=None, merge_adjacent=False,
        )
        assert [(s["speaker"], s["text"]) for s in out] == [
            ("1", "Ответ."), ("0", "Новый вопрос?"), ("0", "Целиком чужая реплика."),
            ("0", "Реплика второго голоса."),
        ]


class TestGeminiClientTimeout:
    def test_client_created_with_timeout(self, monkeypatch):
        """Зависший сокет держал вызов неограниченно — клиент обязан иметь таймаут."""
        import sys
        import types as pytypes

        import backend.postprocess as pp

        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        fake_genai = pytypes.ModuleType("google.genai")
        fake_genai.Client = FakeClient
        fake_google = pytypes.ModuleType("google")
        fake_google.genai = fake_genai
        monkeypatch.setitem(sys.modules, "google", fake_google)
        monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
        monkeypatch.setattr(pp, "GEMINI_API_KEY", "test-key")

        client = pp._get_gemini_client()
        assert isinstance(client, FakeClient)
        assert captured["api_key"] == "test-key"
        assert captured["http_options"] == {"timeout": pp.GEMINI_TIMEOUT_SECONDS * 1000}


class TestSingingAndRehearsalPrompts:
    def test_singing_in_both_prompts(self):
        import backend.postprocess as pp
        for prompt in (pp._TECH_MOMENT_PROMPT, pp._TECH_MOMENT_PROMPT_AGGRESSIVE):
            assert "Happy birthday" in prompt
            assert "ПЕНИЕ" in prompt

    def test_rehearsal_and_wrapup_in_aggressive(self):
        import backend.postprocess as pp
        assert "дубл" in pp._TECH_MOMENT_PROMPT_AGGRESSIVE
        assert "Так, ну всё" in pp._TECH_MOMENT_PROMPT_AGGRESSIVE
        assert "Так, ну всё" not in pp._TECH_MOMENT_PROMPT

    def test_stray_latin_word_rule_in_polish(self, monkeypatch):
        import backend.postprocess as pp
        prompts = []
        monkeypatch.setattr(pp, "_gemini_call", lambda p, **kw: prompts.append(p) or "ок")
        pp.gemini_polish("текст")
        assert prompts and "many разных судеб" in prompts[0]

    def test_flag_defaults_flipped(self, monkeypatch):
        # Дефолты с ф4: агрессивные техмоменты и правка границ включены.
        import importlib

        import backend.config as config
        monkeypatch.delenv("TECH_MOMENT_AGGRESSIVE", raising=False)
        monkeypatch.delenv("SPEAKER_BOUNDARY_CORRECTION", raising=False)
        fresh = importlib.reload(config)
        try:
            assert fresh.TECH_MOMENT_AGGRESSIVE is True
            assert fresh.SPEAKER_BOUNDARY_CORRECTION is True
        finally:
            importlib.reload(config)


class TestF4PromptNudges:
    def test_glued_interjection_case_in_boundary_prompt(self, monkeypatch):
        import backend.postprocess as pp
        prompts = []
        monkeypatch.setattr(pp, "_gemini_call",
                            lambda p, **kw: prompts.append(p) or "[]")
        pp.correct_speaker_boundaries(
            [{"speaker": "0", "text": "Вопрос?"}, {"speaker": "1", "text": "Ответ."}],
            speaker_labels={"0": "АЗК", "1": "Гость"}, interviewer_id="0",
        )
        assert prompts and "Когда же ещё с таким праздником" in prompts[0]

    def test_staged_greeting_kept_in_aggressive(self):
        import backend.postprocess as pp
        assert "Hello, Марина" in pp._TECH_MOMENT_PROMPT_AGGRESSIVE
        assert "ПОСТАНОВОЧНЫЕ" in pp._TECH_MOMENT_PROMPT_AGGRESSIVE


class TestQuestionReassignGuard:
    def _events(self):
        return [
            {"speaker": "0", "text": "Какие-то яркие признания в любви случались?"},
            {"speaker": "1", "text": "Бывает иногда, мне даже в соцсетях пишут, но я так понимаю, что не мне."},
        ]

    def test_question_to_answerer_rejected(self, monkeypatch):
        import backend.postprocess as pp
        # Модель предлагает отдать вопрос спикеру следующего длинного ответа.
        monkeypatch.setattr(pp, "_gemini_call",
                            lambda p, **kw: '[{"id": 0, "speaker": "1"}]')
        out = pp.correct_speaker_boundaries(
            self._events(), speaker_labels={"0": "АЗК", "1": "Гость"},
            interviewer_id="0", merge_adjacent=False,
        )
        assert out[0]["speaker"] == "0"  # реассайн отклонён

    def test_legit_reassign_passes(self, monkeypatch):
        import backend.postprocess as pp
        # Не вопрос — обычная реплика, переназначение легитимно.
        events = [
            {"speaker": "0", "text": "Да, конечно, я готова начать."},
            {"speaker": "1", "text": "Отлично, тогда поехали дальше по плану интервью."},
        ]
        monkeypatch.setattr(pp, "_gemini_call",
                            lambda p, **kw: '[{"id": 0, "speaker": "1"}]')
        out = pp.correct_speaker_boundaries(
            events, speaker_labels={"0": "АЗК", "1": "Гость"},
            interviewer_id="0", merge_adjacent=False,
        )
        assert out[0]["speaker"] == "1"

    def test_never_reassign_rule_in_prompt(self, monkeypatch):
        import backend.postprocess as pp
        prompts = []
        monkeypatch.setattr(pp, "_gemini_call",
                            lambda p, **kw: prompts.append(p) or "[]")
        pp.correct_speaker_boundaries(
            self._events(), speaker_labels={"0": "АЗК", "1": "Гость"},
            interviewer_id="0",
        )
        assert prompts and "НИКОГДА не переназначай" in prompts[0]


class TestTailQuestionCandidates:
    def _idx(self, events):
        return list(enumerate(events))

    def test_multi_sentence_tail_question_marked(self):
        import backend.postprocess as pp
        events = [
            {"speaker": "1", "text": "Я ответила спасибо, мы расстались. Когда же ещё поздравят при жизни?"},
            {"speaker": "0", "text": "Расскажите про образ."},
        ]
        assert pp._tail_question_candidates(self._idx(events)) == {0}

    def test_single_sentence_question_not_marked(self):
        import backend.postprocess as pp
        events = [
            {"speaker": "0", "text": "Автографы люди не берут?"},
            {"speaker": "1", "text": "Берут иногда."},
        ]
        assert pp._tail_question_candidates(self._idx(events)) == set()

    def test_same_speaker_next_not_marked(self):
        import backend.postprocess as pp
        events = [
            {"speaker": "1", "text": "Первая мысль. Вторая мысль под вопросом?"},
            {"speaker": "1", "text": "Продолжение того же голоса."},
        ]
        assert pp._tail_question_candidates(self._idx(events)) == set()

    def test_mark_reaches_prompt(self, monkeypatch):
        import backend.postprocess as pp
        prompts = []
        monkeypatch.setattr(pp, "_gemini_call", lambda p, **kw: prompts.append(p) or "[]")
        pp.correct_speaker_boundaries(
            [
                {"speaker": "1", "text": "Длинный ответ гостя. А это чей вопрос в хвосте?"},
                {"speaker": "0", "text": "Следующий вопрос ведущего."},
            ],
            speaker_labels={"0": "АЗК", "1": "Гость"}, interviewer_id="0",
        )
        assert prompts and "⚠ПРОВЕРЬ-ХВОСТ" in prompts[0]
        assert "По КАЖДОМУ" in prompts[0]


class TestTailQuestionRecheck:
    EVENTS = [
        {"speaker": "1", "text": "Я ответила спасибо, мы расстались. Когда же ещё с таким праздником поздравят при жизни?",
         "start_s": 0.0, "end_s": 10.0},
        {"speaker": "0", "text": "Расскажите, вы разделяете это ощущение?", "start_s": 11.0, "end_s": 13.0},
    ]

    def test_next_verdict_splits_tail_to_next_speaker(self, monkeypatch):
        import backend.postprocess as pp
        monkeypatch.setattr(pp, "_gemini_call",
                            lambda p, **kw: '[{"id": 0, "tail": "next"}]')
        out, n = pp._recheck_tail_questions([dict(e) for e in self.EVENTS], {"0": "АЗК", "1": "КМ"})
        assert n == 1
        texts = [(s["speaker"], s["text"]) for s in out]
        assert ("1", "Я ответила спасибо, мы расстались.") in texts
        assert ("0", "Когда же ещё с таким праздником поздравят при жизни?") in texts

    def test_same_verdict_keeps(self, monkeypatch):
        import backend.postprocess as pp
        monkeypatch.setattr(pp, "_gemini_call",
                            lambda p, **kw: '[{"id": 0, "tail": "same"}]')
        events = [dict(e) for e in self.EVENTS]
        out, n = pp._recheck_tail_questions(events, {})
        assert n == 0 and out == events

    def test_no_candidates_no_call(self, monkeypatch):
        import backend.postprocess as pp
        def boom(p, **kw):
            raise AssertionError("не должен вызываться")
        monkeypatch.setattr(pp, "_gemini_call", boom)
        events = [{"speaker": "0", "text": "Просто реплика.", "start_s": 0, "end_s": 1}]
        out, n = pp._recheck_tail_questions(events, {})
        assert n == 0

    def test_gemini_failure_soft(self, monkeypatch):
        import backend.postprocess as pp
        monkeypatch.setattr(pp, "_gemini_call", lambda p, **kw: None)
        events = [dict(e) for e in self.EVENTS]
        out, n = pp._recheck_tail_questions(events, {})
        assert n == 0 and out == events

    def test_recheck_result_survives_when_main_pass_empty(self, monkeypatch):
        # Главный пасс вернул [], но до-пасс разрезал — результат не должен
        # откатиться к исходному списку.
        import backend.postprocess as pp
        calls = {"n": 0}

        def fake_call(p, **kw):
            calls["n"] += 1
            return "[]" if calls["n"] == 1 else '[{"id": 0, "tail": "next"}]'

        monkeypatch.setattr(pp, "_gemini_call", fake_call)
        out = pp.correct_speaker_boundaries(
            [dict(e) for e in self.EVENTS],
            speaker_labels={"0": "АЗК", "1": "КМ"}, interviewer_id="0",
            merge_adjacent=False,
        )
        assert any(s["speaker"] == "0" and s["text"].startswith("Когда же ещё") for s in out)


class TestMultiCutSplits:
    def test_dialog_chain_cut_into_three(self, monkeypatch):
        import backend.postprocess as pp
        events = [
            {"speaker": "0", "text": "Вы переехали после Олимпиады. После первой Олимпиады? После второй. Понятно, наездами жили?",
             "start_s": 0.0, "end_s": 12.0},
            {"speaker": "1", "text": "Да, так и жили.", "start_s": 13.0, "end_s": 15.0},
        ]
        monkeypatch.setattr(pp, "_gemini_call", lambda p, **kw:
                            '[{"id": 0, "cuts": [{"after": 2, "speaker": "1"}, {"after": 3, "speaker": "0"}]}]')
        out = pp.correct_speaker_boundaries(
            events, speaker_labels={"0": "БЯ", "1": "КС"}, interviewer_id="0",
            merge_adjacent=False,
        )
        texts = [(s["speaker"], s["text"]) for s in out]
        assert ("0", "Вы переехали после Олимпиады. После первой Олимпиады?") in texts
        assert ("1", "После второй.") in texts
        assert ("0", "Понятно, наездами жили?") in texts
        # тайминги монотонны
        times = [(s["start_s"], s["end_s"]) for s in out if "start_s" in s]
        assert all(a[1] <= b[0] + 1e-6 for a, b in zip(times, times[1:]))

    def test_invalid_cut_indexes_skipped(self):
        import backend.postprocess as pp
        events = [{"speaker": "0", "text": "Одно предложение.", "start_s": 0.0, "end_s": 1.0}]
        out, n = pp._apply_event_splits(events, {0: [(5, "1")]})
        assert n == 0 and out == events

    def test_cuts_protocol_in_prompt(self, monkeypatch):
        import backend.postprocess as pp
        prompts = []
        monkeypatch.setattr(pp, "_gemini_call", lambda p, **kw: prompts.append(p) or "[]")
        pp.correct_speaker_boundaries(
            [{"speaker": "0", "text": "Раз."}, {"speaker": "1", "text": "Два."}],
            speaker_labels={"0": "А", "1": "Б"}, interviewer_id=None,
        )
        assert prompts and '"cuts"' in prompts[0]

    def test_no_dash_dialog_rule_in_polish(self, monkeypatch):
        import backend.postprocess as pp
        prompts = []
        monkeypatch.setattr(pp, "_gemini_call", lambda p, **kw: prompts.append(p) or "ок")
        pp.gemini_polish("текст")
        assert prompts and "НИКОГДА не оформляй" in prompts[0]


class TestLifeProtection:
    def test_description_and_life_rule_in_tech_prompt(self, monkeypatch):
        import backend.postprocess as pp
        prompts = []
        monkeypatch.setattr(pp, "_gemini_call", lambda p, **kw: prompts.append(p) or "НЕТ")
        pp.detect_technical_segments(
            [{"text": "Пойдёмте, покажу дом, где мы жили.", "words": []}],
            description="интервью + лайфы: прогулка героя по родному городу",
        )
        assert prompts
        assert "прогулка героя по родному городу" in prompts[0]
        assert "ЛАЙФЫ" in prompts[0]
        assert "НЕ технические моменты" in prompts[0]

    def test_no_description_no_life_block(self, monkeypatch):
        import backend.postprocess as pp
        prompts = []
        monkeypatch.setattr(pp, "_gemini_call", lambda p, **kw: prompts.append(p) or "НЕТ")
        pp.detect_technical_segments([{"text": "Реплика.", "words": []}])
        assert prompts and "ЛАЙФЫ" not in prompts[0]

    def test_description_threaded_from_postprocess(self, monkeypatch):
        import backend.postprocess as pp
        seen = {}

        def fake_detect(segments, warnings=None, crew_names=None, description=None):
            seen["description"] = description
            return segments

        monkeypatch.setattr(pp, "GEMINI_API_KEY", "k")
        monkeypatch.setattr(pp, "_gemini_ready", lambda: True)
        monkeypatch.setattr(pp, "TECH_MOMENT_DETECTION", True)
        monkeypatch.setattr(pp, "detect_technical_segments", fake_detect)
        monkeypatch.setattr(pp, "gemini_polish", lambda text, **kw: text)
        pp.postprocess_segments(
            [{"text": "Реплика.", "words": []}],
            description="интервью + лайфы",
        )
        assert seen["description"] == "интервью + лайфы"


class TestBoundaryWindows:
    def _events(self, n, text="Реплика номер один в достаточно длинной форме?"):
        return [{"speaker": str(i % 2), "text": f"{text} {i}", "start_s": float(i), "end_s": i + 0.9}
                for i in range(n)]

    def test_long_transcript_multiple_windows(self, monkeypatch):
        import backend.postprocess as pp
        monkeypatch.setattr(pp, "_BOUNDARY_CHUNK_CHARS", 500)
        prompts = []
        monkeypatch.setattr(pp, "_gemini_call", lambda p, **kw: prompts.append(p) or "[]")
        pp.correct_speaker_boundaries(
            self._events(30), speaker_labels={"0": "А", "1": "Б"},
            interviewer_id="0", merge_adjacent=False,
        )
        # Хвост-вопросы дают ещё один маленький вызов — окон минимум 2 и больше 1.
        window_calls = [p for p in prompts if "Стенограмма (реплики" in p]
        assert len(window_calls) >= 3
        # Последняя реплика дошла до модели (обрезки 40k больше нет).
        assert any("Реплика номер один в достаточно длинной форме? 29" in p for p in window_calls)

    def test_context_line_corrections_ignored(self, monkeypatch):
        import backend.postprocess as pp
        monkeypatch.setattr(pp, "_BOUNDARY_CHUNK_CHARS", 500)
        calls = {"n": 0}

        def fake_call(p, **kw):
            calls["n"] += 1
            # Каждое окно пытается «исправить» реплику 0 — принять её должно
            # только первое окно (в остальных она контекст/вне диапазона).
            return '[{"id": 0, "speaker": "1"}]'

        monkeypatch.setattr(pp, "_gemini_call", fake_call)
        out = pp.correct_speaker_boundaries(
            self._events(30), speaker_labels={"0": "А", "1": "Б"},
            interviewer_id="0", merge_adjacent=False,
        )
        reassigned = [s for s in out if s["speaker"] == "1" and s["text"].endswith(" 0")]
        assert len(reassigned) == 1

    def test_partial_window_failure_warns_but_applies(self, monkeypatch):
        import backend.postprocess as pp
        monkeypatch.setattr(pp, "_BOUNDARY_CHUNK_CHARS", 500)
        calls = {"n": 0}

        def flaky(p, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise pp.GeminiPolishError("timeout")
            return '[{"id": 5, "speaker": "1"}]' if "Стенограмма (реплики" in p else "[]"

        monkeypatch.setattr(pp, "_gemini_call", flaky)
        warnings = []
        out = pp.correct_speaker_boundaries(
            self._events(30), speaker_labels={"0": "А", "1": "Б"},
            interviewer_id="0", warnings=warnings, merge_adjacent=False,
        )
        assert any("частично" in w for w in warnings)
        assert any(s["speaker"] == "1" and s["text"].endswith(" 5") for s in out)

    def test_all_windows_fail_soft(self, monkeypatch):
        import backend.postprocess as pp

        def boom(p, **kw):
            raise pp.GeminiPolishError("timeout")

        monkeypatch.setattr(pp, "_gemini_call", boom)
        events = self._events(4)
        warnings = []
        out = pp.correct_speaker_boundaries(
            events, speaker_labels={"0": "А", "1": "Б"},
            interviewer_id="0", warnings=warnings, merge_adjacent=False,
        )
        assert out == events
        assert any("не выполнена" in w for w in warnings)
