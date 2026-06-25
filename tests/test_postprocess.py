import backend.postprocess as pp
from backend.postprocess import (
    GeminiPolishError,
    gemini_infer_speaker_names,
    postprocess_segments,
    regex_cleanup,
)


class TestGeminiInferSpeakerNames:
    @staticmethod
    def _seg(sp, text):
        return {"timecode": "00:00:00:00", "speaker": sp, "text": text}

    def test_parses_and_validates(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt: '{"1": "Олег Александрович", "2": "Галина Васильевна", "0": null}',
        )
        segs = [self._seg("0", "Вопрос."), self._seg("1", "Ответ."), self._seg("2", "Ответ2.")]
        res = gemini_infer_speaker_names(segs, interviewer_id="0", guest_ids=["1", "2"])
        assert res == {"1": "Олег Александрович", "2": "Галина Васильевна"}

    def test_rejects_non_patronymic(self, monkeypatch):
        # «Олег» — одно слово; «Иванов Иван» — второе слово не отчество → отброшены.
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt: '{"1": "Олег", "2": "Иванов Иван"}')
        segs = [self._seg("1", "a"), self._seg("2", "b")]
        assert gemini_infer_speaker_names(segs, interviewer_id=None, guest_ids=["1", "2"]) is None

    def test_only_guest_ids(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt: '{"0": "Олег Александрович", "1": "Галина Васильевна"}',
        )
        segs = [self._seg("0", "a"), self._seg("1", "b")]
        res = gemini_infer_speaker_names(segs, interviewer_id="0", guest_ids=["1"])
        assert res == {"1": "Галина Васильевна"}  # "0" не в guest_ids

    def test_none_on_gemini_failure(self, monkeypatch):
        def boom(prompt):
            raise GeminiPolishError("fail")
        monkeypatch.setattr(pp, "_gemini_call", boom)
        assert gemini_infer_speaker_names([self._seg("1", "a")], interviewer_id=None, guest_ids=["1"]) is None

    def test_none_when_gemini_unavailable(self, monkeypatch):
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt: None)
        assert gemini_infer_speaker_names([self._seg("1", "a")], interviewer_id=None, guest_ids=["1"]) is None

    def test_extracts_json_with_preamble(self, monkeypatch):
        monkeypatch.setattr(
            pp, "_gemini_call",
            lambda prompt: 'Вот результат: {"1": "Олег Александрович"} — готово',
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

        def fake_call(prompt):
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

        def boom(text):
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
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt: "2")
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
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt: "НЕТ")
        segs = [{"text": "А."}, {"text": "Б."}]
        out = pp.detect_technical_segments(segs)
        assert [s["text"] for s in out] == ["А.", "Б."]

    def test_gemini_failure_warns_and_keeps(self, monkeypatch):
        import backend.postprocess as pp

        def boom(prompt):
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

        def fake(prompt):
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

        def fake(prompt):
            called["n"] += 1
            return "1"

        monkeypatch.setattr(pp, "_gemini_call", fake)
        monkeypatch.setattr(pp, "gemini_polish", lambda t: t)
        out = postprocess_segments([{"text": "Камера дубль.", "words": []}], warnings=[])
        assert out[0]["text"] == "Камера дубль."
        assert called["n"] == 0

    def test_postprocess_marks_and_skips_polish(self, monkeypatch):
        import backend.postprocess as pp
        from backend.turns import TECH_BREAK_TEXT
        monkeypatch.setattr(pp, "GEMINI_API_KEY", "fake-key")
        monkeypatch.setattr(pp, "TECH_MOMENT_DETECTION", True)
        monkeypatch.setattr(pp, "GLOSSARY_REPLACEMENT_PAIRS", [])
        monkeypatch.setattr(pp, "_gemini_call", lambda prompt: "1")
        polished = []

        def fake_polish(text):
            polished.append(text)
            return text + " [P]"

        monkeypatch.setattr(pp, "gemini_polish", fake_polish)
        out = postprocess_segments([{"text": "Камера, дубль.", "words": []}], warnings=[])
        assert out[0]["text"] == TECH_BREAK_TEXT
        assert polished == []  # маркер тех. момента не полируется
