from backend.postprocess import postprocess_segments, regex_cleanup


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
        from backend.postprocess import gemini_polish
        import unittest.mock as mock

        captured_prompt = {}
        fake_response = mock.MagicMock()
        fake_response.text = "хм понятно"

        fake_model = mock.MagicMock()
        fake_model.generate_content.return_value = fake_response

        import backend.postprocess as pp
        original_model = pp._gemini_model
        pp._gemini_model = fake_model
        try:
            gemini_polish("хм понятно")
            prompt_text = fake_model.generate_content.call_args[0][0]
            assert "«хм»" in prompt_text
            assert "сохраняй" in prompt_text.split("«хм»")[1][:50]
            assert "Убирай" not in prompt_text or "«хм»" not in prompt_text.split("Убирай")[1][:50]
        finally:
            pp._gemini_model = original_model


class TestAbbreviationCapitalization:
    def test_abbreviation_expansion_lowercases_next_word(self):
        assert regex_cleanup("и т.д. Потом мы пошли") == "и так далее потом мы пошли"

    def test_abbreviation_expansion_keeps_acronym_after(self):
        assert regex_cleanup("и т.д. МХАТ продолжил") == "и так далее МХАТ продолжил"
