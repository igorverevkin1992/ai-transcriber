from backend.postprocess import postprocess_segments, regex_cleanup


class TestRegexCleanup:
    def test_removes_filler_words(self):
        assert regex_cleanup("эээ давай") == "Давай"
        assert regex_cleanup("ну вот хорошо") == "Хорошо"
        assert regex_cleanup("мммм да") == "Да"

    def test_collapses_repeated_words(self):
        assert regex_cleanup("слово слово слово") == "Слово"
        assert regex_cleanup("привет привет мир") == "Привет мир"

    def test_capitalizes_after_punctuation(self):
        assert regex_cleanup("привет. как дела") == "Привет. Как дела"
        assert regex_cleanup("да? хорошо") == "Да? Хорошо"

    def test_capitalizes_first_letter(self):
        assert regex_cleanup("привет мир") == "Привет мир"

    def test_empty_string(self):
        assert regex_cleanup("") == ""

    def test_clean_text_unchanged(self):
        assert regex_cleanup("Привет, мир.") == "Привет, мир."

    def test_multi_space_collapse(self):
        assert regex_cleanup("слово    другое") == "Слово другое"


class TestPostprocessSegments:
    def test_applies_regex_to_all_segments(self):
        segs = [
            {"text": "эээ привет", "channel_tag": 0},
            {"text": "ну вот как дела", "channel_tag": 1},
        ]
        result = postprocess_segments(segs, use_gemini=False)
        assert result[0]["text"] == "Привет"
        assert result[1]["text"] == "Как дела"

    def test_preserves_metadata(self):
        segs = [{"text": "эээ да", "channel_tag": "SPEAKER_00", "start_ms": 100}]
        result = postprocess_segments(segs, use_gemini=False)
        assert result[0]["channel_tag"] == "SPEAKER_00"
        assert result[0]["start_ms"] == 100

    def test_empty_list(self):
        assert postprocess_segments([], use_gemini=False) == []
