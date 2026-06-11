from backend.services import _compute_smart_abbreviations


class TestComputeSmartAbbreviations:
    def test_single_speaker(self):
        assert _compute_smart_abbreviations({"0": "Майданов Денис"}) == {"0": "М"}

    def test_distinct_first_letters(self):
        result = _compute_smart_abbreviations({"0": "Майданов", "1": "Антипенко"})
        assert result == {"0": "М", "1": "А"}

    def test_collision_two_speakers(self):
        result = _compute_smart_abbreviations({"0": "Майданов", "1": "Мартынов"})
        assert result == {"0": "М1", "1": "М2"}

    def test_collision_three_speakers(self):
        result = _compute_smart_abbreviations({
            "0": "Майданов", "1": "Мартынов", "2": "Михайлов",
        })
        assert result == {"0": "М1", "1": "М2", "2": "М3"}

    def test_mixed_collision_and_unique(self):
        result = _compute_smart_abbreviations({
            "0": "Майданов", "1": "Мартынов",
            "2": "Антипенко", "3": "Аркадьев",
            "4": "Зайцев",
        })
        assert result == {"0": "М1", "1": "М2", "2": "А1", "3": "А2", "4": "З"}

    def test_empty_name_fallback(self):
        result = _compute_smart_abbreviations({"0": "", "1": "Иванов"})
        assert result == {"0": "С0", "1": "И"}

    def test_whitespace_name_fallback(self):
        result = _compute_smart_abbreviations({"0": "   "})
        assert result == {"0": "С0"}

    def test_non_letter_start_fallback(self):
        result = _compute_smart_abbreviations({"0": "123 Спикер"})
        assert result == {"0": "С0"}

    def test_unicode_uppercase(self):
        result = _compute_smart_abbreviations({"0": "ёлкина Анна"})
        # ё uppercase should work
        assert result["0"] in ("Ё", "ё".upper())


class TestHallucinationFilter:
    def _seg(self, text):
        return {"text": text, "channel_tag": 0, "start_ms": 0, "end_ms": 1000, "words": []}

    def test_known_hallucinations_dropped(self):
        from backend.services import _is_hallucination
        assert _is_hallucination("Субтитры сделал DimaTorzok")
        assert _is_hallucination("Редактор субтитров А.Семкин")
        assert _is_hallucination("Продолжение следует...")
        assert _is_hallucination("Спасибо за просмотр!")
        assert _is_hallucination("ПОДПИШИСЬ НА КАНАЛ")

    def test_long_speech_about_subtitles_kept(self):
        from backend.services import _is_hallucination
        text = ("Мы тогда долго спорили о том, нужны ли субтитры в этом фильме, "
                "потому что зритель привык к дубляжу, и в итоге решили оставить оба варианта")
        assert not _is_hallucination(text)

    def test_prompt_echo_dropped(self):
        from backend.services import _is_hallucination
        prompt = "Интервью на русском языке. Участники: Денис Майданов."
        assert _is_hallucination("Интервью на русском языке.", prompt)
        assert not _is_hallucination("Мы поехали на гастроли.", prompt)

    def test_prompt_echo_does_not_flag_speaker_name(self):
        from backend.services import _is_hallucination
        prompt = "Интервью на русском языке. Участники: Денис Майданов, Григорий Антипенко."
        assert not _is_hallucination("Денис Майданов", prompt)
        assert not _is_hallucination("Григорий Антипенко", prompt)

    def test_prompt_echo_flags_full_echo(self):
        from backend.services import _is_hallucination
        prompt = "Интервью на русском языке. Участники: Денис Майданов."
        assert _is_hallucination(prompt, prompt)

    def test_filter_keeps_normal_speech(self):
        from backend.services import _filter_hallucinated_segments
        segs = [self._seg("Обычная речь."), self._seg("Субтитры сделал DimaTorzok")]
        result = _filter_hallucinated_segments("test-project", segs)
        assert len(result) == 1
        assert result[0]["text"] == "Обычная речь."


class TestResolveWhisperText:
    def test_normal_segment_kept(self):
        from backend.services import _resolve_whisper_text
        assert _resolve_whisper_text("Привет.", -0.3, 0.1) == "Привет."

    def test_no_speech_dropped(self):
        from backend.services import _resolve_whisper_text
        assert _resolve_whisper_text("шум", -0.5, 0.95) is None

    def test_low_confidence_becomes_unclear(self):
        from backend.services import _resolve_whisper_text
        from backend.turns import UNCLEAR_TEXT
        assert _resolve_whisper_text("выдуманный текст", -2.0, 0.2) == UNCLEAR_TEXT

    def test_missing_metrics_kept(self):
        from backend.services import _resolve_whisper_text
        assert _resolve_whisper_text("Текст.", None, None) == "Текст."
