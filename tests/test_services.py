from backend.services import (
    _compute_smart_abbreviations,
    _fold_unnamed_speakers_into_tech,
    _invert_name,
    _resegment_by_word_speakers,
)
from backend.turns import TECH_BREAK_TEXT


def _w(word, start, end, speaker=None):
    """Построить word-dict в формате whisperx (start/end в секундах)."""
    d = {"word": word, "start": start, "end": end}
    if speaker is not None:
        d["speaker"] = speaker
    return d


class TestResegmentByWordSpeakers:
    def test_single_speaker_preserves_original_text(self):
        # Все слова одного спикера → один сегмент, ИСХОДНЫЙ текст дословно.
        seg = {
            "text": "Полный текст реплики.", "start": 1.0, "end": 3.0,
            "speaker": "SPEAKER_01",
            "words": [_w("Полный", 1.0, 1.4, "SPEAKER_01"),
                      _w("текст", 1.4, 1.8, "SPEAKER_01"),
                      _w("реплики.", 1.8, 2.5, "SPEAKER_01")],
        }
        parts = _resegment_by_word_speakers(seg, "0")
        assert len(parts) == 1
        assert parts[0]["text"] == "Полный текст реплики."
        assert parts[0]["channel_tag"] == "1"
        assert parts[0]["start_ms"] == 1000
        assert parts[0]["end_ms"] == 3000

    def test_split_on_speaker_change(self):
        # Вставка другого спикера внутри сегмента → два куска.
        seg = {
            "text": "Около тридцати видов. Вы часто играете?",
            "start": 10.0, "end": 13.0, "speaker": "SPEAKER_01",
            "words": [_w("Около", 10.0, 10.3, "SPEAKER_01"),
                      _w("тридцати", 10.3, 10.7, "SPEAKER_01"),
                      _w("видов.", 10.7, 11.5, "SPEAKER_01"),
                      _w("Вы", 12.0, 12.2, "SPEAKER_00"),
                      _w("часто", 12.2, 12.5, "SPEAKER_00"),
                      _w("играете?", 12.5, 13.0, "SPEAKER_00")],
        }
        parts = _resegment_by_word_speakers(seg, "0")
        assert [p["channel_tag"] for p in parts] == ["1", "0"]
        assert parts[0]["text"] == "Около тридцати видов."
        assert parts[1]["text"] == "Вы часто играете?"
        assert parts[0]["start_ms"] == 10000
        assert parts[0]["end_ms"] == 11500
        assert parts[1]["start_ms"] == 12000
        assert parts[1]["end_ms"] == 13000

    def test_word_without_label_inherits_previous(self):
        # Слово без своей метки наследует спикера предыдущего слова.
        # sentence_boundary_only=False — проверяем чистую механику наследования
        # без правила границы предложения.
        seg = {
            "text": "А Б В Г", "start": 0.0, "end": 4.0, "speaker": "SPEAKER_00",
            "words": [_w("А", 0.0, 1.0, "SPEAKER_00"),
                      _w("Б", 1.0, 2.0),  # нет метки → наследует SPEAKER_00
                      _w("В", 2.0, 3.0, "SPEAKER_01"),
                      _w("Г", 3.0, 4.0)],  # наследует SPEAKER_01
        }
        parts = _resegment_by_word_speakers(seg, "0", sentence_boundary_only=False)
        assert [p["channel_tag"] for p in parts] == ["0", "1"]
        assert parts[0]["text"] == "А Б"
        assert parts[1]["text"] == "В Г"

    def test_no_split_on_midsentence_flip(self):
        # Хвост фразы героя, ошибочно помеченный другим спикером ПОСРЕДИ
        # предложения (нет точки перед сменой), не отрезается: «там попался»
        # остаётся за героем, а не утекает в начало реплики ведущего.
        seg = {
            "text": "Я выиграл тех, кто там попался.",
            "start": 10.0, "end": 13.0, "speaker": "SPEAKER_01",
            "words": [_w("Я", 10.0, 10.2, "SPEAKER_01"),
                      _w("выиграл", 10.2, 10.6, "SPEAKER_01"),
                      _w("тех,", 10.6, 10.9, "SPEAKER_01"),
                      _w("кто", 10.9, 11.2, "SPEAKER_01"),
                      _w("там", 11.2, 11.5, "SPEAKER_00"),      # ошибочный флип
                      _w("попался.", 11.5, 12.0, "SPEAKER_00")],
        }
        parts = _resegment_by_word_speakers(seg, "0")
        assert len(parts) == 1
        assert parts[0]["channel_tag"] == "1"
        assert parts[0]["text"] == "Я выиграл тех, кто там попался."

    def test_splits_at_sentence_boundary(self):
        # Настоящая вставка ведущего стоит ПОСЛЕ конца предложения → режется.
        seg = {
            "text": "Профессионально культивируем. Вы со студентами часто играете?",
            "start": 20.0, "end": 24.0, "speaker": "SPEAKER_01",
            "words": [_w("Профессионально", 20.0, 20.6, "SPEAKER_01"),
                      _w("культивируем.", 20.6, 21.2, "SPEAKER_01"),
                      _w("Вы", 21.5, 21.7, "SPEAKER_00"),
                      _w("со", 21.7, 21.9, "SPEAKER_00"),
                      _w("студентами", 21.9, 22.4, "SPEAKER_00"),
                      _w("часто", 22.4, 22.7, "SPEAKER_00"),
                      _w("играете?", 22.7, 23.2, "SPEAKER_00")],
        }
        parts = _resegment_by_word_speakers(seg, "0")
        assert [p["channel_tag"] for p in parts] == ["1", "0"]
        assert parts[0]["text"] == "Профессионально культивируем."
        assert parts[1]["text"] == "Вы со студентами часто играете?"

    def test_no_word_labels_uses_segment_speaker(self):
        # Ни одно слово не размечено, но у сегмента есть speaker → не режем.
        seg = {
            "text": "Текст.", "start": 0.0, "end": 2.0, "speaker": "SPEAKER_02",
            "words": [_w("Текст.", 0.0, 1.5)],
        }
        parts = _resegment_by_word_speakers(seg, "0")
        assert len(parts) == 1
        assert parts[0]["channel_tag"] == "2"
        assert parts[0]["text"] == "Текст."

    def test_no_speaker_anywhere_uses_fallback(self):
        # Нет меток ни у сегмента, ни у слов → спикер предыдущего сегмента.
        seg = {
            "text": "Текст.", "start": 0.0, "end": 2.0,
            "words": [_w("Текст.", 0.0, 1.5)],
        }
        parts = _resegment_by_word_speakers(seg, "3")
        assert len(parts) == 1
        assert parts[0]["channel_tag"] == "3"

    def test_no_words_falls_back_to_segment(self):
        # Сегмент без слов → один кусок с синтетическим словом и текстом.
        seg = {"text": "Без слов.", "start": 5.0, "end": 6.0, "speaker": "SPEAKER_00", "words": []}
        parts = _resegment_by_word_speakers(seg, "0")
        assert len(parts) == 1
        assert parts[0]["text"] == "Без слов."
        assert parts[0]["words"] == [{"text": "Без слов.", "start_ms": 5000, "end_ms": 6000}]

    def test_split_part_words_are_homogeneous(self):
        # Каждый кусок несёт только свои слова с таймкодами.
        seg = {
            "text": "Раз два три", "start": 0.0, "end": 3.0, "speaker": "SPEAKER_00",
            "words": [_w("Раз", 0.0, 1.0, "SPEAKER_00"),
                      _w("два", 1.0, 2.0, "SPEAKER_01"),
                      _w("три", 2.0, 3.0, "SPEAKER_01")],
        }
        parts = _resegment_by_word_speakers(seg, "0", sentence_boundary_only=False)
        assert len(parts) == 2
        assert [wd["text"] for wd in parts[0]["words"]] == ["Раз"]
        assert [wd["text"] for wd in parts[1]["words"]] == ["два", "три"]


class TestFoldUnnamedSpeakersIntoTech:
    def _speakers(self):
        return {
            "0": {"duration_sec": 100.0, "suggested_name": "АЗК"},
            "1": {"duration_sec": 200.0, "suggested_name": "Олег Александрович"},
            "3": {"duration_sec": 10.0, "suggested_name": "Спикер 4"},  # crew
        }

    def test_folds_generic_minor_speaker(self):
        segs = [
            {"timecode": "00:00:01:00", "speaker": "1", "text": "Реплика гостя."},
            {"timecode": "00:00:05:00", "speaker": "3", "text": "В моторе, да?"},
            {"timecode": "00:00:07:00", "speaker": "0", "text": "Вопрос."},
        ]
        new, folded = _fold_unnamed_speakers_into_tech(
            segs, self._speakers(), {"0", "1"}, 310.0, 0.15,
        )
        assert folded == {"3"}
        assert new[1]["text"] == TECH_BREAK_TEXT
        assert new[1]["speaker"] == "3"  # speaker сохранён, заменён только текст
        assert new[0]["text"] == "Реплика гостя."
        assert new[2]["text"] == "Вопрос."

    def test_does_not_fold_named_or_interviewer(self):
        speakers = {
            "0": {"duration_sec": 100.0, "suggested_name": "АЗК"},
            "1": {"duration_sec": 5.0, "suggested_name": "Олег Александрович"},  # малая доля, но НАЗВАН
        }
        segs = [{"timecode": "00:00:01:00", "speaker": "1", "text": "Коротко."}]
        new, folded = _fold_unnamed_speakers_into_tech(segs, speakers, {"0", "1"}, 105.0, 0.15)
        assert folded == set()
        assert new == segs

    def test_share_guard_protects_talkative_unnamed(self):
        # Генерик-спикер, но доля речи выше порога → не считаем крауд-группой.
        speakers = {
            "0": {"duration_sec": 100.0, "suggested_name": "АЗК"},
            "3": {"duration_sec": 100.0, "suggested_name": "Спикер 4"},
        }
        segs = [{"timecode": "00:00:01:00", "speaker": "3", "text": "Длинная реплика."}]
        new, folded = _fold_unnamed_speakers_into_tech(segs, speakers, {"0"}, 200.0, 0.15)
        assert folded == set()
        assert new == segs

    def test_collapses_adjacent_tech_markers(self):
        # Краудовый техмаркер рядом с паузным → схлопываются в один.
        segs = [
            {"timecode": "00:00:01:00", "speaker": "1", "text": TECH_BREAK_TEXT},  # пауза
            {"timecode": "00:00:03:00", "speaker": "3", "text": "В моторе."},      # crew → tech
            {"timecode": "00:00:05:00", "speaker": "1", "text": "Дальше."},
        ]
        new, folded = _fold_unnamed_speakers_into_tech(segs, self._speakers(), {"0", "1"}, 310.0, 0.15)
        assert folded == {"3"}
        assert [s["text"] for s in new] == [TECH_BREAK_TEXT, "Дальше."]
        assert new[0]["timecode"] == "00:00:01:00"  # таймкод первого маркера

    def test_no_generic_speakers_is_noop(self):
        speakers = {
            "0": {"duration_sec": 100.0, "suggested_name": "АЗК"},
            "1": {"duration_sec": 200.0, "suggested_name": "Олег Александрович"},
        }
        segs = [{"timecode": "00:00:01:00", "speaker": "1", "text": "Текст."}]
        new, folded = _fold_unnamed_speakers_into_tech(segs, speakers, {"0", "1"}, 300.0, 0.15)
        assert folded == set()
        assert new == segs


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

    def test_patronymic_uses_initials(self):
        # Имя-отчество → инициалы обоих слов (как в эталоне: ОА, ГВ).
        result = _compute_smart_abbreviations({
            "0": "Олег Александрович", "1": "Галина Васильевна",
        })
        assert result == {"0": "ОА", "1": "ГВ"}

    def test_patronymic_initials_collision(self):
        result = _compute_smart_abbreviations({
            "0": "Олег Александрович", "1": "Ольга Алексеевна",
        })
        assert result == {"0": "ОА1", "1": "ОА2"}


class TestInvertName:
    def test_surname_first_inverted(self):
        # «Фамилия Имя» из имени файла → «Имя Фамилия».
        assert _invert_name("Довлатова Алла") == "Алла Довлатова"

    def test_patronymic_not_inverted(self):
        # «Имя Отчество» не переставляем.
        assert _invert_name("Олег Александрович") == "Олег Александрович"
        assert _invert_name("Галина Васильевна") == "Галина Васильевна"

    def test_single_word_unchanged(self):
        assert _invert_name("Антипенко") == "Антипенко"


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


class TestAbbrTwoLetter:
    def test_two_letter_mode(self, monkeypatch):
        import backend.services as services
        monkeypatch.setattr(services, "ABBR_TWO_LETTER", True)
        result = _compute_smart_abbreviations({
            "0": "Арнальди Федерико", "1": "Морфео Доменико",
        })
        assert result == {"0": "АФ", "1": "МД"}

    def test_default_single_letter(self):
        result = _compute_smart_abbreviations({"0": "Майданов Денис"})
        assert result == {"0": "М"}

    def test_two_letter_patronymic_unchanged(self, monkeypatch):
        # Имя-отчество и так даёт инициалы — флаг ничего не меняет.
        import backend.services as services
        monkeypatch.setattr(services, "ABBR_TWO_LETTER", True)
        result = _compute_smart_abbreviations({"0": "Олег Александрович"})
        assert result == {"0": "ОА"}


class TestComputeDisplayNamesAndAbbrs:
    def test_default_keeps_surname_letter(self):
        # Конвенция f7/f8: показ «Имя Фамилия», аббревиатура — от фамилии.
        from backend.services import compute_display_names_and_abbrs
        display, abbrs = compute_display_names_and_abbrs({"0": "Довлатова Алла"})
        assert display == {"0": "Алла Довлатова"}
        assert abbrs == {"0": "Д"}

    def test_two_letter_from_displayed_form(self, monkeypatch):
        # Конвенция ф13: паспорт «Федерико Арнальди» → показ «Арнальди Федерико»,
        # аббревиатура — инициалы показываемой формы: «АФ».
        import backend.services as services
        from backend.services import compute_display_names_and_abbrs
        monkeypatch.setattr(services, "ABBR_TWO_LETTER", True)
        display, abbrs = compute_display_names_and_abbrs(
            {"0": "Федерико Арнальди", "1": "Доменико Морфео"})
        assert display == {"0": "Арнальди Федерико", "1": "Морфео Доменико"}
        assert abbrs == {"0": "АФ", "1": "МД"}
