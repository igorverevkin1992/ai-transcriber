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
