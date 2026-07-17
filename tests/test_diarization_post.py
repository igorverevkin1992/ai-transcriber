"""Tests for backend.diarization_post (interviewer/АЗК detection, ordering)."""

from backend.diarization_post import (
    build_speaker_sequence,
    detect_interviewer,
    first_appearance_order,
    infer_speaker_names_by_vocative,
)


def _ev(speaker, start_s):
    return {"speaker": speaker, "start_s": start_s, "end_s": start_s + 1}


class TestBuildSpeakerSequence:
    def test_collapses_consecutive_same_speaker(self):
        events = [_ev("0", 0), _ev("0", 1), _ev("1", 2), _ev("0", 3)]
        assert build_speaker_sequence(events) == ["0", "1", "0"]

    def test_sorts_by_start_time(self):
        events = [_ev("1", 5), _ev("0", 0), _ev("1", 6)]
        assert build_speaker_sequence(events) == ["0", "1"]


class TestFirstAppearanceOrder:
    def test_order_by_first_occurrence(self):
        events = [_ev("2", 0), _ev("0", 1), _ev("2", 2), _ev("1", 3)]
        assert first_appearance_order(events) == ["2", "0", "1"]


class TestDetectInterviewer:
    def test_multi_guest_three_interviews(self):
        # A (interviewer) alternates with X, Y, Z across three back-to-back blocks.
        seq = ["A", "X", "A", "X", "A", "Y", "A", "Y", "A", "Z", "A", "Z"]
        durations = {"A": 10.0, "X": 30.0, "Y": 30.0, "Z": 30.0}
        assert detect_interviewer(seq, durations) == "A"

    def test_two_person_returns_none_by_default(self):
        seq = ["A", "B", "A", "B"]
        durations = {"A": 5.0, "B": 20.0}
        assert detect_interviewer(seq, durations) is None

    def test_two_person_none_without_question_shares(self):
        # Только длительности недостаточно: без сигнала «кто спрашивает»
        # пометка в 1-на-1 не ставится (прежний min-duration-гесс был опасен).
        seq = ["A", "B", "A", "B"]
        durations = {"A": 5.0, "B": 20.0}
        assert detect_interviewer(seq, durations, label_single_guest=True) is None

    def test_two_person_question_asker_and_shorter_wins(self):
        # ф4: ведущая задаёт вопросы и говорит меньше → АЗК.
        seq = ["A", "B", "A", "B"]
        durations = {"A": 60.0, "B": 400.0}
        shares = {"A": 0.9, "B": 0.1}
        assert detect_interviewer(seq, durations, label_single_guest=True,
                                  question_shares=shares) == "A"

    def test_two_person_conflicting_signals_none(self):
        # Спрашивает чаще, но говорит БОЛЬШЕ — сигналы противоречат → None.
        seq = ["A", "B", "A", "B"]
        durations = {"A": 400.0, "B": 60.0}
        shares = {"A": 0.9, "B": 0.1}
        assert detect_interviewer(seq, durations, label_single_guest=True,
                                  question_shares=shares) is None

    def test_two_person_similar_question_shares_none(self):
        # Оба спрашивают примерно одинаково (беседа) → нет уверенности.
        seq = ["A", "B", "A", "B"]
        durations = {"A": 100.0, "B": 200.0}
        shares = {"A": 0.5, "B": 0.4}
        assert detect_interviewer(seq, durations, label_single_guest=True,
                                  question_shares=shares) is None

    def test_two_person_disabled_flag_none(self):
        seq = ["A", "B", "A", "B"]
        durations = {"A": 5.0, "B": 20.0}
        shares = {"A": 0.9, "B": 0.0}
        assert detect_interviewer(seq, durations, label_single_guest=False,
                                  question_shares=shares) is None

    def test_round_robin_no_clear_interviewer(self):
        seq = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]
        durations = {"A": 10.0, "B": 10.0, "C": 10.0}
        assert detect_interviewer(seq, durations) is None

    def test_tiebreak_by_turn_count(self):
        # All three tie on distinct-neighbour count (2); A has the most turns.
        seq = ["A", "B", "A", "C", "B", "C", "A"]
        durations = {"A": 5.0, "B": 5.0, "C": 5.0}
        assert detect_interviewer(seq, durations) == "A"

    def test_single_speaker_returns_none(self):
        assert detect_interviewer(["A", "A"], {"A": 10.0}) is None


class TestInferSpeakerNamesByVocative:
    @staticmethod
    def _seg(speaker, text):
        return {"timecode": "00:00:00:00", "speaker": speaker, "text": text}

    def test_addressed_guest_named(self):
        # Интервьюер (0) дважды обращается к гостю (1) → гость получает имя.
        segs = [
            self._seg("0", "Олег Александрович, расскажите."),
            self._seg("1", "Да, конечно."),
            self._seg("0", "А что думаете, Олег Александрович?"),
            self._seg("1", "Думаю так."),
        ]
        res = infer_speaker_names_by_vocative(segs, interviewer_id="0", guest_ids=["1"])
        assert res == {"1": "Олег Александрович"}

    def test_single_direct_address_names(self):
        # Одного ПРЯМОГО обращения (с запятой) достаточно.
        segs = [
            self._seg("0", "Олег Александрович, расскажите."),
            self._seg("1", "Да."),
        ]
        res = infer_speaker_names_by_vocative(segs, interviewer_id="0", guest_ids=["1"])
        assert res == {"1": "Олег Александрович"}

    def test_third_person_mention_not_counted(self):
        # «…а Галина Васильевна как опытный тренер нас рассудит» — упоминание в
        # 3-м лице (без запятой-обращения): к гостю (2), который отвечает, имя
        # ошибочно НЕ приписывается. А прямое обращение к ней — приписывается.
        segs = [
            self._seg("0", "Сыграем, а Галина Васильевна как тренер нас рассудит."),
            self._seg("1", "Давайте."),  # активный гость продолжает
            self._seg("0", "Галина Васильевна, оцените игру."),
            self._seg("2", "Мне понравилось."),
        ]
        res = infer_speaker_names_by_vocative(segs, interviewer_id="0", guest_ids=["1", "2"])
        assert res == {"2": "Галина Васильевна"}

    def test_mention_not_misattributed(self):
        # Гость (1) упоминает «Галина Васильевна»; следующий — интервьюер (0),
        # не гость → не засчитывается.
        segs = [
            self._seg("1", "Меня учила Галина Васильевна."),
            self._seg("0", "Понятно."),
            self._seg("1", "Снова Галина Васильевна, говорю."),
            self._seg("0", "Хорошо."),
        ]
        assert infer_speaker_names_by_vocative(segs, interviewer_id="0", guest_ids=["1"]) == {}

    def test_oblique_case_not_matched(self):
        # «Галины Васильевны» — родительный падеж, не обращение → не именуем.
        segs = [
            self._seg("0", "Спросим у Галины Васильевны."),
            self._seg("1", "Да."),
            self._seg("0", "Снова у Галины Васильевны."),
            self._seg("1", "Ага."),
        ]
        assert infer_speaker_names_by_vocative(segs, interviewer_id="0", guest_ids=["1"]) == {}

    def test_two_guests(self):
        segs = [
            self._seg("0", "Олег Александрович, вопрос."),
            self._seg("1", "Отвечаю."),
            self._seg("0", "Олег Александрович, ещё."),
            self._seg("1", "Ещё отвечаю."),
            self._seg("0", "Галина Васильевна, ваш черёд."),
            self._seg("2", "Да."),
            self._seg("0", "Галина Васильевна, спасибо."),
            self._seg("2", "Пожалуйста."),
        ]
        res = infer_speaker_names_by_vocative(segs, interviewer_id="0", guest_ids=["1", "2"])
        assert res == {"1": "Олег Александрович", "2": "Галина Васильевна"}


class TestDirectAddressValidation:
    SEGS = [
        {"speaker": "0", "text": "Светлана, мы сейчас находимся в знаковом месте. Как вам было отпускать дочь?"},
        {"speaker": "1", "text": "Я не могу сказать, что легко. Лена Николаевна, дочка Веры Ефремовны, вышла к ней тогда."},
        {"speaker": "0", "text": "Понимаю."},
        {"speaker": "1", "text": "Такой сложный вопрос, Яна, задаёшь."},
        {"speaker": "0", "text": "Спасибо."},
    ]

    def test_direct_address_validates(self):
        from backend.diarization_post import is_direct_address
        assert is_direct_address(self.SEGS, "Светлана", "1") is True

    def test_third_person_mention_rejected(self):
        from backend.diarization_post import is_direct_address
        # «Лена Николаевна, дочка…» — упоминание, за ним говорит "0", не "1";
        # и даже для "0" это часть повествования, но адресация формально к "0" —
        # ключевой кейс: имя НЕ должно валидироваться для гостьи "1".
        assert is_direct_address(self.SEGS, "Лена Николаевна", "1") is False

    def test_interviewer_addressed_by_name(self):
        from backend.diarization_post import infer_name_for_speaker
        assert infer_name_for_speaker(self.SEGS, "0") == "Яна"

    def test_truncated_form_merges(self):
        from backend.diarization_post import infer_name_for_speaker
        segs = [
            {"speaker": "1", "text": "Я переехала, Ян, уже после Олимпиады."},
            {"speaker": "0", "text": "После первой?"},
            {"speaker": "1", "text": "Такой вопрос, Яна, задаёшь."},
            {"speaker": "0", "text": "Ну да."},
        ]
        assert infer_name_for_speaker(segs, "0") == "Яна"

    def test_stopwords_not_names(self):
        from backend.diarization_post import _find_single_name_vocatives
        assert _find_single_name_vocatives("Знаете, даже не думала. Господи, как страшно.") == []
        assert _find_single_name_vocatives("Светлана, мы начинаем.") == ["Светлана"]

    def test_single_vocative_feeds_fallback_heuristic(self):
        from backend.diarization_post import infer_speaker_names_by_vocative
        segs = [
            {"speaker": "0", "text": "Светлана, расскажите про Омск."},
            {"speaker": "1", "text": "Мы жили там до переезда."},
        ]
        out = infer_speaker_names_by_vocative(segs, interviewer_id="0", guest_ids=["1"])
        assert out == {"1": "Светлана"}


class TestFalseVocativeGuards:
    def test_vvodnoe_stopword(self):
        from backend.diarization_post import _find_single_name_vocatives
        assert _find_single_name_vocatives("Естественно, большая нагрузка.") == []
        assert _find_single_name_vocatives("Действительно, так и было.") == []

    def test_self_use_rejects_candidate(self):
        # «Особенно» — нет в стоп-листе, но спикер "0" употребляет его САМ →
        # структурный фильтр отбрасывает; побеждает настоящее имя.
        from backend.diarization_post import infer_name_for_speaker
        segs = [
            {"speaker": "1", "text": "Особенно, знаете, тяжело было. Я переехала, Ян, после Олимпиады."},
            {"speaker": "0", "text": "Особенно в те годы это было сложно."},
            {"speaker": "1", "text": "Такой вопрос, Яна, задаёшь."},
            {"speaker": "0", "text": "Ну расскажите."},
        ]
        assert infer_name_for_speaker(segs, "0") == "Яна"

    def test_real_f14_shape(self):
        # Форма реального ф14: вводные у гостьи + два обращения к ведущей.
        from backend.diarization_post import infer_name_for_speaker
        segs = [
            {"speaker": "С", "text": "Нет, я переехала, Ян, уже после Олимпиады."},
            {"speaker": "КЕ", "text": "После первой Олимпиады?"},
            {"speaker": "С", "text": "Естественно, в омской школе стало сложно."},
            {"speaker": "КЕ", "text": "Понимаю. Естественно, это непросто."},
            {"speaker": "С", "text": "Такой сложный вопрос, Яна, задаешь."},
            {"speaker": "КЕ", "text": "Спасибо."},
        ]
        assert infer_name_for_speaker(segs, "КЕ") == "Яна"
