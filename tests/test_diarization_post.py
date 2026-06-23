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

    def test_two_person_labels_shorter_when_enabled(self):
        seq = ["A", "B", "A", "B"]
        durations = {"A": 5.0, "B": 20.0}
        assert detect_interviewer(seq, durations, label_single_guest=True) == "A"

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
