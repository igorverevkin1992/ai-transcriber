"""Tests for backend.diarization_post (interviewer/АЗК detection, ordering)."""

from backend.diarization_post import (
    build_speaker_sequence,
    detect_interviewer,
    first_appearance_order,
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
