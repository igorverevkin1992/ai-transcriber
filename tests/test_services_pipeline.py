"""Tests for core pipeline functions in backend.services.

These cover the deterministic units (result processing, cancellation,
auto-export, retry scheduling, executor lifecycle) with the heavy
Whisper/SpeechKit/ffmpeg dependencies mocked out or avoided.
"""

from pathlib import Path

import pytest

from backend import services
from backend.models import ProjectStatusEnum
from backend.store import ProjectStore


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """Swap services.projects_db with an isolated store + temp/output dirs."""
    store = ProjectStore(db_path=str(tmp_path / "svc.db"))
    temp_dir = tmp_path / "temp"
    out_dir = tmp_path / "out"
    temp_dir.mkdir()
    out_dir.mkdir()
    monkeypatch.setattr(services, "projects_db", store)
    monkeypatch.setattr(services, "TEMP_DIR", temp_dir)
    monkeypatch.setattr(services, "OUTPUT_DIR", out_dir)
    return store, temp_dir, out_dir


def _seg(channel, text, start_ms, end_ms):
    return {
        "channel_tag": channel,
        "text": text,
        "words": [{"text": text, "start_ms": start_ms, "end_ms": end_ms}],
    }


class TestProcessRecognitionResult:
    def test_builds_segments_speakers_and_timecodes(self, fresh_store):
        store, _, _ = fresh_store
        store.create("p1", {"id": "p1", "status": ProjectStatusEnum.TRANSCRIBING, "created_at": 1.0})
        segments = [
            _seg("0", "Привет.", 0, 1000),
            _seg("1", "Здравствуйте.", 1000, 3000),
            _seg("0", "Как дела?", 3000, 4000),
        ]
        services._process_recognition_result("p1", segments, "Иванов_Петров.mp4", Path("/nonexistent.mp4"))

        result = store["p1"]["result"]
        assert len(result["segments"]) == 3
        assert result["segments"][0]["timecode"] == "00:00:00:00"
        # speaker "1" speaks 2s vs "0" 2s total — both present
        assert set(result["speakers"].keys()) == {"0", "1"}
        assert store["p1"]["fps"] == 25  # fallback when video missing

    def test_filters_empty_text_segments(self, fresh_store):
        store, _, _ = fresh_store
        store.create("p2", {"id": "p2", "status": ProjectStatusEnum.TRANSCRIBING, "created_at": 1.0})
        segments = [
            _seg("0", "Реальный текст.", 0, 1000),
            _seg("0", "   ", 1000, 2000),  # whitespace-only -> dropped
            {"channel_tag": "0", "text": "Без слов", "words": []},  # no words -> dropped
        ]
        services._process_recognition_result("p2", segments, "file.mp4", Path("/nope.mp4"))
        assert len(store["p2"]["result"]["segments"]) == 1

    def test_speaker_names_from_filename(self, fresh_store):
        store, _, _ = fresh_store
        store.create("p3", {"id": "p3", "status": ProjectStatusEnum.TRANSCRIBING, "created_at": 1.0})
        # speaker 0 talks longest -> gets first filename name
        segments = [
            _seg("0", "много текста тут.", 0, 10000),
            _seg("1", "мало.", 10000, 11000),
        ]
        services._process_recognition_result("p3", segments, "Маданов_Антипенко.mp4", Path("/nope.mp4"))
        speakers = store["p3"]["result"]["speakers"]
        assert speakers["0"]["suggested_name"] == "Маданов"
        assert speakers["1"]["suggested_name"] == "Антипенко"

    def test_single_speaker_sets_low_confidence(self, fresh_store):
        store, _, _ = fresh_store
        store.create("p4", {"id": "p4", "status": ProjectStatusEnum.TRANSCRIBING, "created_at": 1.0})
        services._process_recognition_result(
            "p4", [_seg("0", "Один голос.", 0, 1000)], "file.mp4", Path("/nope.mp4")
        )
        assert store["p4"]["result"]["low_confidence_diarization"] is True

    def test_merges_same_speaker_turns(self, fresh_store):
        store, _, _ = fresh_store
        store.create("pm1", {"id": "pm1", "status": ProjectStatusEnum.TRANSCRIBING, "created_at": 1.0})
        segments = [
            _seg("0", "Привет.", 0, 1000),
            _seg("0", "Как дела?", 1500, 2500),
            _seg("1", "Хорошо.", 3000, 4000),
        ]
        services._process_recognition_result("pm1", segments, "file.mp4", Path("/nope.mp4"))
        result = store["pm1"]["result"]["segments"]
        assert len(result) == 2
        assert result[0]["text"] == "Привет. Как дела?"
        assert result[1]["text"] == "Хорошо."

    def test_speaker_durations_unaffected_by_merge(self, fresh_store):
        store, _, _ = fresh_store
        store.create("pm2", {"id": "pm2", "status": ProjectStatusEnum.TRANSCRIBING, "created_at": 1.0})
        segments = [
            _seg("0", "Привет.", 0, 1000),
            _seg("0", "Как дела?", 1500, 2500),
            _seg("1", "Хорошо.", 3000, 4000),
        ]
        services._process_recognition_result("pm2", segments, "file.mp4", Path("/nope.mp4"))
        speakers = store["pm2"]["result"]["speakers"]
        assert speakers["0"]["duration_sec"] == 2.0
        assert speakers["1"]["duration_sec"] == 1.0

    def test_turn_merge_disabled(self, fresh_store, monkeypatch):
        store, _, _ = fresh_store
        monkeypatch.setattr(services, "TURN_MERGE_ENABLED", False)
        store.create("pm3", {"id": "pm3", "status": ProjectStatusEnum.TRANSCRIBING, "created_at": 1.0})
        segments = [
            _seg("0", "Привет.", 0, 1000),
            _seg("0", "Как дела?", 1500, 2500),
        ]
        services._process_recognition_result("pm3", segments, "file.mp4", Path("/nope.mp4"))
        assert len(store["pm3"]["result"]["segments"]) == 2

    def test_tech_break_inserted_on_long_gap(self, fresh_store):
        store, _, _ = fresh_store
        store.create("pm4", {"id": "pm4", "status": ProjectStatusEnum.TRANSCRIBING, "created_at": 1.0})
        segments = [
            _seg("0", "До перерыва.", 0, 5000),
            _seg("0", "После перерыва.", 50000, 55000),
        ]
        services._process_recognition_result("pm4", segments, "file.mp4", Path("/nope.mp4"))
        result = store["pm4"]["result"]["segments"]
        assert [s["text"] for s in result] == [
            "До перерыва.", "(Технические моменты).", "После перерыва.",
        ]
        assert result[1]["timecode"] == "00:00:05:00"


class TestProcessVideoTask:
    def test_runs_postprocessing(self, fresh_store, monkeypatch):
        """Регрессия: SpeechKit-путь (Яндекс.Диск) должен прогонять постобработку."""
        store, _, _ = fresh_store
        store.create("pv1", {"id": "pv1", "status": ProjectStatusEnum.QUEUED, "created_at": 1.0})

        calls = {}
        monkeypatch.setattr(services, "_download_from_yadisk",
                            lambda pid, url, path: "Иванов_Петров.mp4")
        monkeypatch.setattr(services, "_convert_to_opus", lambda pid, i, o: None)
        monkeypatch.setattr(services, "_transcribe_with_speechkit",
                            lambda pid, path: [_seg("0", "привет.", 0, 1000)])

        import backend.postprocess as pp

        def fake_postprocess(segments, use_gemini=True):
            calls["postprocess"] = True
            return segments

        monkeypatch.setattr(pp, "postprocess_segments", fake_postprocess)

        services.process_video_task("pv1", "http://disk")

        assert calls.get("postprocess") is True
        assert store["pv1"]["status"] == ProjectStatusEnum.COMPLETED


class TestCancelProject:
    def test_returns_false_for_missing(self, fresh_store):
        assert services.cancel_project("ghost") is False

    def test_removes_project_and_temp_files(self, fresh_store):
        store, temp_dir, _ = fresh_store
        store.create("p5", {"id": "p5", "status": ProjectStatusEnum.QUEUED, "created_at": 1.0})
        # create matching temp artifacts
        (temp_dir / "p5_video.mp4").write_bytes(b"x")
        (temp_dir / "p5.opus").write_bytes(b"y")

        assert services.cancel_project("p5") is True
        assert "p5" not in store
        assert not (temp_dir / "p5_video.mp4").exists()
        assert not (temp_dir / "p5.opus").exists()


class TestAutoExportProject:
    def test_returns_none_without_result(self, fresh_store):
        store, _, _ = fresh_store
        store.create("p6", {"id": "p6", "status": ProjectStatusEnum.QUEUED, "created_at": 1.0})
        assert services.auto_export_project("p6", "/tmp/x.docx") is None

    def test_generates_docx_and_excludes_tech_speakers(self, fresh_store):
        store, temp_dir, _ = fresh_store
        store.create("p7", {
            "id": "p7",
            "status": ProjectStatusEnum.COMPLETED,
            "created_at": 1.0,
            "original_filename": "interview.mp4",
            "result": {
                "speakers": {
                    "0": {"duration_sec": 100.0, "suggested_name": "Иванов"},
                    "1": {"duration_sec": 50.0, "suggested_name": "АЗК"},
                },
                "segments": [
                    {"timecode": "00:00:01:00", "speaker": "0", "text": "Текст."},
                    {"timecode": "00:00:05:00", "speaker": "1", "text": "Вопрос."},
                ],
            },
        })
        out = str(temp_dir / "p7.docx")
        name = services.auto_export_project("p7", out)
        assert name == "interview.docx"
        assert Path(out).exists()

        from docx import Document
        doc = Document(out)
        legend = [p.text for p in doc.paragraphs if "–" in p.text]
        assert not any("АЗК" in t for t in legend)  # tech speaker excluded from legend
        assert any("Иванов" in t for t in legend)


    def test_service_labels_keep_full_prefix_and_dont_pollute_abbrs(self, fresh_store):
        store, temp_dir, _ = fresh_store
        store.create("p7b", {
            "id": "p7b",
            "status": ProjectStatusEnum.COMPLETED,
            "created_at": 1.0,
            "original_filename": "interview.mp4",
            "result": {
                "speakers": {
                    "0": {"duration_sec": 100.0, "suggested_name": "Антипенко"},
                    "1": {"duration_sec": 50.0, "suggested_name": "АЗК"},
                    "2": {"duration_sec": 30.0, "suggested_name": "Интервьюер"},
                },
                "segments": [
                    {"timecode": "00:00:01:00", "speaker": "0", "text": "Текст."},
                    {"timecode": "00:00:05:00", "speaker": "1", "text": "Вопрос."},
                    {"timecode": "00:00:10:00", "speaker": "2", "text": "Ответ."},
                ],
            },
        })
        out = str(temp_dir / "p7b.docx")
        services.auto_export_project("p7b", out)

        from docx import Document
        doc = Document(out)
        all_text = "\n".join(p.text for p in doc.paragraphs)
        # АЗК uses its full name as prefix, not a computed abbreviation
        assert "АЗК:" in all_text
        # Антипенко gets clean "А" (no collision with АЗК)
        assert "А:" in all_text


class TestMaybeRetry:
    def test_no_retry_when_not_error(self, fresh_store):
        store, _, _ = fresh_store
        store.create("p8", {"id": "p8", "status": ProjectStatusEnum.COMPLETED, "created_at": 1.0})
        services._maybe_retry("p8")
        assert store["p8"]["status"] == ProjectStatusEnum.COMPLETED

    def test_requeues_failed_project(self, fresh_store, monkeypatch):
        store, _, _ = fresh_store
        # capture Timer instead of really scheduling
        scheduled = {}

        class FakeTimer:
            def __init__(self, delay, func, args=(), kwargs=None):
                scheduled["delay"] = delay
                scheduled["args"] = args
            def start(self):
                scheduled["started"] = True

        monkeypatch.setattr(services.threading, "Timer", FakeTimer)
        store.create("p9", {
            "id": "p9",
            "status": ProjectStatusEnum.ERROR,
            "created_at": 1.0,
            "retry_count": 0,
            "task_func": "process_video_task",
            "task_args": ("p9", "http://disk"),
        })
        services._maybe_retry("p9")
        assert store["p9"]["status"] == ProjectStatusEnum.QUEUED
        assert store["p9"]["retry_count"] == 1
        assert scheduled.get("started") is True

    def test_stops_at_max_retries(self, fresh_store, monkeypatch):
        store, _, _ = fresh_store
        monkeypatch.setattr(services, "MAX_RETRIES", 3)
        store.create("p10", {
            "id": "p10",
            "status": ProjectStatusEnum.ERROR,
            "created_at": 1.0,
            "retry_count": 3,
            "task_func": "process_video_task",
            "task_args": ("p10", "http://disk"),
        })
        services._maybe_retry("p10")
        # stays ERROR, no requeue
        assert store["p10"]["status"] == ProjectStatusEnum.ERROR
        assert store["p10"]["retry_count"] == 3


class TestEnsureExecutor:
    def test_recreates_after_shutdown(self, monkeypatch):
        # simulate a shut-down executor
        services.shutdown_executor()
        assert services._executor_alive is False
        ex = services._ensure_executor()
        assert services._executor_alive is True
        # submit a trivial task to prove it's live
        fut = ex.submit(lambda: 42)
        assert fut.result(timeout=5) == 42
