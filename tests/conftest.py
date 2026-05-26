
import pytest


@pytest.fixture(autouse=True)
def isolated_temp_dirs(tmp_path, monkeypatch):
    """Изолирует TEMP_DIR и OUTPUT_DIR в tmp_path для каждого теста."""
    temp_dir = tmp_path / "temp_files"
    output_dir = tmp_path / "completed_docx"
    temp_dir.mkdir()
    output_dir.mkdir()
    monkeypatch.setattr("backend.config.TEMP_DIR", temp_dir)
    monkeypatch.setattr("backend.config.OUTPUT_DIR", output_dir)
    yield


@pytest.fixture
def mock_whisper_segments():
    """Sample WhisperX-style segments with diarization."""
    return [
        {
            "text": "Привет, это первый сегмент.",
            "channel_tag": "SPEAKER_00",
            "start_ms": 0,
            "end_ms": 3000,
            "words": [{"text": "Привет", "start_ms": 0, "end_ms": 500}],
        },
        {
            "text": "Хорошо, продолжаем.",
            "channel_tag": "SPEAKER_01",
            "start_ms": 3000,
            "end_ms": 5000,
            "words": [{"text": "Хорошо", "start_ms": 3000, "end_ms": 3500}],
        },
    ]


@pytest.fixture
def sample_project():
    """Готовый проект с result для тестов docx_export."""
    return {
        "original_filename": "test_interview.mp4",
        "result": {
            "speakers": {
                "0": {"duration_sec": 120.0, "suggested_name": "Майданов Денис"},
                "1": {"duration_sec": 80.0, "suggested_name": "Антипенко Григорий"},
            },
            "segments": [
                {"timecode": "11:04:15:00", "speaker": "0", "text": "(Технические моменты.)"},
                {"timecode": "11:04:18:00", "speaker": "0", "text": "Привет всем."},
                {"timecode": "11:04:25:00", "speaker": "1", "text": "Хорошо, давайте."},
            ],
        },
    }
