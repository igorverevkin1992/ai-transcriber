"""Tests for startup in-flight recovery toggle in main._recover_inflight_projects."""

import main
from backend.models import ProjectStatusEnum
from backend.store import ProjectStore


def _make_store(tmp_path):
    store = ProjectStore(db_path=str(tmp_path / "rec.db"))
    store.create("q1", {
        "id": "q1",
        "status": ProjectStatusEnum.QUEUED,
        "created_at": 1.0,
        "task_func": "process_video_task",
        "task_args": ("q1", "http://disk"),
    })
    return store


def test_recovery_disabled_marks_error(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    monkeypatch.setattr(main, "projects_db", store)
    monkeypatch.setattr(main, "RECOVER_INFLIGHT_ON_STARTUP", False)

    submitted = []
    monkeypatch.setattr(main, "submit_task", lambda *a, **k: submitted.append((a, k)))

    main._recover_inflight_projects()

    assert store["q1"]["status"] == ProjectStatusEnum.ERROR
    assert submitted == []  # ничего не возобновлено


def test_recovery_enabled_resubmits(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    monkeypatch.setattr(main, "projects_db", store)
    monkeypatch.setattr(main, "RECOVER_INFLIGHT_ON_STARTUP", True)

    submitted = []
    monkeypatch.setattr(main, "submit_task", lambda *a, **k: submitted.append((a, k)))

    main._recover_inflight_projects()

    assert store["q1"]["status"] == ProjectStatusEnum.QUEUED
    assert len(submitted) == 1  # задача возобновлена
