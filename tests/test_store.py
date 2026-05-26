"""Tests for ProjectStore (SQLite-backed project storage)."""
import os
import tempfile

import pytest

from backend.store import ProjectStore


@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return ProjectStore(db_path=db_path)


class TestProjectStoreBasic:
    def test_create_and_get(self, store):
        store.create("p1", {"id": "p1", "status": "queued", "created_at": 1.0})
        assert "p1" in store
        assert store["p1"]["status"] == "queued"
        assert store.get("p1")["id"] == "p1"

    def test_get_missing(self, store):
        assert store.get("missing") is None
        assert store.get("missing", "default") == "default"

    def test_getitem_missing_raises(self, store):
        with pytest.raises(KeyError):
            _ = store["missing"]

    def test_contains(self, store):
        assert "p1" not in store
        store.create("p1", {"id": "p1", "status": "queued", "created_at": 1.0})
        assert "p1" in store

    def test_len(self, store):
        assert len(store) == 0
        store.create("p1", {"id": "p1", "status": "queued", "created_at": 1.0})
        assert len(store) == 1
        store.create("p2", {"id": "p2", "status": "queued", "created_at": 1.0})
        assert len(store) == 2

    def test_pop(self, store):
        store.create("p1", {"id": "p1", "status": "queued", "created_at": 1.0})
        val = store.pop("p1")
        assert val["id"] == "p1"
        assert "p1" not in store
        assert store.pop("p1") is None
        assert store.pop("p1", "gone") == "gone"

    def test_items_snapshot(self, store):
        store.create("a", {"id": "a", "status": "queued", "created_at": 1.0})
        store.create("b", {"id": "b", "status": "queued", "created_at": 1.0})
        items = store.items()
        assert len(items) == 2
        ids = {pid for pid, _ in items}
        assert ids == {"a", "b"}


class TestProjectStoreUpdates:
    def test_update_field_no_persist(self, store):
        store.create("p1", {"id": "p1", "status": "queued", "created_at": 1.0})
        store.update_field("p1", "progress_percent", 42)
        assert store["p1"]["progress_percent"] == 42

    def test_update_status_persists(self, store, tmp_path):
        db_path = str(tmp_path / "test.db")
        store.create("p1", {"id": "p1", "status": "queued", "created_at": 1.0})
        store.update_status("p1", "transcribing")
        # Reload from disk
        store2 = ProjectStore(db_path=db_path)
        assert store2["p1"]["status"] == "transcribing"

    def test_update_status_with_error(self, store):
        store.create("p1", {"id": "p1", "status": "queued", "created_at": 1.0})
        store.update_status("p1", "error", error="something failed")
        assert store["p1"]["error"] == "something failed"

    def test_set_result(self, store, tmp_path):
        db_path = str(tmp_path / "test.db")
        store.create("p1", {"id": "p1", "status": "queued", "created_at": 1.0})
        store.set_result("p1", {"segments": [1, 2, 3]}, fps=25)
        assert store["p1"]["result"]["segments"] == [1, 2, 3]
        assert store["p1"]["fps"] == 25
        # Verify persistence
        store2 = ProjectStore(db_path=db_path)
        assert store2["p1"]["result"]["segments"] == [1, 2, 3]

    def test_update_missing_project_is_noop(self, store):
        store.update_field("missing", "x", 1)
        store.update_status("missing", "error")
        store.set_result("missing", {})


class TestProjectStorePersistence:
    def test_survives_reload(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        s1 = ProjectStore(db_path=db_path)
        s1.create("p1", {"id": "p1", "status": "queued", "created_at": 100.0})
        s1.update_status("p1", "completed")
        del s1

        s2 = ProjectStore(db_path=db_path)
        assert "p1" in s2
        assert s2["p1"]["status"] == "completed"

    def test_pop_removes_from_disk(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        s1 = ProjectStore(db_path=db_path)
        s1.create("p1", {"id": "p1", "status": "queued", "created_at": 100.0})
        s1.pop("p1")
        del s1

        s2 = ProjectStore(db_path=db_path)
        assert "p1" not in s2

    def test_progress_not_persisted_by_default(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        s1 = ProjectStore(db_path=db_path)
        s1.create("p1", {"id": "p1", "status": "queued", "created_at": 100.0})
        s1.update_field("p1", "progress_percent", 75)  # persist=False by default
        del s1

        s2 = ProjectStore(db_path=db_path)
        assert s2["p1"].get("progress_percent") is None


class TestProjectStoreCleanup:
    def test_cleanup_old_removes_completed(self, store):
        import time
        store.create("old", {"id": "old", "status": "completed", "created_at": 1.0})
        store.create("new", {"id": "new", "status": "completed", "created_at": time.time()})
        store.create("active", {"id": "active", "status": "queued", "created_at": 1.0})
        deleted = store.cleanup_old(3600)
        assert deleted == 1
        assert "old" not in store
        assert "new" in store
        assert "active" in store

    def test_cleanup_old_removes_error(self, store):
        store.create("err", {"id": "err", "status": "error", "created_at": 1.0})
        deleted = store.cleanup_old(3600)
        assert deleted == 1
        assert "err" not in store
