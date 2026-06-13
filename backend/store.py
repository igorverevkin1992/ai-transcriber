import json
import sqlite3
import threading
import time
from typing import Any

from backend.models import ProjectStatusEnum


class _EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "value"):
            return obj.value
        return super().default(obj)


class ProjectStore:
    """Thread-safe, SQLite-backed project store with in-memory cache.

    Hot data (progress_percent) lives only in cache.
    Persistent writes happen on: create, status change, result storage, delete.

    Concurrency contract: all mutations go through ``self._lock``. Read methods
    (``get``/``__getitem__``/``__contains__``/``items``/``__len__``) deliberately
    skip the lock — single dict operations are atomic under the GIL, and
    ``items()`` returns a snapshot list. Callers may observe a value that is
    being concurrently replaced, but never a corrupted cache.
    """

    def __init__(self, db_path: str = "projects.db"):
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()
        self._load_cache()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()

    def _load_cache(self) -> None:
        conn = self._get_conn()
        rows = conn.execute("SELECT id, data FROM projects").fetchall()
        for pid, data_json in rows:
            try:
                data = json.loads(data_json)
                status = data.get("status")
                if isinstance(status, str):
                    try:
                        data["status"] = ProjectStatusEnum(status)
                    except ValueError:
                        pass
                self._cache[pid] = data
            except (json.JSONDecodeError, TypeError):
                pass

    def _persist(self, pid: str) -> None:
        proj = self._cache.get(pid)
        if proj is None:
            return
        data_json = json.dumps(proj, cls=_EnumEncoder, ensure_ascii=False)
        status = proj.get("status", "")
        if hasattr(status, "value"):
            status = status.value
        created_at = proj.get("created_at", time.time())
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO projects (id, data, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET data=excluded.data, status=excluded.status, updated_at=excluded.updated_at""",
            (pid, data_json, str(status), created_at, time.time()),
        )
        conn.commit()

    def _delete_from_db(self, pid: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM projects WHERE id = ?", (pid,))
        conn.commit()

    # --- Dict-like interface ---

    def __contains__(self, pid: str) -> bool:
        return pid in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, pid: str) -> dict:
        return self._cache[pid]

    def get(self, pid: str, default=None) -> dict | None:
        return self._cache.get(pid, default)

    def create(self, pid: str, data: dict) -> None:
        with self._lock:
            self._cache[pid] = data
            self._persist(pid)

    def update_field(self, pid: str, field: str, value: Any, *, persist: bool = False) -> None:
        with self._lock:
            proj = self._cache.get(pid)
            if proj is None:
                return
            proj[field] = value
            if persist:
                self._persist(pid)

    def update_status(self, pid: str, status, *, error: str | None = None) -> None:
        with self._lock:
            proj = self._cache.get(pid)
            if proj is None:
                return
            proj["status"] = status
            if error is not None:
                proj["error"] = error
            self._persist(pid)

    def set_result(self, pid: str, result: dict, **extra_fields) -> None:
        with self._lock:
            proj = self._cache.get(pid)
            if proj is None:
                return
            proj["result"] = result
            for k, v in extra_fields.items():
                proj[k] = v
            self._persist(pid)

    def pop(self, pid: str, default=None) -> dict | None:
        with self._lock:
            val = self._cache.pop(pid, default)
        if val is not default:
            self._delete_from_db(pid)
        return val

    def items(self) -> list[tuple[str, dict]]:
        return list(self._cache.items())

    def cleanup_old(self, ttl_seconds: float, terminal_statuses: set | None = None) -> int:
        now = time.time()
        if terminal_statuses is None:
            terminal_statuses = {ProjectStatusEnum.COMPLETED.value, ProjectStatusEnum.ERROR.value}

        to_delete = []
        for pid, proj in self.items():
            status = proj.get("status")
            status_val = status.value if hasattr(status, "value") else str(status)
            if status_val in terminal_statuses:
                created = proj.get("created_at", now)
                if now - created > ttl_seconds:
                    to_delete.append(pid)
        for pid in to_delete:
            self.pop(pid, None)
        return len(to_delete)
