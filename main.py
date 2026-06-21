import asyncio
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.auth import ApiKeyMiddleware
from backend.config import API_KEY, AUTO_RECOVER_ON_STARTUP, CORS_ORIGINS, OUTPUT_DIR, SQLITE_DB_PATH, TEMP_DIR, YANDEX_API_KEY, logger
from backend.metrics import CONTENT_TYPE_LATEST, generate_latest
from backend.models import HealthResponse, ProjectStatusEnum
from backend.routes import router
from backend.services import (
    _TASK_REGISTRY,
    _cleanup_old_projects,
    projects_db,
    shutdown_executor,
    submit_task,
)

DOCX_TTL_SECONDS = 30 * 24 * 3600  # 30 days
CLEANUP_INTERVAL_SECONDS = 3600  # run housekeeping every hour


SQLITE_BACKUP_KEEP = 5  # number of timestamped backups to retain


def _backup_sqlite():
    """Создаёт timestamped-копию SQLite перед запуском и ротирует старые."""
    db_path = Path(SQLITE_DB_PATH)
    if not db_path.exists():
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    backup_path = db_path.with_suffix(db_path.suffix + f".backup-{stamp}")
    try:
        shutil.copy2(db_path, backup_path)
        logger.info("SQLite резервная копия создана: %s", backup_path)
    except OSError as e:
        logger.warning("Не удалось создать backup SQLite: %s", e)
        return

    # Retain only the newest SQLITE_BACKUP_KEEP backups
    backups = sorted(db_path.parent.glob(db_path.name + ".backup-*"))
    for old in backups[:-SQLITE_BACKUP_KEEP]:
        try:
            old.unlink()
        except OSError:
            pass


def _cleanup_old_docx():
    """Удаляет DOCX из OUTPUT_DIR старше DOCX_TTL_SECONDS."""
    now = time.time()
    removed = 0
    for f in OUTPUT_DIR.glob("*.docx"):
        try:
            if now - f.stat().st_mtime > DOCX_TTL_SECONDS:
                f.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        logger.info("Очищено %d старых DOCX-файлов (>%d дней)", removed, DOCX_TTL_SECONDS // 86400)


def _recover_inflight_projects():
    """Обработать задачи, прерванные рестартом сервера.

    AUTO_RECOVER_ON_STARTUP=true — автоматически пере-отправить их в очередь.
    AUTO_RECOVER_ON_STARTUP=false (по умолчанию) — только пометить как
    прерванные, сохранив task_func/task_args, чтобы пользователь мог запустить
    продолжение вручную (эндпоинт /projects/{pid}/resume).
    """
    in_flight_statuses = (
        ProjectStatusEnum.DOWNLOADING,
        ProjectStatusEnum.CONVERTING,
        ProjectStatusEnum.TRANSCRIBING,
    )
    recovered = 0
    interrupted = 0
    for pid, proj in projects_db.items():
        status = proj.get("status")
        if status not in in_flight_statuses and status != ProjectStatusEnum.QUEUED:
            continue

        task_func_name = proj.get("task_func")
        task_args = proj.get("task_args")
        resumable = bool(task_func_name and task_func_name in _TASK_REGISTRY and task_args)

        if not resumable:
            projects_db.update_status(pid, ProjectStatusEnum.ERROR,
                                      error="Сервер перезапущен, задача не может быть восстановлена")
            continue

        if AUTO_RECOVER_ON_STARTUP:
            func = _TASK_REGISTRY[task_func_name]
            task_kwargs = proj.get("task_kwargs", {})
            projects_db.update_status(pid, ProjectStatusEnum.QUEUED)
            submit_task(func, *task_args, project_id=pid, **task_kwargs)
            recovered += 1
            logger.info("[%s] Восстановлен из '%s' → QUEUED", pid[:8], status.value if hasattr(status, "value") else status)
        else:
            # Не запускаем — оставляем task_args для ручного возобновления.
            projects_db.update_status(pid, ProjectStatusEnum.ERROR,
                                      error="Прервано перезапуском сервера. Нажмите «Продолжить» для возобновления.")
            interrupted += 1

    if recovered:
        logger.info("Восстановлено %d задач после рестарта", recovered)
    if interrupted:
        logger.info("Авто-восстановление отключено: %d задач помечено как прерванные", interrupted)


async def _periodic_cleanup():
    """Каждый час удаляет старые DOCX и завершённые проекты."""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        try:
            _cleanup_old_docx()
            _cleanup_old_projects()
        except Exception as e:
            logger.warning("Периодическая очистка завершилась с ошибкой: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Проверки при старте и очистка при завершении."""
    logger.info("--- ЗАПУСК ПРОВЕРОК ---")

    if not YANDEX_API_KEY:
        logger.warning("YANDEX_API_KEY не задан — облачное распознавание (SpeechKit) не будет работать.")

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            logger.info("FFmpeg найден.")
        else:
            logger.error("FFmpeg вернул код ошибки %d.", result.returncode)
    except FileNotFoundError:
        logger.error("FFmpeg не найден в PATH. Установите ffmpeg.")
    except Exception as e:
        logger.error("Ошибка при проверке FFmpeg: %s", e)

    _backup_sqlite()
    _cleanup_old_docx()
    _recover_inflight_projects()

    cleanup_task = asyncio.create_task(_periodic_cleanup())

    logger.info("--- ПРОВЕРКИ ЗАВЕРШЕНЫ ---")
    yield

    cleanup_task.cancel()
    shutdown_executor()
    for f in TEMP_DIR.iterdir():
        try:
            f.unlink()
        except OSError:
            pass


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.add_middleware(ApiKeyMiddleware)

if API_KEY:
    logger.info("API_KEY задан — аутентификация по X-API-Key включена.")
else:
    logger.warning("API_KEY не задан — все /api/* эндпоинты открыты без аутентификации!")

app.include_router(router)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Проверка здоровья сервера (для Docker HEALTHCHECK)."""
    return HealthResponse(
        status="ok",
        service="ABTGS Backend",
        message="Сервер работает.",
    )


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus metrics endpoint (без auth для сборщиков)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Serve built frontend in production (Docker copies dist/ to static/)
static_dir = Path("static")
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    logger.info("Раздача статики из /static включена (production mode).")


if __name__ == "__main__":
    import uvicorn

    logger.info("Запуск ABTGS Backend на http://localhost:8000")
    # Привязка к 0.0.0.0 намеренная: контейнерный деплой, порт публикуется через docker-compose
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
