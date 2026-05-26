import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.auth import ApiKeyMiddleware
from backend.config import API_KEY, CORS_ORIGINS, TEMP_DIR, YANDEX_API_KEY, logger
from backend.models import HealthResponse
from backend.routes import router
from backend.models import ProjectStatusEnum
from backend.services import (
    _maybe_retry,
    projects_db,
    shutdown_executor,
    submit_task,
    _TASK_REGISTRY,
)


def _recover_inflight_projects():
    """Mark projects that were in-flight at shutdown and schedule retries."""
    in_flight_statuses = (
        ProjectStatusEnum.DOWNLOADING,
        ProjectStatusEnum.CONVERTING,
        ProjectStatusEnum.TRANSCRIBING,
    )
    recovered = 0
    for pid, proj in projects_db.items():
        status = proj.get("status")
        if status in in_flight_statuses or status == ProjectStatusEnum.QUEUED:
            task_func_name = proj.get("task_func")
            task_args = proj.get("task_args")
            if task_func_name and task_func_name in _TASK_REGISTRY and task_args:
                func = _TASK_REGISTRY[task_func_name]
                task_kwargs = proj.get("task_kwargs", {})
                projects_db.update_status(pid, ProjectStatusEnum.QUEUED)
                submit_task(func, *task_args, project_id=pid, **task_kwargs)
                recovered += 1
                logger.info("[%s] Восстановлен из '%s' → QUEUED", pid[:8], status.value if hasattr(status, "value") else status)
            else:
                projects_db.update_status(pid, ProjectStatusEnum.ERROR,
                                          error="Сервер перезапущен, задача не может быть восстановлена")
    if recovered:
        logger.info("Восстановлено %d задач после рестарта", recovered)


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

    _recover_inflight_projects()

    logger.info("--- ПРОВЕРКИ ЗАВЕРШЕНЫ ---")
    yield

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

app.include_router(router)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Проверка здоровья сервера (для Docker HEALTHCHECK)."""
    return HealthResponse(
        status="ok",
        service="ABTGS Backend",
        message="Сервер работает.",
    )


# Serve built frontend in production (Docker copies dist/ to static/)
static_dir = Path("static")
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
    logger.info("Раздача статики из /static включена (production mode).")


if __name__ == "__main__":
    import uvicorn

    logger.info("Запуск ABTGS Backend на http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
