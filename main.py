import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import CORS_ORIGINS, TEMP_DIR, YANDEX_API_KEY, logger
from backend.models import HealthResponse
from backend.routes import router
from backend.services import shutdown_executor


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
    allow_methods=["*"],
    allow_headers=["*"],
)

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
