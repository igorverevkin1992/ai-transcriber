import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import CORS_ORIGINS, TEMP_DIR, YANDEX_API_KEY, logger
from backend.models import HealthResponse
from backend.routes import router
from backend.s3 import check_bucket


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Проверки при старте и очистка при завершении."""
    logger.info("--- ЗАПУСК ПРОВЕРОК ---")

    if not YANDEX_API_KEY:
        logger.warning("YANDEX_API_KEY не задан — распознавание речи не будет работать.")

    check_bucket()

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


@app.get("/", response_model=HealthResponse)
def read_root():
    """Проверка здоровья сервера."""
    return HealthResponse(
        status="ok",
        service="ABTGS Backend",
        message="Перейдите на http://localhost:3000 для работы с интерфейсом.",
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Запуск ABTGS Backend на http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
