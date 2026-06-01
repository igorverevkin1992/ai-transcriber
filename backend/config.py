import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class JsonFormatter(logging.Formatter):
    """Структурированный JSON-вывод для логов (для парсинга в ELK / Cloud Logging)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()
_handler = logging.StreamHandler()
if LOG_FORMAT == "json":
    _handler.setFormatter(JsonFormatter())
else:
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger("abtgs")

# --- API Keys ---
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
API_KEY = os.getenv("API_KEY")  # if set, X-API-Key header required on /api/* requests
HF_TOKEN = os.getenv("HF_TOKEN")  # HuggingFace token for pyannote diarization models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Gemini API for text post-processing

# --- Paths ---
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("completed_docx")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Limits ---
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB


def _auto_detect_concurrent_tasks() -> int:
    """Автоопределение MAX_CONCURRENT_TASKS на основе VRAM (если есть GPU)."""
    env_val = os.getenv("MAX_CONCURRENT_TASKS")
    if env_val:
        return int(env_val)
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # WhisperX large + pyannote ~8GB per task
            return max(1, int(vram_gb / 8))
    except ImportError:
        pass
    return 2


MAX_CONCURRENT_TASKS = _auto_detect_concurrent_tasks()
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mov", ".mxf", ".mp4", ".wmv", ".avi", ".mkv", ".ogg", ".flac"}
ALLOWED_URL_HOSTS = {"yadi.sk", "disk.yandex.ru", "disk.yandex.com"}

# --- CORS ---
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "projects.db")

# --- CORS ---
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
