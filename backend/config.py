import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("abtgs")

# --- API Keys ---
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")

# --- Paths ---
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path("completed_docx")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Limits ---
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
# faster-whisper with INT8 uses ~1-2GB RAM per transcription (vs ~5GB for openai-whisper).
# 2 concurrent tasks is safe on most machines; override via env var if needed.
MAX_CONCURRENT_TASKS = int(os.getenv("MAX_CONCURRENT_TASKS", "2"))
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".mov", ".mxf", ".mp4", ".wmv", ".avi", ".mkv", ".ogg", ".flac"}
ALLOWED_URL_HOSTS = {"yadi.sk", "disk.yandex.ru", "disk.yandex.com"}

# --- CORS ---
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
