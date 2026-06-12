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

# --- Turn building (склейка реплик в абзацы, как в эталонных стенограммах) ---
TURN_MERGE_ENABLED = os.getenv("TURN_MERGE_ENABLED", "true").lower() in ("1", "true", "yes")
TURN_INLINE_TC_SECONDS = float(os.getenv("TURN_INLINE_TC_SECONDS", "60"))
TECH_BREAK_GAP_SECONDS = float(os.getenv("TECH_BREAK_GAP_SECONDS", "30"))

# --- Whisper tuning ---
# beam_size: выше = точнее, но медленнее (рекомендуется 5-10)
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
# Подсказка декодеру: контекст и стиль; имена спикеров из имени файла
# добавляются автоматически (см. services._build_whisper_prompt)
WHISPER_INITIAL_PROMPT = os.getenv("WHISPER_INITIAL_PROMPT", "Интервью на русском языке.")

# --- ASR anti-hallucination ---
# Известные Whisper-галлюцинации на музыке/тишине (подстрочный поиск,
# регистронезависимо, только для коротких сегментов). Env заменяет список.
_DEFAULT_HALLUCINATIONS = (
    "субтитры,редактор субтитров,продолжение следует,спасибо за просмотр,"
    "dimatorzok,дима торжок,подпишись,подписывайтесь на канал"
)
HALLUCINATION_BLACKLIST = [
    p.strip().lower()
    for p in os.getenv("HALLUCINATION_BLACKLIST", _DEFAULT_HALLUCINATIONS).split(",")
    if p.strip()
]
# Сегмент с avg_logprob ниже порога — неразборчивая речь: человек пишет
# "(неразборчиво)" (цензус: 8 случаев в эталонах), а не выдуманный текст.
UNCLEAR_LOGPROB_THRESHOLD = float(os.getenv("UNCLEAR_LOGPROB_THRESHOLD", "-1.2"))
# Сегмент с no_speech_prob выше порога — не речь, отбрасывается.
NO_SPEECH_PROB_THRESHOLD = float(os.getenv("NO_SPEECH_PROB_THRESHOLD", "0.85"))

# --- SpeechKit tuning ---
SPEECHKIT_LITERATURE_TEXT = os.getenv("SPEECHKIT_LITERATURE_TEXT", "false").lower() in ("1", "true", "yes")

# --- Gemini tuning ---
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# --- DOCX metadata ---
DOCX_AUTHOR = os.getenv("DOCX_AUTHOR", "")


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
