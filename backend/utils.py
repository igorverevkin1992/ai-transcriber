import json
import re
import subprocess
from urllib.parse import urlparse

from backend.config import ALLOWED_EXTENSIONS, ALLOWED_URL_HOSTS, logger

# Control chars (C0/C1), zero-width and bidi-override chars used for filename spoofing
_FILENAME_BAD_CHARS_RE = re.compile(
    "[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\ufeff]"
)

FILENAME_STOP_WORDS = {
    "лайф", "лайфы", "интер", "синхрон", "снх", "бз",
    "f8", "wav", "mp3", "mp4", "mov", "wmv", "mxf",
}


_FILE_CODE_RE = re.compile(r"^[fф]\d+(-\d+)?$", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$|^\d{4}\.\d{2}\.\d{2}$")
_PURE_DIGITS_RE = re.compile(r"^\d+$")


def parse_filename_metadata(filename: str) -> dict:
    """Извлекает имена спикеров и стартовый таймкод из названия файла."""
    result = {"speakers": [], "start_tc": "00:00:00:00"}

    tc_match = re.search(r"(\d{2}:\d{2}:\d{2}:\d{2})", filename)
    if tc_match:
        result["start_tc"] = tc_match.group(1)
        filename = filename.replace(result["start_tc"], "")

    clean_name = re.sub(r"\.[^.]+$", "", filename)
    parts = re.split(r"[,_]+", clean_name)

    for part in parts:
        word = part.strip()
        if (
            word
            and word.lower() not in FILENAME_STOP_WORDS
            and not _DATE_RE.match(word)
            and not _FILE_CODE_RE.match(word)
            and not _PURE_DIGITS_RE.match(word)
        ):
            result["speakers"].append(word)

    return result


def frames_to_tc(frames: int, fps: int = 25) -> str:
    """Конвертирует кадры в SMPTE таймкод."""
    if frames < 0:
        frames = 0
    h = frames // (fps * 3600)
    rem = frames % (fps * 3600)
    m = rem // (fps * 60)
    rem = rem % (fps * 60)
    s = rem // fps
    f = rem % fps
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def tc_to_frames(tc_str: str, fps: int = 25) -> int:
    """Конвертирует SMPTE таймкод в кадры."""
    try:
        parts = list(map(int, tc_str.split(":")))
        return (parts[0] * 3600 + parts[1] * 60 + parts[2]) * fps + parts[3]
    except (ValueError, IndexError):
        return 0


def strip_extension(filename: str) -> str:
    """Убирает расширение файла."""
    return re.sub(r"\.[^.]+$", "", filename)


def sanitize_filename(filename: str, base_dir: str | None = None) -> str:
    """Очищает имя файла от path traversal и опасных символов.

    If base_dir is provided, verifies the resolved path stays within it.
    """
    import os
    name = os.path.basename(filename)
    name = name.replace("\\", "").replace("/", "")
    # Strip control, zero-width and bidi-override chars (filename spoofing)
    name = _FILENAME_BAD_CHARS_RE.sub("", name)
    name = name.strip(". ")
    if not name:
        return "unnamed_file"
    if base_dir:
        resolved = os.path.realpath(os.path.join(base_dir, name))
        if not resolved.startswith(os.path.realpath(base_dir)):
            return "unnamed_file"
    return name


def validate_url(url: str) -> str | None:
    """Проверяет, что URL ведёт на разрешённый хост. Возвращает ошибку или None."""
    try:
        parsed = urlparse(url)
        if parsed.hostname not in ALLOWED_URL_HOSTS:
            return f"URL должен вести на Яндекс.Диск ({', '.join(ALLOWED_URL_HOSTS)})"
        if parsed.scheme not in ("http", "https"):
            return "URL должен использовать протокол http или https"
    except Exception:
        return "Некорректный URL"
    return None


def validate_file_extension(filename: str) -> str | None:
    """Проверяет расширение файла. Возвращает ошибку или None."""
    ext = re.search(r"(\.[^.]+)$", filename.lower())
    if not ext or ext.group(1) not in ALLOWED_EXTENSIONS:
        return f"Формат файла не поддерживается. Допустимые: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
    return None


def detect_fps(file_path: str) -> int:
    """Определяет FPS видеофайла через ffprobe. Возвращает 25 по умолчанию."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "json",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                rate_str = streams[0].get("r_frame_rate", "25/1")
                num, den = map(int, rate_str.split("/"))
                fps = round(num / den) if den else 25
                if fps > 0:
                    logger.info("Определён FPS: %d", fps)
                    return fps
    except Exception as e:
        logger.warning("Не удалось определить FPS: %s", e)
    logger.warning("FPS не определён, используется значение по умолчанию: 25")
    return 25
