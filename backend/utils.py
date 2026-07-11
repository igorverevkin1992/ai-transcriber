import json
import re
import subprocess
from urllib.parse import urlparse

from backend.config import ALLOWED_EXTENSIONS, ALLOWED_URL_HOSTS, logger

# Control chars (C0/C1), zero-width and bidi-override chars used for filename spoofing
_FILENAME_BAD_CHARS_RE = re.compile(
    "[\x00-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\ufeff]"
)

# \u041b\u0438\u043c\u0438\u0442 \u0434\u043b\u0438\u043d\u044b \u0438\u043c\u0435\u043d\u0438 \u0444\u0430\u0439\u043b\u0430 (\u0441\u0438\u043c\u0432\u043e\u043b\u043e\u0432) \u2014 \u0442\u0438\u043f\u0438\u0447\u043d\u044b\u0439 \u043f\u0440\u0435\u0434\u0435\u043b \u0444\u0430\u0439\u043b\u043e\u0432\u044b\u0445 \u0441\u0438\u0441\u0442\u0435\u043c.
_MAX_FILENAME_LEN = 255

FILENAME_STOP_WORDS = {
    "лайф", "лайфы", "интер", "синхрон", "снх", "бз",
    "f8", "wav", "mp3", "mp4", "mov", "wmv", "mxf",
}


_FILE_CODE_RE = re.compile(r"^[fф]\d+(-\d+)?$", re.IGNORECASE)
# Токен числа говорящих: `s3` / `с3` (лат. и кир. «с») — подсказка для диаризации,
# когда имён в имени файла нет. По аналогии с `fN` (FPS).
_SPEAKER_COUNT_RE = re.compile(r"^[sс](\d+)$", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$|^\d{4}\.\d{2}\.\d{2}$")
_PURE_DIGITS_RE = re.compile(r"^\d+$")

# SMPTE-таймкод в имени файла. Двоеточия запрещены в Windows-именах,
# поэтому допускаем также точку/дефис/подчёркивание (один и тот же
# разделитель между всеми группами — \2 отсекает даты вида 02.03_05).
_FILENAME_TC_RE = re.compile(r"(?<!\d)(\d{2})([:.\-_])(\d{2})\2(\d{2})\2(\d{2})(?!\d)")


def _validate_tc_parts(h: int, m: int, s: int, f: int) -> bool:
    return h < 24 and m < 60 and s < 60 and f < 60


def parse_filename_metadata(filename: str) -> dict:
    """Извлекает имена спикеров и стартовый таймкод из названия файла."""
    result = {"speakers": [], "start_tc": "00:00:00:00", "num_speakers": 0}
    # Гард от регекспов по аномально длинному пользовательскому вводу:
    # легитимные имена файлов не превышают 255 байт (лимит файловых систем).
    filename = filename[:_MAX_FILENAME_LEN]

    tc_match = _FILENAME_TC_RE.search(filename)
    if tc_match:
        h, _sep, m, s, f = tc_match.groups()
        if _validate_tc_parts(int(h), int(m), int(s), int(f)):
            result["start_tc"] = f"{h}:{m}:{s}:{f}"
            filename = filename.replace(tc_match.group(0), "")

    clean_name = re.sub(r"\.[^.]+$", "", filename)
    parts = re.split(r"[,_]+", clean_name)

    for part in parts:
        word = part.strip()
        sc_match = _SPEAKER_COUNT_RE.match(word)
        if sc_match:
            result["num_speakers"] = int(sc_match.group(1))
            continue
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


def offset_tc(start_frames: int, seconds: float, fps: int = 25) -> str:
    """ТК начала речи с точностью до секунды (кадры :00)."""
    total = start_frames + round(seconds * fps)
    return frames_to_tc(total - total % fps, fps)


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
    if len(name) > _MAX_FILENAME_LEN:
        # Обрезаем с сохранением расширения
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) <= 16:
            name = stem[: _MAX_FILENAME_LEN - len(ext) - 1] + "." + ext
        else:
            name = name[:_MAX_FILENAME_LEN]
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


_EMBEDDED_TC_RE = re.compile(r"^(\d{2})[:;](\d{2})[:;](\d{2})[:;](\d{2})$")


def _normalize_embedded_tc(value: str) -> str | None:
    """Если значение — валидный SMPTE-таймкод, нормализует к "HH:MM:SS:FF"."""
    m = _EMBEDDED_TC_RE.match(value.strip())
    if m and _validate_tc_parts(*map(int, m.groups())):
        # ";" — drop-frame нотация (NTSC), нормализуем к ":"
        return "{}:{}:{}:{}".format(*m.groups())
    return None


def detect_start_timecode(file_path: str) -> str | None:
    """Читает встроенный SMPTE-таймкод из видео через ffprobe.

    Сначала ищет стандартный тег ``timecode`` (tmcd-дорожка MOV/MXF), затем —
    запасным проходом — сканирует ВСЕ теги формата и потоков на случай
    нестандартного поля (встречается в .wmv/ASF). Возвращает "HH:MM:SS:FF"
    или None, если таймкод отсутствует.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format_tags:stream_tags",
                "-of", "json",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        tag_dicts = [data.get("format", {}).get("tags", {}) or {}]
        for stream in data.get("streams", []):
            tag_dicts.append(stream.get("tags", {}) or {})

        # Приоритет — явный тег timecode.
        for tags in tag_dicts:
            tc = tags.get("timecode")
            if tc:
                normalized = _normalize_embedded_tc(tc)
                if normalized:
                    logger.info("Встроенный таймкод: %s", normalized)
                    return normalized
        # Запасной — любое строковое значение тега, выглядящее как ТК.
        for tags in tag_dicts:
            for key, val in tags.items():
                if key == "timecode" or not isinstance(val, str):
                    continue
                normalized = _normalize_embedded_tc(val)
                if normalized:
                    logger.info("Встроенный таймкод (тег %s): %s", key, normalized)
                    return normalized
        # ТК не найден — логируем, какие теги вообще есть в контейнере: по логу
        # видно, где (и есть ли) ТК в конкретном .wmv, без доступа к файлу.
        all_tags = {k: v for tags in tag_dicts for k, v in tags.items()}
        logger.info("ТК в контейнере не найден; теги контейнера: %s",
                    (json.dumps(all_tags, ensure_ascii=False)[:500] if all_tags else "нет"))
    except Exception as e:
        logger.warning("Не удалось прочитать встроенный таймкод: %s", e)
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
