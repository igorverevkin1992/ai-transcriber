import ipaddress
import socket
from urllib.parse import urlparse

from backend.config import logger

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    magic = None
    MAGIC_AVAILABLE = False

ALLOWED_MIME_TYPES = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave",
    "audio/ogg", "audio/flac", "audio/x-flac", "audio/mp4", "audio/x-m4a",
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska",
    "video/x-ms-wmv", "video/x-mpegts", "video/mpeg", "application/mxf",
    "video/mxf", "application/octet-stream",
}


def validate_mime_type(file_bytes: bytes) -> str | None:
    """Проверяет MIME-тип по magic bytes. Возвращает ошибку или None.

    Если python-magic не установлен — пропускает проверку (с warning).
    """
    if not MAGIC_AVAILABLE:
        logger.warning("python-magic не установлен — MIME-валидация пропущена")
        return None

    try:
        mime = magic.from_buffer(file_bytes, mime=True)
    except Exception as e:
        logger.warning("MIME detection failed: %s", e)
        return None

    if mime not in ALLOWED_MIME_TYPES:
        return f"Тип файла '{mime}' не разрешён. Файл, возможно, не является аудио/видео."
    return None


def is_private_ip(ip_str: str) -> bool:
    """Проверяет, что IP — приватный, loopback, link-local или multicast."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_external_url(url: str) -> str | None:
    """Защита от SSRF: резолвит hostname и блокирует приватные IP.

    Возвращает текст ошибки или None если URL безопасен.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return "Некорректный URL"

    if parsed.scheme not in ("http", "https"):
        return "URL должен использовать http или https"

    hostname = parsed.hostname
    if not hostname:
        return "URL не содержит hostname"

    try:
        ips = {info[4][0] for info in socket.getaddrinfo(hostname, None)}
    except socket.gaierror:
        return f"Не удалось разрешить hostname: {hostname}"

    for ip in ips:
        if is_private_ip(ip):
            logger.warning("SSRF блок: hostname=%s резолвится в приватный IP %s", hostname, ip)
            return "URL ведёт на приватный IP — запрещено"

    return None


def mask_url(url: str, prefix_len: int = 30) -> str:
    """Маскирует URL для логов — оставляет только domain + начало пути."""
    try:
        parsed = urlparse(url)
        safe = f"{parsed.scheme}://{parsed.netloc}{parsed.path[:20]}"
        return safe[:prefix_len] + "..." if len(safe) > prefix_len else safe
    except Exception:
        return "<invalid-url>"
