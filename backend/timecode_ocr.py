"""Чтение выжженного в кадр SMPTE-таймкода через OCR (easyocr).

Используется, когда стартовый таймкод не задан ни в имени файла, ни в
метаданных контейнера. Извлекает несколько кадров в начале записи, распознаёт
область с таймкодом и по согласованности между кадрами вычисляет таймкод
ПЕРВОГО кадра файла (от которого затем отсчитываются реплики).
"""
from __future__ import annotations

import re
import subprocess
import threading
import uuid
from pathlib import Path

from backend.config import OCR_TIMECODE_REGION, TEMP_DIR, logger
from backend.utils import _validate_tc_parts, frames_to_tc, tc_to_frames

# ТК в распознанном тексте: разделители OCR часто путает (':', ';', '.', пробел)
# или теряет вовсе — допускаем необязательный разделитель между парами цифр.
_TC_IN_TEXT_RE = re.compile(r"(\d{2})[:;.\s]?(\d{2})[:;.\s]?(\d{2})[:;.\s]?(\d{2})")

_reader = None
_reader_lock = threading.Lock()


def _get_reader():
    """Лениво создаёт и кэширует easyocr.Reader (модель грузится один раз)."""
    global _reader
    if _reader is not None:
        return _reader
    with _reader_lock:
        if _reader is None:
            try:
                import easyocr
            except ImportError:
                logger.warning(
                    "easyocr не установлен — чтение выжженного в кадр ТК недоступно "
                    "(pip install easyocr)."
                )
                return None
            try:
                import torch
                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
            _reader = easyocr.Reader(["en"], gpu=gpu)
    return _reader


def _parse_region(raw: str | None) -> tuple[float, float, float, float] | None:
    """Разбирает "left,top,right,bottom" (доли 0-1) в кортеж или None."""
    if not raw:
        return None
    try:
        left, top, right, bottom = (float(x) for x in raw.split(","))
    except (ValueError, AttributeError):
        return None
    if 0 <= left < right <= 1 and 0 <= top < bottom <= 1:
        return left, top, right, bottom
    return None


def parse_start_timecode_from_ocr(
    samples: list[tuple[int, list[str]]], fps: int
) -> str | None:
    """Чистая логика: по OCR-результатам кадров вычислить ТК первого кадра файла.

    ``samples`` — список ``(индекс_кадра, [распознанные_строки])``. Для каждого
    кадра считаем «подразумеваемые старты» = ТК минус номер кадра; берём старт,
    согласованный минимум в 2 кадрах (±1 кадр). Это отсекает мусор OCR и
    подтверждает, что таймкод действительно идёт непрерывно.
    """
    if fps <= 0:
        return None

    starts_per_frame: list[set[int]] = []
    for frame_index, texts in samples:
        frame_starts: set[int] = set()
        for text in texts:
            for m in _TC_IN_TEXT_RE.finditer(text):
                h, mn, s, f = (int(g) for g in m.groups())
                if f >= fps or not _validate_tc_parts(h, mn, s, f):
                    continue
                tc = f"{h:02d}:{mn:02d}:{s:02d}:{f:02d}"
                start = tc_to_frames(tc, fps) - frame_index
                if start >= 0:
                    frame_starts.add(start)
        starts_per_frame.append(frame_starts)

    candidates = sorted({s for fs in starts_per_frame for s in fs})
    best, best_count = None, 0
    for cand in candidates:
        count = sum(
            1 for fs in starts_per_frame if any(abs(s - cand) <= 1 for s in fs)
        )
        if count > best_count:
            best, best_count = cand, count

    if best is None or best_count < 2:
        return None
    return frames_to_tc(best, fps)


def _extract_frame(video_path: str, frame_index: int, region) -> str | None:
    """Извлекает один кадр (по точному номеру) в PNG; при region — кадрирует."""
    out = str(TEMP_DIR / f"tcocr_{uuid.uuid4().hex}.png")
    vf = f"select=eq(n\\,{frame_index})"
    if region:
        left, top, right, bottom = region
        vf += (
            f",crop=iw*{right - left:.4f}:ih*{bottom - top:.4f}:"
            f"iw*{left:.4f}:ih*{top:.4f}"
        )
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-y", "-i", video_path,
             "-vf", vf, "-frames:v", "1", "-an", out],
            capture_output=True,
            timeout=60,
        )
        if result.returncode == 0 and Path(out).exists():
            return out
    except Exception as e:
        logger.warning("Не удалось извлечь кадр %d для OCR ТК: %s", frame_index, e)
    return None


def detect_burned_in_timecode(
    video_path: str,
    fps: int,
    *,
    region: str | None = None,
    sample_seconds: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> str | None:
    """Считывает выжженный в кадр ТК и возвращает ТК первого кадра файла (или None)."""
    reader = _get_reader()
    if reader is None:
        return None

    reg = _parse_region(region if region is not None else OCR_TIMECODE_REGION)

    samples: list[tuple[int, list[str]]] = []
    tmp_files: list[str] = []
    try:
        for sec in sample_seconds:
            n = int(round(sec * fps))
            frame = _extract_frame(video_path, n, reg)
            if not frame:
                continue
            tmp_files.append(frame)
            try:
                texts = reader.readtext(frame, detail=0, allowlist="0123456789:;. ")
            except Exception as e:
                logger.warning("OCR кадра не удался: %s", e)
                continue
            samples.append((n, texts))
    finally:
        for f in tmp_files:
            try:
                Path(f).unlink()
            except OSError:
                pass

    tc = parse_start_timecode_from_ocr(samples, fps)
    if tc:
        logger.info("Выжженный в кадр ТК распознан (OCR): %s", tc)
    else:
        logger.info("Выжженный в кадр ТК распознать не удалось.")
    return tc
