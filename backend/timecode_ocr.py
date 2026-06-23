"""Чтение выжженного в кадр SMPTE-таймкода через OCR (easyocr).

Используется, когда стартовый таймкод не задан ни в имени файла, ни в
метаданных контейнера. Извлекает несколько кадров в начале записи, распознаёт
область с таймкодом и по согласованности между кадрами вычисляет таймкод
ПЕРВОГО кадра файла (от которого затем отсчитываются реплики).

Надёжность:
- крупные цифры с широкими пробелами вокруг «:» OCR часто возвращает
  ОТДЕЛЬНЫМИ токенами (["16","39","57","11"]) — поэтому кандидаты ищем и в
  каждом токене, и в их склейке, а разделитель в регэкспе необязателен (0+);
- голосуем по СТАРТУ с точностью до секунды (вывод всё равно с кадрами :00),
  принимаем значение, согласованное минимум в 2 кадрах;
- если по заданной области ничего не нашли — повторяем OCR по всему кадру;
- подробно логируем сырой текст OCR по кадрам, чтобы при сбое донастроить.
"""
from __future__ import annotations

import re
import subprocess
import threading
import uuid
from pathlib import Path

from backend.config import OCR_TIMECODE_REGION, TEMP_DIR, logger
from backend.utils import _validate_tc_parts, frames_to_tc

# ТК в распознанном тексте: разделители OCR часто путает (':', ';', '.', пробел)
# или теряет вовсе — разделитель между парами цифр необязателен (0+ символов),
# чтобы ловить и «16 : 39 : 57 : 11», и «16395711», и склейку отдельных токенов.
_TC_IN_TEXT_RE = re.compile(r"(\d{2})[:;.\s]*(\d{2})[:;.\s]*(\d{2})[:;.\s]*(\d{2})")

# Секунды от начала файла, на которых пробуем прочитать ТК. Шире окно — на случай
# чёрного лидера/слейта в самом начале (ТК появляется не сразу).
_DEFAULT_SAMPLE_SECONDS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0)

_OCR_ALLOWLIST = "0123456789:;. "

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


def _candidates_from_text(text: str, fps: int) -> set[int]:
    """Все валидные «подразумеваемые секунды ТК» из одной строки.

    Возвращает множество h*3600+m*60+s для каждого распознанного в строке ТК
    (поле кадров f отбрасывается — голосуем по секундам). Кандидаты с f>=fps
    считаем мисридом и пропускаем.
    """
    out: set[int] = set()
    for m in _TC_IN_TEXT_RE.finditer(text):
        h, mn, s, f = (int(g) for g in m.groups())
        if f >= fps or not _validate_tc_parts(h, mn, s, f):
            continue
        out.add(h * 3600 + mn * 60 + s)
    return out


def parse_start_timecode_from_ocr(
    samples: list[tuple[int, list[str]]], fps: int
) -> str | None:
    """Чистая логика: по OCR-результатам кадров вычислить ТК первого кадра файла.

    ``samples`` — список ``(индекс_кадра, [распознанные_строки])``. Для каждого
    кадра считаем «подразумеваемый старт» (в секундах) = секунды ТК − секунды
    кадра; берём старт, согласованный минимум в 2 кадрах. Кандидаты ищем и в
    отдельных токенах, и в их склейке (крупные цифры OCR дробит на токены).
    """
    if fps <= 0:
        return None

    starts_per_frame: list[set[int]] = []
    for frame_index, texts in samples:
        frame_offset_s = round(frame_index / fps)
        tc_seconds: set[int] = set()
        # И каждая строка, и склейка всех строк кадра (через пробел и без).
        joined = " ".join(texts)
        for candidate_text in (*texts, joined, joined.replace(" ", "")):
            tc_seconds |= _candidates_from_text(candidate_text, fps)
        frame_starts = {sec - frame_offset_s for sec in tc_seconds if sec - frame_offset_s >= 0}
        starts_per_frame.append(frame_starts)

    candidates = sorted({s for fs in starts_per_frame for s in fs})
    best, best_count = None, 0
    for cand in candidates:
        count = sum(1 for fs in starts_per_frame if cand in fs)
        if count > best_count:
            best, best_count = cand, count

    if best is None or best_count < 2:
        return None
    return frames_to_tc(best * fps, fps)


def _extract_frames(
    video_path: str, frame_indices: list[int], region
) -> list[tuple[int, str]]:
    """Извлекает кадры по точным номерам ОДНИМ проходом ffmpeg.

    Возвращает список ``(индекс_кадра, путь_png)`` в порядке возрастания номеров.
    При region кадрирует и увеличивает кроп ×3 (крупный текст распознаётся точнее).
    """
    if not frame_indices:
        return []
    idxs = sorted(set(frame_indices))
    select_expr = "+".join(f"eq(n\\,{n})" for n in idxs)
    vf = f"select='{select_expr}'"
    if region:
        left, top, right, bottom = region
        vf += (
            f",crop=iw*{right - left:.4f}:ih*{bottom - top:.4f}:"
            f"iw*{left:.4f}:ih*{top:.4f},scale=iw*3:ih*3"
        )
    prefix = f"tcocr_{uuid.uuid4().hex}"
    out_pattern = str(TEMP_DIR / f"{prefix}_%03d.png")
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-y", "-i", video_path,
             "-vf", vf, "-vsync", "0", "-frames:v", str(len(idxs)), "-an", out_pattern],
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.warning("ffmpeg не смог извлечь кадры для OCR ТК (код %d)", result.returncode)
            return []
    except Exception as e:
        logger.warning("Не удалось извлечь кадры для OCR ТК: %s", e)
        return []

    # ffmpeg нумерует вышедшие кадры подряд (001, 002, …) в порядке select.
    pairs: list[tuple[int, str]] = []
    for i, n in enumerate(idxs, start=1):
        path = str(TEMP_DIR / f"{prefix}_{i:03d}.png")
        if Path(path).exists():
            pairs.append((n, path))
    return pairs


def _ocr_frames(reader, pairs: list[tuple[int, str]]) -> list[tuple[int, list[str]]]:
    """OCR каждого извлечённого кадра; логирует сырой текст (для диагностики)."""
    samples: list[tuple[int, list[str]]] = []
    for frame_index, path in pairs:
        try:
            texts = reader.readtext(path, detail=0, allowlist=_OCR_ALLOWLIST)
        except Exception as e:
            logger.warning("OCR кадра %d не удался: %s", frame_index, e)
            continue
        logger.info("[OCR ТК] кадр %d: %r", frame_index, texts)
        samples.append((frame_index, texts))
    return samples


def detect_burned_in_timecode(
    video_path: str,
    fps: int,
    *,
    region: str | None = None,
    sample_seconds: tuple[float, ...] = _DEFAULT_SAMPLE_SECONDS,
    warnings: list[str] | None = None,
) -> str | None:
    """Считывает выжженный в кадр ТК и возвращает ТК первого кадра файла (или None).

    При недоступности easyocr или нераспознанном ТК добавляет понятное
    предупреждение в ``warnings`` (показывается в UI), т.к. OCR — единственный
    автоматический источник абсолютного ТК для .wmv.
    """
    reader = _get_reader()
    if reader is None:
        msg = ("easyocr не установлен — выжженный в кадр таймкод не распознан "
               "(установите: pip install -r requirements.txt).")
        logger.error(msg)
        if warnings is not None:
            warnings.append(msg)
        return None

    if fps <= 0:
        fps = 25
    frame_indices = [max(1, int(round(sec * fps))) for sec in sample_seconds]
    reg = _parse_region(region if region is not None else OCR_TIMECODE_REGION)

    tmp_files: list[str] = []
    try:
        pairs = _extract_frames(video_path, frame_indices, reg)
        tmp_files.extend(p for _, p in pairs)
        samples = _ocr_frames(reader, pairs)
        tc = parse_start_timecode_from_ocr(samples, fps)

        # Fallback: ничего не нашли по области — пробуем весь кадр.
        if tc is None and reg is not None:
            logger.info("[OCR ТК] по области пусто — повтор OCR по всему кадру")
            pairs_full = _extract_frames(video_path, frame_indices, None)
            tmp_files.extend(p for _, p in pairs_full)
            samples_full = _ocr_frames(reader, pairs_full)
            tc = parse_start_timecode_from_ocr(samples_full, fps)
    finally:
        for f in tmp_files:
            try:
                Path(f).unlink()
            except OSError:
                pass

    if tc:
        logger.info("Выжженный в кадр ТК распознан (OCR): %s", tc)
    else:
        msg = ("Выжженный в кадр таймкод не распознан (OCR) — отсчёт с 00:00:00:00. "
               "Проверьте, что ТК виден в первых 15 секундах; область задаётся "
               "OCR_TIMECODE_REGION.")
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
    return tc
