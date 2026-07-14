"""Видео-описания технических моментов (Gemini Vision).

Человеческие эталоны снабжают маркеры техпауз описанием происходящего в кадре:
«(Технические моменты. Съемка нарезки помидоров двумя поварами)». Из аудио это
не восстановить — модуль извлекает 1–2 кадра из интервала каждого маркера
(ffmpeg, один проход на все маркеры) и просит мультимодальный Gemini дать
короткое описание в стиле монтажного листа.

Деградация мягкая: нет ключа/ffmpeg/видео или сбой — маркеры остаются «голыми»,
задача не падает (описания — украшение, не данные).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from backend.config import (
    GEMINI_MODEL,
    TECH_VISION_MAX_MARKERS,
    TECH_VISION_MIN_GAP_SECONDS,
    logger,
)
from backend.utils import tc_to_frames

_MARKER_PREFIX = "(Технические моменты"
_MAX_DESC_CHARS = 80
_BATCH_MARKERS = 8
_JPEG_MAX_WIDTH = 768


def _marker_intervals(
    segments: list[dict],
    fps: int,
    start_frames: int,
    *,
    min_gap_s: float = TECH_VISION_MIN_GAP_SECONDS,
    max_markers: int = TECH_VISION_MAX_MARKERS,
) -> list[tuple[int, float, float]]:
    """Интервалы маркеров техмоментов: (индекс сегмента, start_s, end_s).

    Смещение — из таймкода маркера относительно стартового ТК файла; конец —
    таймкод следующего сегмента (для последнего +10 c). Короче ``min_gap_s`` —
    пропускаются; максимум ``max_markers`` первых.
    """
    if fps <= 0:
        return []
    out: list[tuple[int, float, float]] = []
    for i, seg in enumerate(segments):
        if not str(seg.get("text", "")).startswith(_MARKER_PREFIX):
            continue
        start_s = (tc_to_frames(seg.get("timecode", ""), fps) - start_frames) / fps
        if start_s < 0:
            continue
        if i + 1 < len(segments):
            end_s = (tc_to_frames(segments[i + 1].get("timecode", ""), fps) - start_frames) / fps
        else:
            end_s = start_s + 10.0
        if end_s - start_s < min_gap_s:
            continue
        out.append((i, start_s, end_s))
        if len(out) >= max_markers:
            break
    return out


def _frame_seconds(intervals: list[tuple[int, float, float]]) -> dict[int, list[float]]:
    """Секунды кадров для каждого маркера: середина; на длинных (>20 c) ещё 1/4."""
    out: dict[int, list[float]] = {}
    for idx, start_s, end_s in intervals:
        dur = end_s - start_s
        secs = [start_s + dur / 2]
        if dur > 20:
            secs.insert(0, start_s + dur / 4)
        out[idx] = secs
    return out


def _clean_description(desc: str) -> str:
    """Нормализация описания: одна строка, без кавычек, ≤80 символов, заглавная."""
    desc = re.sub(r"\s+", " ", str(desc or "")).strip().strip('"«»\'')
    desc = desc.rstrip(".").strip()
    if not desc or not any(ch.isalpha() for ch in desc):
        return ""
    if len(desc) > _MAX_DESC_CHARS:
        cut = desc[:_MAX_DESC_CHARS]
        desc = cut[:cut.rfind(" ")] if " " in cut else cut
    return desc[0].upper() + desc[1:]


def _apply_descriptions(segments: list[dict], descriptions: dict[int, str]) -> list[dict]:
    """Вернуть копию segments с маркерами, обогащёнными описаниями."""
    out = []
    for i, seg in enumerate(segments):
        desc = _clean_description(descriptions.get(i, "")) if i in descriptions else ""
        if desc and str(seg.get("text", "")).startswith(_MARKER_PREFIX):
            out.append({**seg, "text": f"(Технические моменты. {desc})"})
        else:
            out.append(seg)
    return out


def _encode_jpeg(png_path: str) -> tuple[bytes, str] | None:
    """PNG-кадр → (bytes, mime) для Vision: JPEG ≤768px (cv2) или PNG как есть."""
    try:
        import cv2
        img = cv2.imread(png_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > _JPEG_MAX_WIDTH:
            nh = int(h * _JPEG_MAX_WIDTH / w)
            img = cv2.resize(img, (_JPEG_MAX_WIDTH, nh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return None
        return bytes(buf), "image/jpeg"
    except ImportError:
        try:
            return Path(png_path).read_bytes(), "image/png"
        except OSError:
            return None


def _image_part(data: bytes, mime: str):
    """Часть contents с изображением (новый SDK google-genai)."""
    from google.genai import types
    return types.Part.from_bytes(data=data, mime_type=mime)


def _vision_generate(contents: list) -> str | None:
    """Мультимодальный вызов Gemini; None — клиент недоступен."""
    import backend.postprocess as pp
    if not pp._gemini_ready():
        return None
    response = pp._gemini_client.models.generate_content(
        model=GEMINI_MODEL, contents=contents, config={"temperature": 0.0},
    )
    return response.text


def gemini_describe_frames(
    marker_images: list[tuple[int, list[tuple[bytes, str]]]],
    *,
    heroes: list[str] | None = None,
    description: str | None = None,
    speech_context: dict[int, tuple[str, str]] | None = None,
) -> dict[int, str]:
    """Описать кадры маркеров через Gemini Vision.

    ``marker_images`` — [(id маркера, [(bytes, mime), …]), …]. Возвращает
    {id: описание}; сбой любого батча — просто пропуск его маркеров.
    ``speech_context`` — {id: (текст до, текст после)}: обрывки реплик вокруг
    паузы, чтобы модель могла отразить и аудио-контекст («герои говорят на
    итальянском»), которого на кадрах не видно.
    """
    results: dict[int, str] = {}
    speech_context = speech_context or {}
    heroes_line = ""
    if heroes:
        heroes_line = (f"Известные участники съёмки: {', '.join(heroes)}. "
                       "Называй их по имени, только если уверен по кадру.\n")
    context_line = f"Контекст съёмки: {description.strip()}\n" if description and description.strip() else ""

    for chunk_start in range(0, len(marker_images), _BATCH_MARKERS):
        chunk = marker_images[chunk_start:chunk_start + _BATCH_MARKERS]
        contents: list = []
        for marker_id, images in chunk:
            contents.append(f"Кадры технической паузы №{marker_id}:")
            before, after = speech_context.get(marker_id, ("", ""))
            if before or after:
                contents.append(
                    f"Речь вокруг паузы №{marker_id}: «…{before}» → «{after}…»"
                )
            for data, mime in images:
                contents.append(_image_part(data, mime))
        contents.append(
            "Это кадры технических пауз телевизионной съёмки.\n"
            f"{context_line}{heroes_line}"
            "Для КАЖДОГО номера паузы дай описание происходящего в кадре — "
            "3–10 слов в стиле монтажного листа. Предпочитай форму «Съемка …»: "
            "«Съемка нарезки помидоров двумя поварами», «Съемка подготовки "
            "рабочей поверхности»; когда «Съемка» не подходит — отглагольное "
            "существительное («Исполнение песни у микрофона»). Если из речи "
            "вокруг паузы очевидно, что участники говорят на иностранном языке "
            "— отрази это («Герои говорят между собой на итальянском»). "
            'Верни СТРОГО JSON-объект вида {"<номер>": "описание"}. '
            "Никакого текста кроме JSON."
        )
        try:
            raw = _vision_generate(contents)
        except Exception as e:
            logger.warning("[Vision ТМ] сбой батча описаний: %s", e)
            continue
        if not raw:
            continue
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            continue
        try:
            data = json.loads(match.group(0))
        except (ValueError, TypeError):
            continue
        if not isinstance(data, dict):
            continue
        valid_ids = {mid for mid, _ in chunk}
        for key, value in data.items():
            try:
                mid = int(key)
            except (ValueError, TypeError):
                continue
            if mid in valid_ids and isinstance(value, str):
                results[mid] = value
    return results


def _speech_context(segments: list[dict], marker_idx: int, span: int = 120) -> tuple[str, str]:
    """Обрывки речи вокруг маркера: хвост предыдущей реплики и начало следующей."""
    before = after = ""
    for j in range(marker_idx - 1, -1, -1):
        text = str(segments[j].get("text", ""))
        if text and not text.startswith(_MARKER_PREFIX):
            before = text[-span:]
            break
    for j in range(marker_idx + 1, len(segments)):
        text = str(segments[j].get("text", ""))
        if text and not text.startswith(_MARKER_PREFIX):
            after = text[:span]
            break
    return before, after


def annotate_tech_markers(
    segments: list[dict],
    video_path: str,
    fps: int,
    start_frames: int,
    *,
    heroes: list[str] | None = None,
    description: str | None = None,
) -> list[dict]:
    """Обогатить маркеры техмоментов описаниями кадров. Никогда не бросает."""
    tmp_files: list[str] = []
    try:
        intervals = _marker_intervals(segments, fps, start_frames)
        total_markers = sum(
            1 for s in segments if str(s.get("text", "")).startswith(_MARKER_PREFIX)
        )
        if not intervals:
            return segments

        frame_secs = _frame_seconds(intervals)
        # Кадр → маркер; один проход ffmpeg на все маркеры (как в OCR ТК).
        frame_to_marker: dict[int, int] = {}
        for idx, secs in frame_secs.items():
            for sec in secs:
                frame_to_marker[max(1, int(round(sec * fps)))] = idx

        from backend.timecode_ocr import _extract_frames
        pairs = _extract_frames(video_path, sorted(frame_to_marker), None)
        tmp_files.extend(p for _, p in pairs)
        if not pairs:
            logger.info("[Vision ТМ] кадры не извлечены — маркеры без описаний")
            return segments

        images_by_marker: dict[int, list[tuple[bytes, str]]] = {}
        for frame_idx, path in pairs:
            marker_idx = frame_to_marker.get(frame_idx)
            if marker_idx is None:
                continue
            encoded = _encode_jpeg(path)
            if encoded is not None:
                images_by_marker.setdefault(marker_idx, []).append(encoded)

        if not images_by_marker:
            return segments
        descriptions = gemini_describe_frames(
            sorted(images_by_marker.items()), heroes=heroes, description=description,
            speech_context={idx: _speech_context(segments, idx) for idx in images_by_marker},
        )
        logger.info("[Vision ТМ] описано %d из %d маркеров (интервалов ≥%.0f c: %d)",
                    len(descriptions), total_markers, TECH_VISION_MIN_GAP_SECONDS,
                    len(intervals))
        if not descriptions:
            return segments
        return _apply_descriptions(segments, descriptions)
    except Exception as e:
        logger.warning("[Vision ТМ] описания пропущены из-за ошибки: %s", e)
        return segments
    finally:
        for f in tmp_files:
            try:
                Path(f).unlink()
            except OSError:
                pass
