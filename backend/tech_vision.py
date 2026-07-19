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
import subprocess
from pathlib import Path

from backend.config import (
    GEMINI_MODEL,
    GEMINI_TIMEOUT_SECONDS,
    LOCAL_LLM_API_KEY,
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_VISION_MODEL,
    TECH_VISION_MAX_MARKERS,
    TECH_VISION_MIN_GAP_SECONDS,
    logger,
)
from backend.utils import offset_tc, tc_to_frames

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
    media_s_fn=None,
) -> list[tuple[int, float, float]]:
    """Интервалы маркеров техмоментов: (индекс сегмента, start_s, end_s).

    Смещение — из таймкода маркера относительно стартового ТК файла; конец —
    таймкод следующего сегмента (для последнего +10 c). Короче ``min_gap_s`` —
    пропускаются; максимум ``max_markers`` первых. ``media_s_fn`` — обратный
    маппер «кадры ТК → секунды медиа» при кусочной коррекции по OCR-якорям
    (free-run ТК: линейная формула дала бы кадры не из того места).
    """
    if fps <= 0:
        return []

    def to_media(tc_str: str) -> float:
        frames = tc_to_frames(tc_str, fps)
        if media_s_fn is not None:
            return media_s_fn(frames)
        return (frames - start_frames) / fps

    out: list[tuple[int, float, float]] = []
    for i, seg in enumerate(segments):
        if not str(seg.get("text", "")).startswith(_MARKER_PREFIX):
            continue
        start_s = to_media(seg.get("timecode", ""))
        if start_s < 0:
            continue
        if i + 1 < len(segments):
            end_s = to_media(segments[i + 1].get("timecode", ""))
        else:
            end_s = start_s + 10.0
        if end_s - start_s < min_gap_s:
            continue
        out.append((i, start_s, end_s))
        if len(out) >= max_markers:
            break
    return out


def _detect_scene_cuts(video_path: str, start_s: float, dur_s: float,
                       threshold: float) -> list[float]:
    """Абсолютные медиа-секунды смен плана внутри [start_s, start_s + dur_s].

    ffmpeg декодирует только нужный кусок (input-seek + ``-t``); ``showinfo``
    печатает pts_time отобранных фильтром ``scene`` кадров в stderr. pts после
    ``-ss`` отсчитывается от нуля куска — прибавляем start_s.
    """
    cmd = [
        "ffmpeg", "-v", "info", "-nostats",
        "-ss", f"{max(0.0, start_s):.3f}", "-t", f"{dur_s:.3f}", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo", "-an", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as e:
        logger.warning("[Сцены ТМ] ffmpeg не отработал (%.0f c): %s", start_s, e)
        return []
    return [start_s + float(m.group(1))
            for m in re.finditer(r"pts_time:\s*([0-9.]+)", result.stderr)]


def split_markers_by_scenes(
    segments: list[dict],
    video_path: str,
    fps: int,
    start_frames: int,
    *,
    tc_fn=None,
    media_s_fn=None,
    threshold: float = 0.30,
    min_span_s: float = 25.0,
    min_sub_s: float = 8.0,
) -> list[dict]:
    """Разрезать длинные интервалы техмоментов по сменам плана в видеоряде.

    Эталонные монтажные листы ставят маркер на КАЖДЫЙ дубль/смену плана
    (~17 маркеров в репетиционной части ф4), а аудио границ дублей не слышит —
    их видно только по видеоряду. Для каждого маркера с интервалом ≥
    ``min_span_s`` детектируются смены плана (ffmpeg scene), и на каждой
    вставляется дополнительный «голый» маркер (позже vision опишет каждый кусок
    отдельно). Куски короче ``min_sub_s`` не плодятся. Никогда не бросает.
    """
    try:
        intervals = _marker_intervals(
            segments, fps, start_frames,
            min_gap_s=min_span_s, max_markers=len(segments) or 1,
            media_s_fn=media_s_fn,
        )
        if not intervals:
            return segments

        def tc_of(seconds: float) -> str:
            if tc_fn is not None:
                return tc_fn(seconds)
            return offset_tc(start_frames, seconds, fps)

        extra: dict[int, list[dict]] = {}
        total_new = 0
        for idx, start_s, end_s in intervals:
            cuts = _detect_scene_cuts(video_path, start_s, end_s - start_s, threshold)
            kept: list[float] = []
            prev = start_s
            for cut in sorted(cuts):
                if cut - prev >= min_sub_s and end_s - cut >= min_sub_s:
                    kept.append(cut)
                    prev = cut
            if kept:
                base = segments[idx]
                extra[idx] = [
                    {**base, "timecode": tc_of(cut)} for cut in kept
                ]
                total_new += len(kept)
        if not extra:
            return segments

        out: list[dict] = []
        for i, seg in enumerate(segments):
            out.append(seg)
            out.extend(extra.get(i, []))
        logger.info("[Сцены ТМ] длинных интервалов: %d, добавлено маркеров по сменам плана: %d",
                    len(intervals), total_new)
        return out
    except Exception as e:
        logger.warning("[Сцены ТМ] сбой разрезания по сценам: %s — маркеры без изменений", e)
        return segments


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
    """Часть contents с изображением: genai-Part (облако) или OpenAI-словарь
    с data-URI (локальная мультимодальная модель)."""
    if LOCAL_LLM_BASE_URL:
        import base64
        b64 = base64.b64encode(data).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
    from google.genai import types
    return types.Part.from_bytes(data=data, mime_type=mime)


def _vision_generate(contents: list) -> str | None:
    """Мультимодальный вызов (Gemini или локальная LLM); None — недоступно."""
    if LOCAL_LLM_BASE_URL:
        if not LOCAL_LLM_VISION_MODEL:
            # Локальный режим без мультимодальной модели: vision мягко выключен.
            return None
        import requests
        message_content = [
            part if isinstance(part, dict) else {"type": "text", "text": str(part)}
            for part in contents
        ]
        resp = requests.post(
            f"{LOCAL_LLM_BASE_URL}/chat/completions",
            json={
                "model": LOCAL_LLM_VISION_MODEL,
                "messages": [{"role": "user", "content": message_content}],
                "temperature": 0.0,
                "stream": False,
            },
            headers={"Authorization": f"Bearer {LOCAL_LLM_API_KEY}"},
            timeout=GEMINI_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        import backend.postprocess as pp
        return pp._THINK_BLOCK_RE.sub("", resp.json()["choices"][0]["message"]["content"] or "")
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
    inside_speech: dict[int, str] | None = None,
) -> dict[int, str]:
    """Описать кадры маркеров через Gemini Vision.

    ``marker_images`` — [(id маркера, [(bytes, mime), …]), …]. Возвращает
    {id: описание}; сбой любого батча — просто пропуск его маркеров.
    ``speech_context`` — {id: (текст до, текст после)}: обрывки реплик вокруг
    паузы, чтобы модель могла отразить и аудио-контекст («герои говорят на
    итальянском»), которого на кадрах не видно. ``inside_speech`` — {id: что
    ЗВУЧАЛО внутри паузы (исходные тексты свёрнутых реплик)}: без него модель
    по одному кадру выдумывала происходящее (ф4: «фотограф» вместо исполнения
    Happy birthday).
    """
    results: dict[int, str] = {}
    speech_context = speech_context or {}
    inside_speech = inside_speech or {}
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
            inside = inside_speech.get(marker_id, "")
            if inside:
                contents.append(f"В паузе №{marker_id} звучало: «{inside}»")
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
            "существительное («Исполнение песни у микрофона»). Опирайся на то, "
            "что ЗВУЧАЛО в паузе: если звучал текст песни — это исполнение "
            "песни («Исполнение песни Happy birthday у микрофона»), если "
            "обсуждали дубль — репетиция. НЕ выдумывай людей и действия, "
            "которых не видно в кадре и не слышно в звучавшем. Если из речи "
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
    media_s_fn=None,
    folded_speech: list[tuple[float, str]] | None = None,
) -> list[dict]:
    """Обогатить маркеры техмоментов описаниями кадров. Никогда не бросает."""
    tmp_files: list[str] = []
    try:
        intervals = _marker_intervals(segments, fps, start_frames, media_s_fn=media_s_fn)
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
        # «Что звучало в паузе»: свёрнутые исходные реплики интервала маркера.
        interval_by_idx = {idx: (s, e) for idx, s, e in intervals}
        inside_speech: dict[int, str] = {}
        for idx in images_by_marker:
            span = interval_by_idx.get(idx)
            if span is None:
                continue
            texts = [t for sec, t in (folded_speech or []) if span[0] <= sec < span[1]]
            if texts:
                inside_speech[idx] = " ".join(texts)[:200]
        descriptions = gemini_describe_frames(
            sorted(images_by_marker.items()), heroes=heroes, description=description,
            speech_context={idx: _speech_context(segments, idx) for idx in images_by_marker},
            inside_speech=inside_speech,
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
