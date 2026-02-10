import asyncio
import shutil
import time
from pathlib import Path

import ffmpeg
import grpc
import requests

from backend.config import (
    MAX_CONCURRENT_TASKS,
    MAX_FILE_SIZE_BYTES,
    OUTPUT_DIR,
    TEMP_DIR,
    YANDEX_API_KEY,
    logger,
)
from backend.models import ProjectStatusEnum
from backend.utils import (
    detect_fps,
    frames_to_tc,
    parse_filename_metadata,
    strip_extension,
    tc_to_frames,
    validate_file_extension,
)

# --- In-memory storage ---
projects_db: dict = {}

# Semaphore to limit concurrent heavy tasks
_task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# --- SpeechKit gRPC v3 ---
SPEECHKIT_GRPC_HOST = "stt.api.cloud.yandex.net:443"
GRPC_CHUNK_SIZE = 4000  # 4 KB chunks for streaming audio
GRPC_TIMEOUT = 7200  # 2 hours max for recognition


def _download_from_yadisk(project_id: str, disk_url: str, local_video_path) -> str:
    """Скачивает файл с Яндекс.Диска. Возвращает оригинальное имя файла."""
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    resp = requests.get(api_url, params={"public_key": disk_url}, timeout=30)
    resp.raise_for_status()
    download_url = resp.json()["href"]

    original_filename = "video_source.mp4"
    try:
        meta_url = "https://cloud-api.yandex.net/v1/disk/public/resources"
        meta_resp = requests.get(meta_url, params={"public_key": disk_url}, timeout=15)
        if meta_resp.status_code == 200:
            meta_data = meta_resp.json()
            original_filename = meta_data.get("name", original_filename)
            file_size = meta_data.get("size", 0)
            if file_size > MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"Файл слишком большой ({file_size / (1024**3):.1f} ГБ). "
                    f"Максимум: {MAX_FILE_SIZE_BYTES / (1024**3):.0f} ГБ."
                )
    except requests.RequestException as e:
        logger.warning("[%s] Не удалось получить метаданные файла: %s", project_id[:8], e)

    ext_error = validate_file_extension(original_filename)
    if ext_error:
        raise ValueError(ext_error)

    downloaded_size = 0
    with requests.get(download_url, stream=True, timeout=600) as r:
        r.raise_for_status()
        content_length = int(r.headers.get("content-length", 0))
        with open(local_video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded_size += len(chunk)
                if content_length > 0:
                    pct = min(int(downloaded_size / content_length * 100), 100)
                    projects_db[project_id]["progress_percent"] = pct

    logger.info("[%s] Файл скачан: %s", project_id[:8], original_filename)
    return original_filename


def _convert_to_opus(project_id: str, input_path, output_path):
    """Конвертирует видео/аудио в OPUS."""
    logger.info("[%s] Конвертация в OPUS...", project_id[:8])
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), acodec="libopus", ac=1, ar=48000)
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info("[%s] Конвертация завершена.", project_id[:8])


# ==================== SpeechKit API v3 (gRPC) ====================


def _generate_recognition_requests(audio_path):
    """Генератор gRPC-запросов: сначала настройки сессии, затем чанки аудио."""
    from yandex.cloud.ai.stt.v3 import stt_pb2

    recognize_options = stt_pb2.StreamingOptions(
        recognition_model=stt_pb2.RecognitionModelOptions(
            model="general",
            audio_format=stt_pb2.AudioFormatOptions(
                container_audio=stt_pb2.ContainerAudio(
                    container_audio_type=stt_pb2.ContainerAudio.OGG_OPUS,
                )
            ),
            text_normalization=stt_pb2.TextNormalizationOptions(
                text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                profanity_filter=False,
                literature_text=True,
            ),
            language_restriction=stt_pb2.LanguageRestrictionOptions(
                restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                language_code=["ru-RU"],
            ),
            audio_processing_type=stt_pb2.RecognitionModelOptions.FULL_DATA,
        ),
        speaker_labeling=stt_pb2.SpeakerLabelingOptions(
            speaker_labeling=stt_pb2.SpeakerLabelingOptions.SPEAKER_LABELING_ENABLED,
        ),
    )
    yield stt_pb2.StreamingRequest(session_options=recognize_options)

    # Stream audio file in chunks
    with open(str(audio_path), "rb") as f:
        while True:
            data = f.read(GRPC_CHUNK_SIZE)
            if not data:
                break
            yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=data))


def _transcribe_with_speechkit(project_id: str, audio_path) -> list[dict]:
    """Распознавание через SpeechKit API v3 (gRPC) с диаризацией спикеров.

    Стримит локальный аудиофайл напрямую в SpeechKit (S3 не нужен).
    Возвращает список сегментов с channel_tag (0 или 1) для каждого спикера.
    """
    if not YANDEX_API_KEY:
        raise RuntimeError("YANDEX_API_KEY не задан. Распознавание невозможно.")

    from yandex.cloud.ai.stt.v3 import stt_service_pb2_grpc

    cred = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel(SPEECHKIT_GRPC_HOST, cred)
    stub = stt_service_pb2_grpc.RecognizerStub(channel)

    logger.info("[%s] Стримим аудио в SpeechKit v3 (gRPC с диаризацией)...", project_id[:8])

    try:
        responses = stub.RecognizeStreaming(
            _generate_recognition_requests(audio_path),
            metadata=[("authorization", f"Api-Key {YANDEX_API_KEY}")],
            timeout=GRPC_TIMEOUT,
        )

        segments = []
        for r in responses:
            event_type = r.WhichOneof("Event")
            if event_type == "final_refinement":
                alts = r.final_refinement.normalized_text.alternatives
                if not alts:
                    continue
                alt = alts[0]
                text = alt.text
                words = list(alt.words)
                if not words:
                    continue
                segments.append({
                    "text": text,
                    "channel_tag": r.channel_tag,
                    "start_ms": words[0].start_time_ms,
                    "end_ms": words[-1].end_time_ms,
                    "words": [
                        {
                            "text": w.text,
                            "start_ms": w.start_time_ms,
                            "end_ms": w.end_time_ms,
                        }
                        for w in words
                    ],
                })

        logger.info(
            "[%s] Распознавание завершено. Сегментов: %d",
            project_id[:8], len(segments),
        )
        return segments

    except grpc.RpcError as e:
        logger.error(
            "[%s] gRPC ошибка: code=%s, details=%s",
            project_id[:8], e.code(), e.details(),
        )
        raise RuntimeError(f"SpeechKit gRPC: {e.details()}") from e
    finally:
        channel.close()


def _process_recognition_result(project_id: str, segments: list[dict], original_filename: str, video_path):
    """Обрабатывает результат распознавания v3 и сохраняет в projects_db."""
    meta = parse_filename_metadata(original_filename)
    projects_db[project_id]["original_filename"] = original_filename

    fps = detect_fps(str(video_path)) if video_path.exists() else 25

    speaker_durations: dict[str, float] = {}
    raw_segments = []
    start_frames = tc_to_frames(meta["start_tc"], fps)

    for seg in segments:
        channel = str(seg["channel_tag"])
        text = seg["text"]
        words = seg["words"]
        if not words:
            continue

        start_s = words[0]["start_ms"] / 1000.0
        end_s = words[-1]["end_ms"] / 1000.0

        dur = end_s - start_s
        speaker_durations[channel] = speaker_durations.get(channel, 0) + dur

        abs_frames = start_frames + int(start_s * fps)
        tc_formatted = frames_to_tc(abs_frames, fps)

        raw_segments.append({
            "timecode": tc_formatted,
            "speaker": channel,
            "text": text,
        })

    detected_speakers = {}
    sorted_voices = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
    file_names = meta["speakers"]

    for i, (voice_id, dur) in enumerate(sorted_voices):
        suggested = f"Спикер {voice_id}"
        if i < len(file_names):
            suggested = file_names[i]

        detected_speakers[voice_id] = {
            "duration_sec": round(dur, 1),
            "suggested_name": suggested,
        }

    projects_db[project_id]["result"] = {
        "segments": raw_segments,
        "speakers": detected_speakers,
        "meta": {**meta, "original_filename": original_filename},
    }
    projects_db[project_id]["fps"] = fps

    logger.info(
        "[%s] Обработка завершена. Сегментов: %d, Спикеров: %d, FPS: %d",
        project_id[:8], len(raw_segments), len(detected_speakers), fps,
    )


# ==================== Task functions ====================


def process_video_task(project_id: str, disk_url: str):
    """Фоновая задача: скачивание -> конвертация -> распознавание с диаризацией."""
    local_video_path = TEMP_DIR / f"{project_id}_video"
    local_audio_path = TEMP_DIR / f"{project_id}.opus"

    try:
        # 1. СКАЧИВАНИЕ
        projects_db[project_id]["status"] = ProjectStatusEnum.DOWNLOADING
        projects_db[project_id]["progress_percent"] = 0
        logger.info("[%s] Скачивание файла с Яндекс.Диска...", project_id[:8])
        original_filename = _download_from_yadisk(project_id, disk_url, local_video_path)

        # 2. КОНВЕРТАЦИЯ
        projects_db[project_id]["status"] = ProjectStatusEnum.CONVERTING
        projects_db[project_id]["progress_percent"] = None
        _convert_to_opus(project_id, local_video_path, local_audio_path)

        # 3. РАСПОЗНАВАНИЕ (gRPC v3 с диаризацией, S3 не нужен)
        projects_db[project_id]["status"] = ProjectStatusEnum.TRANSCRIBING
        logger.info("[%s] Распознавание с диаризацией...", project_id[:8])
        segments = _transcribe_with_speechkit(project_id, local_audio_path)

        # 4. ОБРАБОТКА РЕЗУЛЬТАТА
        _process_recognition_result(project_id, segments, original_filename, local_video_path)
        projects_db[project_id]["status"] = ProjectStatusEnum.COMPLETED

    except Exception as e:
        logger.exception("[%s] Ошибка обработки: %s", project_id[:8], e)
        projects_db[project_id]["status"] = ProjectStatusEnum.ERROR
        projects_db[project_id]["error"] = str(e)

    finally:
        for path in (local_video_path, local_audio_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass


def process_uploaded_file_task(project_id: str, local_video_path, original_filename: str):
    """Фоновая задача для локально загруженного файла: конвертация -> распознавание с диаризацией."""
    local_audio_path = TEMP_DIR / f"{project_id}.opus"
    local_video_path = Path(local_video_path)

    try:
        projects_db[project_id]["original_filename"] = original_filename

        # 1. КОНВЕРТАЦИЯ
        projects_db[project_id]["status"] = ProjectStatusEnum.CONVERTING
        projects_db[project_id]["progress_percent"] = None
        _convert_to_opus(project_id, local_video_path, local_audio_path)

        # 2. РАСПОЗНАВАНИЕ (gRPC v3 с диаризацией, S3 не нужен)
        projects_db[project_id]["status"] = ProjectStatusEnum.TRANSCRIBING
        logger.info("[%s] Распознавание с диаризацией...", project_id[:8])
        segments = _transcribe_with_speechkit(project_id, local_audio_path)

        # 3. ОБРАБОТКА РЕЗУЛЬТАТА
        _process_recognition_result(project_id, segments, original_filename, local_video_path)
        projects_db[project_id]["status"] = ProjectStatusEnum.COMPLETED
        logger.info("[%s] Файл обработан: %s", project_id[:8], original_filename)

        # 4. АВТОСОХРАНЕНИЕ DOCX НА ДИСК
        try:
            docx_path = str(OUTPUT_DIR / f"autosave_{project_id}.docx")
            saved_name = auto_export_project(project_id, docx_path)
            if saved_name:
                final_path = OUTPUT_DIR / saved_name
                if final_path.exists():
                    final_path = OUTPUT_DIR / f"{strip_extension(saved_name)}_{project_id[:8]}.docx"
                shutil.move(docx_path, str(final_path))
                logger.info("[%s] DOCX сохранён: %s", project_id[:8], final_path.name)
        except Exception as e:
            logger.warning("[%s] Не удалось автосохранить DOCX: %s", project_id[:8], e)

    except Exception as e:
        logger.exception("[%s] Ошибка обработки: %s", project_id[:8], e)
        projects_db[project_id]["status"] = ProjectStatusEnum.ERROR
        projects_db[project_id]["error"] = str(e)

    finally:
        for path in (local_video_path, local_audio_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass


def auto_export_project(project_id: str, output_path: str) -> str | None:
    """Автоматически экспортирует проект в DOCX используя имена спикеров из метаданных файла.
    Возвращает имя файла для скачивания или None при ошибке."""
    from backend.docx_export import generate_docx

    proj = projects_db.get(project_id)
    if not proj or "result" not in proj:
        return None

    speakers = proj["result"].get("speakers", {})
    final_map = {}
    abbr_map = {}

    for speaker_id, info in speakers.items():
        name = info.get("suggested_name", f"Спикер {speaker_id}")
        final_map[speaker_id] = name
        abbr_map[speaker_id] = name[:3].upper() if name else f"С{speaker_id}"

    return generate_docx(proj, final_map, abbr_map, output_path)
