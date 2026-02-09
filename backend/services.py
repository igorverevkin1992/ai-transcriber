import asyncio
import time

import ffmpeg
import requests

from backend.config import (
    MAX_CONCURRENT_TASKS,
    MAX_FILE_SIZE_BYTES,
    TEMP_DIR,
    YANDEX_API_KEY,
    logger,
)
from backend.models import ProjectStatusEnum
from backend.s3 import delete_from_s3, upload_to_s3
from backend.utils import (
    detect_fps,
    frames_to_tc,
    parse_filename_metadata,
    tc_to_frames,
    validate_file_extension,
)

# --- In-memory storage ---
projects_db: dict = {}

# Semaphore to limit concurrent heavy tasks
_task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)


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


def _transcribe_with_speechkit(project_id: str, file_uri: str) -> dict:
    """Отправляет аудио на распознавание и ожидает результат."""
    if not YANDEX_API_KEY:
        raise RuntimeError("YANDEX_API_KEY не задан. Распознавание невозможно.")

    sk_body = {
        "config": {
            "specification": {
                "languageCode": "ru-RU",
                "literature_text": True,
                "profanityFilter": False,
                "audioEncoding": "OGG_OPUS",
                "sampleRateHertz": 48000,
                "audioChannelCount": 1,
            }
        },
        "audio": {"uri": file_uri},
    }

    sk_headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}"}

    for attempt in range(3):
        try:
            sk_resp = requests.post(
                "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize",
                json=sk_body,
                headers=sk_headers,
                timeout=60,
            )
            sk_resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise
            logger.warning("[%s] SpeechKit запрос не удался (попытка %d): %s", project_id[:8], attempt + 1, e)
            time.sleep(2 ** (attempt + 1))

    operation_id = sk_resp.json()["id"]
    logger.info("[%s] Операция SpeechKit: %s", project_id[:8], operation_id)

    poll_interval = 3
    max_polls = 600
    for _ in range(max_polls):
        time.sleep(poll_interval)
        try:
            op_resp = requests.get(
                f"https://operation.api.cloud.yandex.net/operations/{operation_id}",
                headers=sk_headers,
                timeout=30,
            )
            op_data = op_resp.json()
        except requests.RequestException as e:
            logger.warning("[%s] Ошибка при проверке статуса операции: %s", project_id[:8], e)
            continue

        if op_data.get("done"):
            break
    else:
        raise TimeoutError("SpeechKit: превышено время ожидания распознавания.")

    if "error" in op_data:
        raise RuntimeError(f"SpeechKit Error: {op_data['error']}")

    return op_data


def _process_recognition_result(project_id: str, op_data: dict, original_filename: str, video_path):
    """Обрабатывает результат распознавания и сохраняет в projects_db."""
    chunks = op_data["response"]["chunks"]
    meta = parse_filename_metadata(original_filename)
    projects_db[project_id]["original_filename"] = original_filename

    fps = detect_fps(str(video_path)) if video_path.exists() else 25

    speaker_durations: dict[str, float] = {}
    raw_segments = []
    start_frames = tc_to_frames(meta["start_tc"], fps)

    for chunk in chunks:
        channel = str(chunk.get("channelTag", "1"))
        alternatives = chunk.get("alternatives", [])
        if not alternatives:
            continue

        alt = alternatives[0]
        text = alt.get("text", "")
        words = alt.get("words", [])
        if not words:
            continue

        start_s_str = words[0].get("startTime", "0s")
        end_s_str = words[-1].get("endTime", "0s")

        start_s = float(start_s_str.replace("s", ""))
        end_s = float(end_s_str.replace("s", ""))

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


def process_video_task(project_id: str, disk_url: str):
    """Фоновая задача: скачивание -> конвертация -> загрузка в S3 -> распознавание."""
    local_video_path = TEMP_DIR / f"{project_id}_video"
    local_audio_path = TEMP_DIR / f"{project_id}.opus"
    object_name = f"{project_id}.opus"

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

        # 3. ЗАГРУЗКА В S3
        projects_db[project_id]["status"] = ProjectStatusEnum.UPLOADING
        logger.info("[%s] Загрузка в S3...", project_id[:8])
        file_uri = upload_to_s3(str(local_audio_path), object_name)
        logger.info("[%s] Файл загружен в S3.", project_id[:8])

        # 4. РАСПОЗНАВАНИЕ
        projects_db[project_id]["status"] = ProjectStatusEnum.TRANSCRIBING
        logger.info("[%s] Отправка на распознавание...", project_id[:8])
        op_data = _transcribe_with_speechkit(project_id, file_uri)

        # 5. ОБРАБОТКА РЕЗУЛЬТАТА
        _process_recognition_result(project_id, op_data, original_filename, local_video_path)
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
        delete_from_s3(object_name)


def process_uploaded_file_task(project_id: str, local_video_path, original_filename: str):
    """Фоновая задача для локально загруженного файла: конвертация -> S3 -> распознавание."""
    local_audio_path = TEMP_DIR / f"{project_id}.opus"
    object_name = f"{project_id}.opus"
    # Преобразуем в Path если передана строка
    from pathlib import Path
    local_video_path = Path(local_video_path)

    try:
        projects_db[project_id]["original_filename"] = original_filename

        # 1. КОНВЕРТАЦИЯ
        projects_db[project_id]["status"] = ProjectStatusEnum.CONVERTING
        projects_db[project_id]["progress_percent"] = None
        _convert_to_opus(project_id, local_video_path, local_audio_path)

        # 2. ЗАГРУЗКА В S3
        projects_db[project_id]["status"] = ProjectStatusEnum.UPLOADING
        logger.info("[%s] Загрузка в S3...", project_id[:8])
        file_uri = upload_to_s3(str(local_audio_path), object_name)
        logger.info("[%s] Файл загружен в S3.", project_id[:8])

        # 3. РАСПОЗНАВАНИЕ
        projects_db[project_id]["status"] = ProjectStatusEnum.TRANSCRIBING
        logger.info("[%s] Отправка на распознавание...", project_id[:8])
        op_data = _transcribe_with_speechkit(project_id, file_uri)

        # 4. ОБРАБОТКА РЕЗУЛЬТАТА
        _process_recognition_result(project_id, op_data, original_filename, local_video_path)
        projects_db[project_id]["status"] = ProjectStatusEnum.COMPLETED
        logger.info("[%s] Файл обработан: %s", project_id[:8], original_filename)

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
        delete_from_s3(object_name)


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
