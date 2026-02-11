import hashlib
import shutil
import threading
import time
import urllib.error
import urllib.request
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

# Semaphore to limit concurrent heavy tasks (threading-based for sync background tasks)
_task_semaphore = threading.Semaphore(MAX_CONCURRENT_TASKS)

# TTL for completed projects (seconds) — cleaned up periodically
PROJECT_TTL_SECONDS = 6 * 3600  # 6 hours

# --- SpeechKit gRPC v3 ---
SPEECHKIT_GRPC_HOST = "stt.api.cloud.yandex.net:443"
GRPC_CHUNK_SIZE = 4000  # 4 KB chunks for streaming audio
GRPC_TIMEOUT = 7200  # 2 hours max for recognition

# --- Whisper (local) ---
try:
    import whisper as whisper_module
    WHISPER_AVAILABLE = True
except ImportError:
    whisper_module = None
    WHISPER_AVAILABLE = False

_whisper_model = None
_whisper_model_name = None
_whisper_lock = threading.Lock()


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


# ==================== Whisper (local, free) ====================


def _get_whisper_cache_dir() -> Path:
    """Возвращает директорию кэша Whisper (~/.cache/whisper)."""
    import os
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return Path(os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper"))


# Max retries for each chunk/resume attempt during download
DOWNLOAD_MAX_RETRIES = 10
DOWNLOAD_RETRY_DELAY = 5  # seconds between resume attempts
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB chunks


def _download_whisper_model_resumable(url: str, target_path: Path, expected_sha256: str):
    """Скачивает модель Whisper с поддержкой докачки (HTTP Range).

    При обрыве соединения продолжает с того же места, а не с нуля.
    После завершения проверяет SHA256.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".pt.downloading")

    for attempt in range(1, DOWNLOAD_MAX_RETRIES + 1):
        try:
            # Determine how much we already have
            downloaded = temp_path.stat().st_size if temp_path.exists() else 0

            # Build request with Range header for resume
            req = urllib.request.Request(url)
            if downloaded > 0:
                req.add_header("Range", f"bytes={downloaded}-")
                logger.info(
                    "Докачка модели с %d МБ (попытка %d/%d)...",
                    downloaded // (1024 * 1024), attempt, DOWNLOAD_MAX_RETRIES,
                )
            else:
                logger.info(
                    "Скачивание модели (попытка %d/%d)...",
                    attempt, DOWNLOAD_MAX_RETRIES,
                )

            with urllib.request.urlopen(req, timeout=30) as response:
                # Get total size
                if downloaded > 0 and response.status == 206:
                    # Partial content — resume worked
                    content_range = response.headers.get("Content-Range", "")
                    total_size = int(content_range.split("/")[-1]) if "/" in content_range else 0
                elif response.status == 200:
                    # Full content — server doesn't support Range or fresh start
                    total_size = int(response.headers.get("Content-Length", 0))
                    if downloaded > 0:
                        # Server sent full file, restart download
                        downloaded = 0
                        temp_path.unlink(missing_ok=True)
                else:
                    total_size = 0

                total_mb = total_size / (1024 * 1024) if total_size else 0

                mode = "ab" if downloaded > 0 and response.status == 206 else "wb"
                with open(temp_path, mode) as f:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = int(downloaded / total_size * 100)
                            dl_mb = downloaded / (1024 * 1024)
                            if pct % 10 == 0:
                                logger.info("  %d%% (%d / %d МБ)", pct, int(dl_mb), int(total_mb))

            # Download complete — verify SHA256
            logger.info("Скачивание завершено. Проверка SHA256...")
            sha256 = hashlib.sha256()
            with open(temp_path, "rb") as f:
                while True:
                    data = f.read(65536)
                    if not data:
                        break
                    sha256.update(data)

            if sha256.hexdigest() != expected_sha256:
                logger.error("SHA256 не совпадает! Удаляю файл и начинаю заново.")
                temp_path.unlink(missing_ok=True)
                if attempt < DOWNLOAD_MAX_RETRIES:
                    time.sleep(DOWNLOAD_RETRY_DELAY)
                    continue
                raise RuntimeError(
                    f"SHA256 не совпадает после {DOWNLOAD_MAX_RETRIES} попыток скачивания"
                )

            # SHA256 OK — move to final location
            shutil.move(str(temp_path), str(target_path))
            logger.info("Модель сохранена: %s", target_path)
            return

        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            logger.warning(
                "Ошибка сети при скачивании (попытка %d/%d): %s",
                attempt, DOWNLOAD_MAX_RETRIES, e,
            )
            if attempt < DOWNLOAD_MAX_RETRIES:
                logger.info("Повтор через %d сек (файл сохранён, докачка)...", DOWNLOAD_RETRY_DELAY)
                time.sleep(DOWNLOAD_RETRY_DELAY)
            else:
                raise RuntimeError(
                    f"Не удалось скачать модель после {DOWNLOAD_MAX_RETRIES} попыток: {e}"
                ) from e


def _ensure_whisper_model_downloaded(model_name: str) -> str:
    """Гарантирует что файл модели скачан и валиден. Возвращает путь к файлу."""
    # Get URL from whisper's model registry
    model_urls = getattr(whisper_module, "_MODELS", {})
    url = model_urls.get(model_name)
    if not url:
        # Fallback: let whisper handle it (unknown model or new version)
        return model_name

    # Extract expected SHA256 from URL (second-to-last path segment)
    expected_sha256 = url.split("/")[-2]
    cache_dir = _get_whisper_cache_dir()
    import os
    model_filename = os.path.basename(url)
    target_path = cache_dir / model_filename

    # Check if already downloaded and valid
    if target_path.exists():
        sha256 = hashlib.sha256()
        with open(target_path, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
        if sha256.hexdigest() == expected_sha256:
            logger.info("Модель '%s' уже скачана и проверена.", model_name)
            return model_name
        else:
            logger.warning("Модель '%s' повреждена, перекачиваю...", model_name)
            target_path.unlink(missing_ok=True)

    # Download with resume support
    _download_whisper_model_resumable(url, target_path, expected_sha256)
    return model_name


def get_whisper_model(model_name: str = "medium"):
    """Загружает и кэширует модель Whisper.

    Потокобезопасна (threading.Lock). Скачивает модель с поддержкой
    докачки (HTTP Range) — при обрыве продолжает с того же места.
    """
    global _whisper_model, _whisper_model_name
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "Whisper не установлен. Выполните: pip install openai-whisper"
        )
    # Fast path: model already loaded
    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model

    with _whisper_lock:
        # Double-check after acquiring lock
        if _whisper_model is not None and _whisper_model_name == model_name:
            return _whisper_model

        # Step 1: Ensure model file is downloaded (resumable)
        _ensure_whisper_model_downloaded(model_name)

        # Step 2: Load model into memory
        logger.info("Загрузка модели Whisper '%s' в память...", model_name)
        _whisper_model = whisper_module.load_model(model_name)
        _whisper_model_name = model_name
        logger.info("Модель Whisper '%s' готова.", model_name)
        return _whisper_model


def _transcribe_with_whisper(project_id: str, file_path, model_name: str = "medium") -> list[dict]:
    """Распознавание через Whisper (локально, бесплатно).

    Принимает любой аудио/видео файл (Whisper использует ffmpeg внутри).
    Конвертация в OPUS не нужна.
    Без диаризации — все сегменты с channel_tag=0.
    """
    model = get_whisper_model(model_name)
    logger.info("[%s] Whisper: распознавание (модель: %s)...", project_id[:8], model_name)

    result = model.transcribe(
        str(file_path),
        language="ru",
        word_timestamps=True,
        verbose=False,
    )

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text:
            continue

        words = []
        for w in seg.get("words", []):
            words.append({
                "text": w["word"].strip(),
                "start_ms": int(w["start"] * 1000),
                "end_ms": int(w["end"] * 1000),
            })

        segments.append({
            "text": text,
            "channel_tag": 0,
            "start_ms": int(seg["start"] * 1000),
            "end_ms": int(seg["end"] * 1000),
            "words": words if words else [
                {"text": text, "start_ms": int(seg["start"] * 1000), "end_ms": int(seg["end"] * 1000)}
            ],
        })

    logger.info("[%s] Whisper: %d сегментов", project_id[:8], len(segments))
    return segments


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


def _cleanup_old_projects():
    """Удаляет завершённые/ошибочные проекты старше PROJECT_TTL_SECONDS."""
    now = time.time()
    to_delete = []
    for pid, proj in projects_db.items():
        status = proj.get("status")
        if status in (ProjectStatusEnum.COMPLETED, ProjectStatusEnum.ERROR):
            created = proj.get("created_at", now)
            if now - created > PROJECT_TTL_SECONDS:
                to_delete.append(pid)
    for pid in to_delete:
        del projects_db[pid]
    if to_delete:
        logger.info("TTL-очистка: удалено %d старых проектов", len(to_delete))


def process_video_task(project_id: str, disk_url: str):
    """Фоновая задача: скачивание -> конвертация -> распознавание с диаризацией."""
    _task_semaphore.acquire()
    local_video_path = TEMP_DIR / f"{project_id}_video"
    local_audio_path = TEMP_DIR / f"{project_id}.opus"

    try:
        _cleanup_old_projects()

        # 1. СКАЧИВАНИЕ
        projects_db[project_id]["status"] = ProjectStatusEnum.DOWNLOADING
        projects_db[project_id]["progress_percent"] = 0
        logger.info("[%s] Скачивание файла с Яндекс.Диска...", project_id[:8])
        original_filename = _download_from_yadisk(project_id, disk_url, local_video_path)

        # 2. КОНВЕРТАЦИЯ
        projects_db[project_id]["status"] = ProjectStatusEnum.CONVERTING
        projects_db[project_id]["progress_percent"] = None
        _convert_to_opus(project_id, local_video_path, local_audio_path)

        # 3. РАСПОЗНАВАНИЕ (gRPC v3 с диаризацией)
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
        _task_semaphore.release()
        for path in (local_video_path, local_audio_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass


def process_uploaded_file_task(
    project_id: str,
    local_video_path,
    original_filename: str,
    engine: str = "speechkit",
    whisper_model: str = "medium",
):
    """Фоновая задача для локально загруженного файла.

    engine='whisper': файл -> Whisper (без конвертации, бесплатно)
    engine='speechkit': файл -> OPUS -> gRPC v3 (диаризация, платно)
    """
    _task_semaphore.acquire()
    local_audio_path = TEMP_DIR / f"{project_id}.opus"
    local_video_path = Path(local_video_path)

    try:
        _cleanup_old_projects()
        projects_db[project_id]["original_filename"] = original_filename

        if engine == "whisper":
            # Whisper: передаём файл напрямую (конвертация не нужна)
            projects_db[project_id]["status"] = ProjectStatusEnum.TRANSCRIBING
            projects_db[project_id]["progress_percent"] = None
            logger.info("[%s] Whisper: модель %s", project_id[:8], whisper_model)
            segments = _transcribe_with_whisper(project_id, local_video_path, whisper_model)
        else:
            # SpeechKit: конвертация в OPUS, затем gRPC
            projects_db[project_id]["status"] = ProjectStatusEnum.CONVERTING
            projects_db[project_id]["progress_percent"] = None
            _convert_to_opus(project_id, local_video_path, local_audio_path)

            projects_db[project_id]["status"] = ProjectStatusEnum.TRANSCRIBING
            logger.info("[%s] SpeechKit v3 с диаризацией...", project_id[:8])
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
        _task_semaphore.release()
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
