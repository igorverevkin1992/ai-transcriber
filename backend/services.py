import os
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ffmpeg
import grpc
import requests

from backend.config import (
    HALLUCINATION_BLACKLIST,
    HF_TOKEN,
    MAX_CONCURRENT_TASKS,
    MAX_FILE_SIZE_BYTES,
    NO_SPEECH_PROB_THRESHOLD,
    OUTPUT_DIR,
    SPEECHKIT_LITERATURE_TEXT,
    SQLITE_DB_PATH,
    TECH_BREAK_GAP_SECONDS,
    TEMP_DIR,
    TURN_INLINE_TC_SECONDS,
    TURN_MERGE_ENABLED,
    UNCLEAR_LOGPROB_THRESHOLD,
    WHISPER_BEAM_SIZE,
    WHISPER_INITIAL_PROMPT,
    YANDEX_API_KEY,
    logger,
)
from backend.metrics import (
    active_projects,
    diarization_speakers,
    project_errors,
    projects_total,
    transcribe_duration,
)
from backend.models import ProjectStatusEnum
from backend.store import ProjectStore
from backend.turns import UNCLEAR_TEXT, build_turns
from backend.utils import (
    detect_fps,
    detect_start_timecode,
    offset_tc,
    parse_filename_metadata,
    strip_extension,
    tc_to_frames,
    validate_file_extension,
)

# --- SQLite-backed project storage (in-memory cache + persistence) ---
projects_db = ProjectStore(db_path=SQLITE_DB_PATH)

_task_executor = ThreadPoolExecutor(
    max_workers=MAX_CONCURRENT_TASKS,
    thread_name_prefix="transcribe",
)
_executor_lock = threading.Lock()
_executor_alive = True


def _ensure_executor() -> ThreadPoolExecutor:
    """Return a live executor, recreating it if a previous shutdown closed it.

    Keeps the app (and tests that run the lifespan more than once) able to
    submit tasks after a shutdown instead of raising 'cannot schedule new
    futures after shutdown'.
    """
    global _task_executor, _executor_alive
    with _executor_lock:
        if not _executor_alive:
            _task_executor = ThreadPoolExecutor(
                max_workers=MAX_CONCURRENT_TASKS,
                thread_name_prefix="transcribe",
            )
            _executor_alive = True
        return _task_executor

PROJECT_TTL_SECONDS = 6 * 3600  # 6 hours

# --- Retry ---
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAYS = [10, 30, 60]  # seconds
_TASK_REGISTRY: dict[str, callable] = {}

# --- SpeechKit gRPC v3 ---
SPEECHKIT_GRPC_HOST = "stt.api.cloud.yandex.net:443"
GRPC_CHUNK_SIZE = 4000  # 4 KB chunks for streaming audio
GRPC_TIMEOUT = 7200  # 2 hours max for recognition

# --- WhisperX (transcription + diarization) / faster-whisper fallback ---
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    whisperx = None
    WHISPERX_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WhisperModel = None
    WHISPER_AVAILABLE = WHISPERX_AVAILABLE  # whisperx bundles faster-whisper

_whisper_model = None
_whisper_model_name = None
_whisper_lock = threading.Lock()
_whisperx_model = None
_whisperx_model_name = None
_whisperx_align_model = None
_whisperx_align_meta = None
_whisperx_diarize_pipeline = None


def _cleanup_gpu_memory():
    """Освобождает CUDA-кеш после транскрипции (если GPU доступна)."""
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def unload_whisper_models():
    """Полностью выгружает все Whisper/WhisperX модели из памяти."""
    global _whisper_model, _whisper_model_name
    global _whisperx_model, _whisperx_model_name
    global _whisperx_align_model, _whisperx_align_meta, _whisperx_diarize_pipeline
    with _whisper_lock:
        _whisper_model = None
        _whisper_model_name = None
        _whisperx_model = None
        _whisperx_model_name = None
        _whisperx_align_model = None
        _whisperx_align_meta = None
        _whisperx_diarize_pipeline = None
    _cleanup_gpu_memory()
    logger.info("Whisper/WhisperX модели выгружены из памяти")


def _detect_device_and_compute() -> tuple[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA доступна: %s", torch.cuda.get_device_name(0))
            return "cuda", "float16"
    except ImportError:
        pass
    return "cpu", "int8"


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _check_disk_space(required_bytes: int, target_dir: Path = None) -> None:
    """Проверяет, что в TEMP_DIR хватает места. Бросает RuntimeError если нет."""
    target = target_dir or TEMP_DIR
    free = shutil.disk_usage(str(target)).free
    if free < required_bytes:
        raise RuntimeError(
            f"Недостаточно места на диске: требуется {required_bytes / 1024**3:.1f} ГБ, "
            f"доступно {free / 1024**3:.1f} ГБ"
        )


def _download_from_yadisk(project_id: str, disk_url: str, local_video_path) -> str:
    """Скачивает файл с Яндекс.Диска. Возвращает оригинальное имя файла."""
    from backend.security import mask_url, validate_external_url

    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    resp = requests.get(api_url, params={"public_key": disk_url}, timeout=30, allow_redirects=False)
    if resp.status_code in (301, 302, 303, 307, 308):
        raise RuntimeError("Я.Диск API вернул редирект — отказано (SSRF protection)")
    resp.raise_for_status()
    download_url = resp.json()["href"]

    ssrf_err = validate_external_url(download_url)
    if ssrf_err:
        raise RuntimeError(f"SSRF protection: {ssrf_err}")

    original_filename = "video_source.mp4"
    try:
        meta_url = "https://cloud-api.yandex.net/v1/disk/public/resources"
        meta_resp = requests.get(meta_url, params={"public_key": disk_url}, timeout=15, allow_redirects=False)
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
    with requests.get(download_url, stream=True, timeout=600, allow_redirects=False) as r:
        if r.status_code in (301, 302, 303, 307, 308):
            raise RuntimeError("Download URL вернул редирект — отказано (SSRF protection)")
        r.raise_for_status()
        content_length = int(r.headers.get("content-length", 0))
        with open(local_video_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded_size += len(chunk)
                if downloaded_size > MAX_FILE_SIZE_BYTES:
                    raise ValueError(
                        f"Файл слишком большой (>{MAX_FILE_SIZE_BYTES / (1024**3):.0f} ГБ). Скачивание прервано."
                    )
                if content_length > 0:
                    pct = min(int(downloaded_size / content_length * 100), 100)
                    projects_db.update_field(project_id, "progress_percent", pct)

    logger.info("[%s] Файл скачан: %s (src=%s)", project_id[:8], original_filename, mask_url(disk_url))
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
                literature_text=SPEECHKIT_LITERATURE_TEXT,
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
                text = alt.text.strip()
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


# ==================== Whisper (local, free, via faster-whisper) ====================


def get_whisper_model(model_name: str = "medium"):
    """Загружает и кэширует модель faster-whisper (CTranslate2).

    Потокобезопасна (threading.Lock). Модели скачиваются автоматически
    через huggingface-hub. INT8 квантизация на CPU — ~4x быстрее openai-whisper.
    """
    global _whisper_model, _whisper_model_name
    if not WHISPER_AVAILABLE:
        raise RuntimeError(
            "faster-whisper не установлен. Выполните: pip install faster-whisper"
        )
    # Fast path: model already loaded
    if _whisper_model is not None and _whisper_model_name == model_name:
        return _whisper_model

    with _whisper_lock:
        # Double-check after acquiring lock
        if _whisper_model is not None and _whisper_model_name == model_name:
            return _whisper_model

        # Освобождаем старую модель перед загрузкой новой
        if _whisper_model is not None:
            logger.info("Выгрузка предыдущей модели '%s' перед загрузкой '%s'", _whisper_model_name, model_name)
            _whisper_model = None
            _cleanup_gpu_memory()

        device, compute_type = _detect_device_and_compute()
        logger.info("Загрузка модели faster-whisper '%s' (%s, %s)...", model_name, device, compute_type)
        try:
            _whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:
            logger.warning("%s/%s не поддерживается: %s. Пробуем cpu/float32...", device, compute_type, e)
            _whisper_model = WhisperModel(model_name, device="cpu", compute_type="float32")
        _whisper_model_name = model_name
        logger.info("Модель faster-whisper '%s' готова.", model_name)
        return _whisper_model


def _build_whisper_prompt(original_filename: str) -> str:
    """initial_prompt для Whisper: базовая подсказка + имена из имени файла.

    Подсказка задаёт декодеру контекст (язык, жанр, имена собственные),
    что снижает ошибки в редких именах и улучшает пунктуацию.
    """
    names = parse_filename_metadata(original_filename).get("speakers") or []
    if names:
        return f"{WHISPER_INITIAL_PROMPT} Участники: {', '.join(names)}."
    return WHISPER_INITIAL_PROMPT


_WHISPERX_SPEAKER_RE = re.compile(r"^SPEAKER_0*(\d+)$")

# Галлюцинации Whisper почти всегда короткие standalone-сегменты;
# длинная живая речь, где упомянуты «субтитры», не отбрасывается.
_HALLUCINATION_MAX_LEN = 100


def _is_hallucination(text: str, initial_prompt: str | None = None) -> bool:
    """True для известных Whisper-галлюцинаций на музыке/тишине и эха промпта."""
    lowered = text.strip().lower()
    if len(lowered) <= _HALLUCINATION_MAX_LEN:
        if any(phrase in lowered for phrase in HALLUCINATION_BLACKLIST):
            return True
    if initial_prompt and len(lowered) > 10:
        prompt_clean = initial_prompt.lower().strip(" .")
        text_clean = lowered.strip(" .")
        if text_clean in prompt_clean and len(text_clean) >= len(prompt_clean) * 0.4:
            return True
    return False


def _filter_hallucinated_segments(
    project_id: str, segments: list[dict], initial_prompt: str | None = None,
) -> list[dict]:
    """Отбрасывает сегменты-галлюцинации, логируя каждый."""
    kept = []
    for seg in segments:
        if _is_hallucination(seg["text"], initial_prompt):
            logger.info("[%s] Отброшена галлюцинация ASR: %r", project_id[:8], seg["text"][:80])
            continue
        kept.append(seg)
    return kept


def _resolve_whisper_text(text: str, avg_logprob: float | None, no_speech_prob: float | None) -> str | None:
    """Решает судьбу сегмента по уверенности ASR.

    None — отбросить (не речь); UNCLEAR_TEXT — человек пишет "(неразборчиво)"
    вместо выдуманного текста (цензус: 8 случаев в эталонах); иначе исходный текст.
    """
    if no_speech_prob is not None and no_speech_prob > NO_SPEECH_PROB_THRESHOLD:
        return None
    if avg_logprob is not None and avg_logprob < UNCLEAR_LOGPROB_THRESHOLD:
        return UNCLEAR_TEXT
    return text


def _normalize_speaker_id(raw: str) -> str:
    m = _WHISPERX_SPEAKER_RE.match(raw)
    return m.group(1) if m else raw


def _set_transcription_option(options, name: str, value):
    """Вернуть копию опций транскрипции с изменённым полем.

    В новых версиях faster-whisper `TranscriptionOptions` стал dataclass
    (нет метода `_replace`, как у NamedTuple). Поддерживаем оба варианта,
    а также прямую мутацию как последний фолбэк.
    """
    if hasattr(options, "_replace"):  # NamedTuple (старые версии)
        return options._replace(**{name: value})
    import dataclasses
    if dataclasses.is_dataclass(options):  # dataclass (новые версии)
        try:
            return dataclasses.replace(options, **{name: value})
        except (TypeError, ValueError):
            pass
    setattr(options, name, value)  # обычный объект / frozen-фолбэк
    return options


def _transcribe_with_whisperx(
    project_id: str,
    file_path,
    model_name: str = "medium",
    initial_prompt: str | None = None,
    original_filename: str | None = None,
) -> list[dict]:
    """Распознавание через WhisperX: транскрипция + alignment + диаризация pyannote.

    Возвращает сегменты с channel_tag = speaker label (нормализованный: "0", "1", ...).
    """
    global _whisperx_model, _whisperx_model_name
    global _whisperx_align_model, _whisperx_align_meta, _whisperx_diarize_pipeline

    file_path = Path(file_path)
    if not file_path.exists():
        raise RuntimeError(f"Файл не найден: {file_path}")
    if file_path.stat().st_size == 0:
        raise RuntimeError(f"Файл пустой (0 байт): {file_path}")

    device = _detect_device()
    file_size_mb = file_path.stat().st_size // (1024 * 1024)
    logger.info("[%s] WhisperX: файл %s (%d МБ, модель: %s, device: %s)...",
                project_id[:8], file_path.name, file_size_mb, model_name, device)

    compute_type = "float16" if device == "cuda" else "int8"
    with _whisper_lock:
        if _whisperx_model is None or _whisperx_model_name != model_name:
            logger.info("[%s] Загрузка WhisperX модели '%s'...", project_id[:8], model_name)
            _whisperx_model = whisperx.load_model(
                model_name, device, compute_type=compute_type, language="ru",
            )
            _whisperx_model_name = model_name
        if initial_prompt and hasattr(_whisperx_model, "options"):
            _whisperx_model.options = _set_transcription_option(
                _whisperx_model.options, "initial_prompt", initial_prompt
            )
        audio = whisperx.load_audio(str(file_path))
        result = _whisperx_model.transcribe(audio, batch_size=16 if device == "cuda" else 4, language="ru")
    logger.info("[%s] WhisperX: транскрипция завершена, %d сегментов", project_id[:8], len(result["segments"]))

    with _whisper_lock:
        if _whisperx_align_model is None:
            logger.info("[%s] Загрузка alignment-модели...", project_id[:8])
            _whisperx_align_model, _whisperx_align_meta = whisperx.load_align_model(
                language_code="ru", device=device,
            )
    result = whisperx.align(
        result["segments"], _whisperx_align_model, _whisperx_align_meta,
        audio, device, return_char_alignments=False,
    )
    logger.info("[%s] WhisperX: alignment завершён", project_id[:8])

    with _whisper_lock:
        if _whisperx_diarize_pipeline is None:
            hf_token = HF_TOKEN
            if not hf_token:
                logger.warning("[%s] HF_TOKEN не задан — диаризация недоступна, все сегменты = speaker 0", project_id[:8])
            else:
                logger.info("[%s] Загрузка diarization pipeline...", project_id[:8])
                _whisperx_diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

    if _whisperx_diarize_pipeline is not None:
        diarize_kwargs = {}
        if original_filename:
            meta = parse_filename_metadata(original_filename)
            num_speakers = len(meta.get("speakers", []))
            if num_speakers >= 2:
                diarize_kwargs["min_speakers"] = num_speakers
                diarize_kwargs["max_speakers"] = num_speakers + 1
                logger.info("[%s] Диаризация: хинт %d-%d спикеров из имени файла",
                            project_id[:8], num_speakers, num_speakers + 1)
        diarize_segments = _whisperx_diarize_pipeline(audio, **diarize_kwargs)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("[%s] WhisperX: диаризация завершена", project_id[:8])

    segments = []
    last_speaker = "0"
    for seg in result["segments"]:
        text = seg.get("text", "").strip()
        if not text:
            continue

        raw_speaker = seg.get("speaker")
        if raw_speaker:
            speaker = _normalize_speaker_id(raw_speaker)
            last_speaker = speaker
        else:
            # Диаризация пропустила сегмент (музыка, перекрытие голосов) —
            # наследуем спикера предыдущего сегмента, а не «нулевого».
            speaker = last_speaker

        words = []
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append({
                    "text": w.get("word", "").strip(),
                    "start_ms": int(w["start"] * 1000),
                    "end_ms": int(w["end"] * 1000),
                })

        start_ms = int(seg.get("start", 0) * 1000)
        end_ms = int(seg.get("end", 0) * 1000)

        segments.append({
            "text": text,
            "channel_tag": speaker,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "words": words if words else [{"text": text, "start_ms": start_ms, "end_ms": end_ms}],
        })

    segments = _filter_hallucinated_segments(project_id, segments, initial_prompt)
    logger.info("[%s] WhisperX: %d сегментов с диаризацией", project_id[:8], len(segments))
    _cleanup_gpu_memory()
    return segments


def _transcribe_with_whisper(
    project_id: str,
    file_path,
    model_name: str = "medium",
    initial_prompt: str | None = None,
) -> list[dict]:
    """Распознавание через faster-whisper (fallback без диаризации).

    Используется когда WhisperX недоступен.
    Все сегменты с channel_tag=0.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise RuntimeError(f"Файл не найден: {file_path}")
    if file_path.stat().st_size == 0:
        raise RuntimeError(f"Файл пустой (0 байт): {file_path}")

    model = get_whisper_model(model_name)
    file_size_mb = file_path.stat().st_size // (1024 * 1024)
    logger.info("[%s] faster-whisper: файл %s (%d МБ, модель: %s)...",
                project_id[:8], file_path.name, file_size_mb, model_name)

    raw_segments, info = model.transcribe(
        str(file_path),
        language="ru",
        word_timestamps=True,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True,
        initial_prompt=initial_prompt,
        # Антигаллюцинации: без условия на предыдущий текст (петли повторов
        # на длинных интервью) и с детектором галлюцинаций на тишине.
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )

    segments = []
    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue

        resolved = _resolve_whisper_text(
            text, getattr(seg, "avg_logprob", None), getattr(seg, "no_speech_prob", None),
        )
        if resolved is None:
            logger.info("[%s] Сегмент отброшен (no_speech_prob=%.2f): %r",
                        project_id[:8], seg.no_speech_prob, text[:60])
            continue
        if resolved == UNCLEAR_TEXT:
            logger.info("[%s] Сегмент неразборчив (avg_logprob=%.2f): %r",
                        project_id[:8], seg.avg_logprob, text[:60])
            text = resolved
            seg_words = []
        else:
            seg_words = seg.words or []

        words = []
        for w in seg_words:
            words.append({
                "text": w.word.strip(),
                "start_ms": int(w.start * 1000),
                "end_ms": int(w.end * 1000),
            })

        segments.append({
            "text": text,
            "channel_tag": 0,
            "start_ms": int(seg.start * 1000),
            "end_ms": int(seg.end * 1000),
            "words": words if words else [
                {"text": text, "start_ms": int(seg.start * 1000), "end_ms": int(seg.end * 1000)}
            ],
        })

    segments = _filter_hallucinated_segments(project_id, segments, initial_prompt)
    logger.info("[%s] faster-whisper: %d сегментов (аудио: %.0f сек)",
                project_id[:8], len(segments), info.duration)
    return segments


def _process_recognition_result(project_id: str, segments: list[dict], original_filename: str, video_path):
    """Обрабатывает результат распознавания v3 и сохраняет в projects_db."""
    meta = parse_filename_metadata(original_filename)
    projects_db.update_field(project_id, "original_filename", original_filename, persist=True)

    if video_path.exists():
        fps = detect_fps(str(video_path))
    else:
        fps = 25
        logger.warning("[%s] Видеофайл не найден для определения FPS, используется 25", project_id[:8])

    # Стартовый таймкод: явный TC в имени файла приоритетнее; иначе —
    # встроенный SMPTE-таймкод контейнера (эталоны начинаются именно с него).
    if meta["start_tc"] == "00:00:00:00" and video_path.exists():
        embedded_tc = detect_start_timecode(str(video_path))
        if embedded_tc:
            meta["start_tc"] = embedded_tc
            logger.info("[%s] Стартовый таймкод из контейнера: %s", project_id[:8], embedded_tc)
        else:
            logger.warning(
                "[%s] Стартовый таймкод не найден ни в имени файла, ни в контейнере — отсчёт с 00:00:00:00",
                project_id[:8],
            )

    speaker_durations: dict[str, float] = {}
    start_frames = tc_to_frames(meta["start_tc"], fps)

    # Длительности спикеров считаются по сырым ASR-сегментам (до склейки).
    events = []
    for seg in segments:
        channel = str(seg["channel_tag"])
        text = seg["text"]
        words = seg["words"]
        if not words or not text.strip():
            continue

        start_s = words[0]["start_ms"] / 1000.0
        end_s = words[-1]["end_ms"] / 1000.0
        speaker_durations[channel] = speaker_durations.get(channel, 0) + (end_s - start_s)
        events.append({
            "speaker": channel,
            "text": text.strip(),
            "start_s": start_s,
            "end_s": end_s,
        })

    if TURN_MERGE_ENABLED:
        raw_segments = build_turns(
            events, start_frames, fps,
            inline_tc_seconds=TURN_INLINE_TC_SECONDS,
            tech_break_gap_seconds=TECH_BREAK_GAP_SECONDS,
        )
    else:
        # Legacy: один ASR-сегмент = один абзац
        raw_segments = [
            {
                "timecode": offset_tc(start_frames, ev["start_s"], fps),
                "speaker": ev["speaker"],
                "text": ev["text"],
            }
            for ev in events
        ]

    detected_speakers = {}
    sorted_voices = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)
    file_names = meta["speakers"]

    if file_names and len(file_names) != len(sorted_voices):
        logger.warning(
            "[%s] Несовпадение: %d имён в файле, %d голосов обнаружено — маппинг может быть неточным",
            project_id[:8], len(file_names), len(sorted_voices),
        )

    for i, (voice_id, dur) in enumerate(sorted_voices):
        suggested = f"Спикер {int(voice_id) + 1}" if voice_id.isdigit() else f"Спикер {voice_id}"
        if i < len(file_names):
            suggested = file_names[i]

        detected_speakers[voice_id] = {
            "duration_sec": round(dur, 1),
            "suggested_name": suggested,
        }

    speaker_count = len(detected_speakers)
    diarization_speakers.observe(speaker_count)
    low_confidence = speaker_count < 2 or speaker_count > 8
    if speaker_count == 1:
        logger.warning(
            "[%s] Обнаружен только 1 спикер — возможно диаризация не сработала (проверьте HF_TOKEN)",
            project_id[:8],
        )
    elif low_confidence:
        logger.warning(
            "[%s] Подозрительное число спикеров (%d) — рекомендуется ручная проверка",
            project_id[:8], speaker_count,
        )

    projects_db.set_result(
        project_id,
        {
            "segments": raw_segments,
            "speakers": detected_speakers,
            "meta": {**meta, "original_filename": original_filename},
            "low_confidence_diarization": low_confidence,
        },
        fps=fps,
    )

    logger.info(
        "[%s] Обработка завершена. Сегментов: %d, Спикеров: %d, FPS: %d",
        project_id[:8], len(raw_segments), len(detected_speakers), fps,
    )


# ==================== Task functions ====================


_TERMINAL_STATUSES = {ProjectStatusEnum.COMPLETED.value, ProjectStatusEnum.ERROR.value}


def _cleanup_old_projects():
    """Удаляет завершённые/ошибочные проекты старше PROJECT_TTL_SECONDS."""
    deleted = projects_db.cleanup_old(PROJECT_TTL_SECONDS, _TERMINAL_STATUSES)
    if deleted:
        logger.info("TTL-очистка: удалено %d старых проектов", deleted)


def process_video_task(project_id: str, disk_url: str):
    """Фоновая задача: скачивание -> конвертация -> распознавание с диаризацией."""
    local_video_path = TEMP_DIR / f"{project_id}_video"
    local_audio_path = TEMP_DIR / f"{project_id}.opus"
    projects_total.labels(engine="speechkit").inc()
    active_projects.inc()
    task_start = time.time()

    try:
        _cleanup_old_projects()

        # 1. СКАЧИВАНИЕ
        projects_db.update_status(project_id, ProjectStatusEnum.DOWNLOADING)
        projects_db.update_field(project_id, "progress_percent", 0)
        logger.info("[%s] Скачивание файла с Яндекс.Диска...", project_id[:8])
        original_filename = _download_from_yadisk(project_id, disk_url, local_video_path)

        # 2. КОНВЕРТАЦИЯ
        projects_db.update_status(project_id, ProjectStatusEnum.CONVERTING)
        projects_db.update_field(project_id, "progress_percent", None)
        _convert_to_opus(project_id, local_video_path, local_audio_path)

        # 3. РАСПОЗНАВАНИЕ (gRPC v3 с диаризацией)
        projects_db.update_status(project_id, ProjectStatusEnum.TRANSCRIBING)
        logger.info("[%s] Распознавание с диаризацией...", project_id[:8])
        segments = _transcribe_with_speechkit(project_id, local_audio_path)

        # 4. ПОСТОБРАБОТКА ТЕКСТА
        from backend.postprocess import postprocess_segments
        logger.info("[%s] Постобработка текста...", project_id[:8])
        segments = postprocess_segments(segments)

        # 5. ОБРАБОТКА РЕЗУЛЬТАТА
        _process_recognition_result(project_id, segments, original_filename, local_video_path)
        projects_db.update_status(project_id, ProjectStatusEnum.COMPLETED)
        transcribe_duration.labels(engine="speechkit").observe(time.time() - task_start)

    except Exception as e:
        project_errors.labels(stage="video_task").inc()
        logger.exception("[%s] Ошибка обработки: %s", project_id[:8], e)
        projects_db.update_status(project_id, ProjectStatusEnum.ERROR,
                                  error=str(e))

    finally:
        active_projects.dec()
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
    local_audio_path = TEMP_DIR / f"{project_id}.opus"
    local_video_path = Path(local_video_path)
    projects_total.labels(engine=engine).inc()
    active_projects.inc()
    task_start = time.time()

    try:
        _cleanup_old_projects()
        projects_db.update_field(project_id, "original_filename", original_filename, persist=True)

        if local_video_path.exists():
            _check_disk_space(local_video_path.stat().st_size * 2)

        if engine == "whisper":
            projects_db.update_status(project_id, ProjectStatusEnum.TRANSCRIBING)
            projects_db.update_field(project_id, "progress_percent", None)
            initial_prompt = _build_whisper_prompt(original_filename)
            if WHISPERX_AVAILABLE:
                logger.info("[%s] WhisperX (с диаризацией): модель %s", project_id[:8], whisper_model)
                segments = _transcribe_with_whisperx(
                    project_id, local_video_path, whisper_model,
                    initial_prompt=initial_prompt,
                    original_filename=original_filename,
                )
            else:
                logger.info("[%s] faster-whisper (без диаризации): модель %s", project_id[:8], whisper_model)
                segments = _transcribe_with_whisper(
                    project_id, local_video_path, whisper_model,
                    initial_prompt=initial_prompt,
                )
        else:
            projects_db.update_status(project_id, ProjectStatusEnum.CONVERTING)
            projects_db.update_field(project_id, "progress_percent", None)
            _convert_to_opus(project_id, local_video_path, local_audio_path)

            projects_db.update_status(project_id, ProjectStatusEnum.TRANSCRIBING)
            logger.info("[%s] SpeechKit v3 с диаризацией...", project_id[:8])
            segments = _transcribe_with_speechkit(project_id, local_audio_path)

        from backend.postprocess import postprocess_segments
        logger.info("[%s] Постобработка текста...", project_id[:8])
        segments = postprocess_segments(segments)

        _process_recognition_result(project_id, segments, original_filename, local_video_path)
        projects_db.update_status(project_id, ProjectStatusEnum.COMPLETED)
        transcribe_duration.labels(engine=engine).observe(time.time() - task_start)
        logger.info("[%s] Файл обработан: %s", project_id[:8], original_filename)

        # АВТОСОХРАНЕНИЕ DOCX НА ДИСК
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
        project_errors.labels(stage="upload_task").inc()
        logger.exception("[%s] Ошибка обработки: %s", project_id[:8], e)
        projects_db.update_status(project_id, ProjectStatusEnum.ERROR,
                                  error=str(e))

    finally:
        active_projects.dec()
        for path in (local_video_path, local_audio_path):
            try:
                if path.exists():
                    path.unlink()
            except OSError:
                pass


# ==================== Task registry & retry ====================

def _register_task(func):
    _TASK_REGISTRY[func.__name__] = func
    return func

process_video_task = _register_task(process_video_task)
process_uploaded_file_task = _register_task(process_uploaded_file_task)


def _maybe_retry(project_id: str):
    """Schedule a retry if the project failed and has retries left."""
    proj = projects_db.get(project_id)
    if not proj or proj.get("status") != ProjectStatusEnum.ERROR:
        return

    retry_count = proj.get("retry_count", 0)
    if retry_count >= MAX_RETRIES:
        logger.info("[%s] Достигнут лимит повторов (%d). Прекращаем.", project_id[:8], MAX_RETRIES)
        return

    task_func_name = proj.get("task_func")
    task_args = proj.get("task_args")
    if not task_func_name or task_func_name not in _TASK_REGISTRY or not task_args:
        return

    delay = RETRY_DELAYS[min(retry_count, len(RETRY_DELAYS) - 1)]
    retry_count += 1

    logger.info("[%s] Повтор %d/%d через %dс...", project_id[:8], retry_count, MAX_RETRIES, delay)

    projects_db.update_field(project_id, "retry_count", retry_count, persist=True)
    projects_db.update_status(project_id, ProjectStatusEnum.QUEUED)
    projects_db.update_field(project_id, "error", None)

    func = _TASK_REGISTRY[task_func_name]
    task_kwargs = proj.get("task_kwargs", {})

    # threading.Timer holds a single sleeping thread per retry that exits
    # right after firing — cheaper than a raw Thread doing time.sleep.
    timer = threading.Timer(
        delay, submit_task, args=(func, *task_args),
        kwargs={"project_id": project_id, **task_kwargs},
    )
    timer.daemon = True
    timer.start()


MIN_RAM_MB = int(os.getenv("MIN_RAM_MB", "500"))

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    _PSUTIL_AVAILABLE = False


def _log_ram(label: str):
    if not _PSUTIL_AVAILABLE:
        return
    mem = psutil.virtual_memory()
    avail_mb = mem.available // (1024 * 1024)
    used_pct = mem.percent
    logger.info("RAM [%s]: %d МБ свободно (%.0f%% занято)", label, avail_mb, used_pct)
    if avail_mb < MIN_RAM_MB:
        logger.warning("RAM КРИТИЧНО: осталось %d МБ (порог: %d МБ)", avail_mb, MIN_RAM_MB)


def submit_task(func, *args, project_id: str | None = None, **kwargs):
    """Submit a heavy processing task to the dedicated executor.

    The executor has max_workers=MAX_CONCURRENT_TASKS, so only N tasks run
    at a time. Extra tasks wait in an internal queue without blocking any threads.
    """
    def _wrapper():
        _log_ram(f"начало {func.__name__}")
        try:
            func(*args, **kwargs)
        except Exception:
            logger.exception("Необработанная ошибка в фоновой задаче")
        finally:
            _log_ram(f"конец {func.__name__}")
            if project_id:
                _maybe_retry(project_id)

    _ensure_executor().submit(_wrapper)


def cancel_project(project_id: str) -> bool:
    """Отменяет/удаляет проект: убирает из БД, чистит temp/output файлы.

    Возвращает True если проект найден и удалён.
    """
    proj = projects_db.get(project_id)
    if not proj:
        return False

    # Best-effort cleanup of any temp/output files
    for prefix in (f"{project_id}_video", f"{project_id}.opus", f"transcript_{project_id}.docx",
                   f"batch_export_{project_id}.docx", f"batch_verified_{project_id}.docx",
                   f"autosave_{project_id}.docx"):
        for p in TEMP_DIR.glob(f"{prefix}*"):
            try:
                p.unlink()
            except OSError:
                pass

    # Remove autosaved DOCX from output dir if it matches project_id pattern
    for p in OUTPUT_DIR.glob(f"*_{project_id[:8]}.docx"):
        try:
            p.unlink()
        except OSError:
            pass

    projects_db.pop(project_id)
    logger.info("[%s] Проект отменён и удалён", project_id[:8])
    return True


def shutdown_executor():
    """Graceful shutdown: persist in-flight projects, then stop executor."""
    logger.info("Завершение фоновых задач...")
    in_flight = (
        ProjectStatusEnum.DOWNLOADING,
        ProjectStatusEnum.CONVERTING,
        ProjectStatusEnum.TRANSCRIBING,
    )
    for pid, proj in projects_db.items():
        if proj.get("status") in in_flight:
            projects_db.update_status(pid, ProjectStatusEnum.QUEUED)
            logger.info("[%s] Сохранён как QUEUED для восстановления", pid[:8])
    global _executor_alive
    with _executor_lock:
        _task_executor.shutdown(wait=False, cancel_futures=True)
        _executor_alive = False


def _invert_name(name: str) -> str:
    words = name.strip().split()
    if len(words) == 2 and all(w[0].isupper() and w.isalpha() for w in words):
        return f"{words[1]} {words[0]}"
    return name


def auto_export_project(project_id: str, output_path: str) -> str | None:
    """Автоматически экспортирует проект в DOCX используя имена спикеров из метаданных файла.
    Возвращает имя файла для скачивания или None при ошибке."""
    from backend.docx_export import generate_docx, is_legend_excluded_name

    proj = projects_db.get(project_id)
    if not proj or "result" not in proj:
        return None

    speakers = proj["result"].get("speakers", {})
    final_map = {}
    abbr_map = {}

    legend_exclude: set[str] = set()

    for speaker_id, info in speakers.items():
        name = info.get("suggested_name", f"Спикер {speaker_id}")
        final_map[speaker_id] = name
        if is_legend_excluded_name(name):
            legend_exclude.add(speaker_id)

    named_map = {sid: n for sid, n in final_map.items() if sid not in legend_exclude}
    abbr_map = _compute_smart_abbreviations(named_map)
    for sid in legend_exclude:
        abbr_map[sid] = final_map[sid]
    for sid in named_map:
        final_map[sid] = _invert_name(final_map[sid])
    return generate_docx(proj, final_map, abbr_map, output_path, legend_exclude=legend_exclude)


def _compute_smart_abbreviations(name_map: dict) -> dict:
    """Атомарно вычисляет аббревиатуры: первая буква фамилии, с индексацией коллизий.

    Двухфазный алгоритм:
    1. Группируем speaker_id → base_letter
    2. Для групп размера >1 — присваиваем индексы (М1, М2, ...)
    """
    base_to_sids: dict[str, list[str]] = {}
    fallback_map: dict[str, str] = {}

    for speaker_id, name in name_map.items():
        if not name or not name.strip():
            fallback_map[speaker_id] = f"С{speaker_id}"
            continue
        words = name.strip().split()
        first_word = words[0]
        if not first_word or not first_word[0].isalpha():
            fallback_map[speaker_id] = f"С{speaker_id}"
            continue
        base = first_word[0].upper()
        base_to_sids.setdefault(base, []).append(speaker_id)

    result = dict(fallback_map)
    for base, sids in base_to_sids.items():
        if len(sids) == 1:
            result[sids[0]] = base
        else:
            for i, sid in enumerate(sids, start=1):
                result[sid] = f"{base}{i}"
    return result
