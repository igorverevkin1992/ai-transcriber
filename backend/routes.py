import asyncio
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config import MAX_FILE_SIZE_BYTES, OUTPUT_DIR, TEMP_DIR, logger
from backend.docx_export import generate_docx
from backend.models import (
    STATUS_LABELS_RU,
    BatchExportRequest,
    BatchFileStatus,
    BatchStatusResponse,
    CreateProjectRequest,
    CreateProjectResponse,
    ExportRequest,
    ProjectStatusEnum,
    ProjectStatusResponse,
)
from backend.security import validate_mime_type
from backend.services import (
    WHISPER_AVAILABLE,
    _compute_smart_abbreviations,
    auto_export_project,
    cancel_project,
    get_whisper_model,
    process_uploaded_file_task,
    process_video_task,
    projects_db,
    resume_project,
    submit_task,
)
from backend.utils import sanitize_filename, validate_file_extension, validate_url

router = APIRouter(prefix="/api/v1")
limiter = Limiter(key_func=get_remote_address)


def _safe_unlink(path: str) -> None:
    """Удаляет файл, игнорируя ошибки (для background-cleanup ZIP-архивов)."""
    try:
        os.unlink(path)
    except OSError:
        pass


def _compute_legend_exclude(name_map: dict[str, str]) -> set[str]:
    """Возвращает speaker_id тех спикеров, которых нужно исключить из легенды."""
    from backend.docx_export import is_legend_excluded_name
    return {sid for sid, name in name_map.items() if is_legend_excluded_name(name)}


@router.post("/projects", response_model=CreateProjectResponse)
@limiter.limit("5/minute")
async def create_project(request: Request, req: CreateProjectRequest):
    """Создает проект и запускает фоновую обработку."""
    url_error = validate_url(req.url)
    if url_error:
        raise HTTPException(status_code=400, detail=url_error)

    pid = str(uuid.uuid4())
    projects_db.create(pid, {
        "id": pid,
        "status": ProjectStatusEnum.QUEUED,
        "created_at": time.time(),
        "retry_count": 0,
        "task_func": "process_video_task",
        "task_args": (pid, req.url),
        "task_kwargs": {},
    })
    from backend.security import mask_url
    submit_task(process_video_task, pid, req.url, project_id=pid)
    logger.info("Проект создан: %s для URL: %s", pid[:8], mask_url(req.url))
    return CreateProjectResponse(id=pid)


@router.get("/projects/{pid}/status", response_model=ProjectStatusResponse)
async def get_status(pid: str):
    """Возвращает текущий статус обработки проекта."""
    if pid not in projects_db:
        raise HTTPException(status_code=404, detail="Проект не найден")
    proj = projects_db[pid]
    status = proj["status"]
    return ProjectStatusResponse(
        status=status.value,
        status_label=STATUS_LABELS_RU.get(status, status.value),
        error=proj.get("error"),
        progress_percent=proj.get("progress_percent"),
    )


@router.get("/projects/{pid}")
async def get_result(pid: str):
    """Возвращает результаты распознавания."""
    proj = projects_db.get(pid)
    if not proj:
        raise HTTPException(status_code=404, detail="Проект не найден")
    if proj["status"] != ProjectStatusEnum.COMPLETED:
        raise HTTPException(status_code=400, detail="Обработка ещё не завершена")
    return proj["result"]


@router.delete("/projects/{pid}")
async def delete_project(pid: str):
    """Отменяет/удаляет проект и связанные temp/output файлы."""
    if not cancel_project(pid):
        raise HTTPException(status_code=404, detail="Проект не найден")
    return {"status": "deleted", "id": pid}


@router.post("/projects/{pid}/resume")
@limiter.limit("30/minute")
async def resume_project_endpoint(request: Request, pid: str):
    """Вручную возобновляет прерванную задачу (после рестарта сервера)."""
    ok, message = resume_project(pid)
    if not ok:
        status_code = 404 if message == "Проект не найден" else 400
        raise HTTPException(status_code=status_code, detail=message)
    return {"status": "resumed", "id": pid, "message": message}


@router.post("/projects/{pid}/export")
async def export_docx(pid: str, req: ExportRequest, background_tasks: BackgroundTasks):
    """Генерирует DOCX-файл с транскриптом и отдает для скачивания."""
    proj = projects_db.get(pid)
    if not proj:
        raise HTTPException(status_code=404, detail="Проект не найден")
    if "result" not in proj:
        raise HTTPException(status_code=400, detail="Результаты обработки отсутствуют")

    final_map = {m.speaker_label: m.mapped_name for m in req.mappings}
    abbr_map = {m.speaker_label: m.abbreviation for m in req.mappings}

    legend_exclude = _compute_legend_exclude(final_map)
    segments_override = None
    if req.edited_segments:
        segments_override = [s.model_dump() for s in req.edited_segments if s.text.strip()]

    output_path = str(TEMP_DIR / f"transcript_{pid}.docx")
    download_name = generate_docx(
        proj, final_map, abbr_map, output_path,
        legend_exclude=legend_exclude,
        segments_override=segments_override,
    )

    background_tasks.add_task(_safe_unlink, output_path)

    return FileResponse(
        path=output_path,
        filename=download_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# ==================== BATCH ENDPOINTS ====================


@router.post("/batch/upload", response_model=CreateProjectResponse)
@limiter.limit("30/minute")
async def upload_file(
    request: Request,
    file: UploadFile,
    engine: str = Form("whisper"),
    whisper_model: str = Form("medium"),
):
    """Загружает один файл с локальной машины и запускает обработку.

    engine: 'whisper' (бесплатно, локально) или 'speechkit' (облако, диаризация)
    whisper_model: 'tiny', 'base', 'small', 'medium', 'large' (только для Whisper)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Имя файла не указано")

    valid_engines = {"whisper", "speechkit"}
    if engine not in valid_engines:
        raise HTTPException(status_code=400, detail=f"Неверный engine: {engine}. Допустимые: {', '.join(valid_engines)}")

    valid_models = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
    if whisper_model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Неверная модель: {whisper_model}. Допустимые: {', '.join(sorted(valid_models))}")

    safe_filename = sanitize_filename(file.filename)

    ext_error = validate_file_extension(safe_filename)
    if ext_error:
        raise HTTPException(status_code=400, detail=ext_error)

    pid = str(uuid.uuid4())
    # Preserve original extension — ffmpeg/Whisper need it to detect container format
    ext = Path(safe_filename).suffix  # e.g. ".wmv", ".mp4"
    local_path = TEMP_DIR / f"{pid}_video{ext}"

    # Потоковая запись на диск с проверкой размера и MIME-типа
    total_size = 0
    mime_validated = False
    first_chunk_buffer = b""
    async with aiofiles.open(str(local_path), "wb") as f:
        while chunk := await file.read(65536):
            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE_BYTES:
                await f.close()
                try:
                    local_path.unlink()
                except OSError:
                    pass
                raise HTTPException(
                    status_code=413,
                    detail=f"Файл слишком большой. Максимум: {MAX_FILE_SIZE_BYTES // (1024**3)} ГБ",
                )
            if not mime_validated:
                first_chunk_buffer += chunk
                if len(first_chunk_buffer) >= 8192:
                    mime_err = validate_mime_type(first_chunk_buffer[:8192])
                    if mime_err:
                        await f.close()
                        try:
                            local_path.unlink()
                        except OSError:
                            pass
                        raise HTTPException(status_code=400, detail=mime_err)
                    mime_validated = True
            await f.write(chunk)

    if not mime_validated and first_chunk_buffer:
        mime_err = validate_mime_type(first_chunk_buffer)
        if mime_err:
            try:
                local_path.unlink()
            except OSError:
                pass
            raise HTTPException(status_code=400, detail=mime_err)

    projects_db.create(pid, {
        "id": pid,
        "status": ProjectStatusEnum.QUEUED,
        "created_at": time.time(),
        "original_filename": safe_filename,
        "engine": engine,
        "retry_count": 0,
        "task_func": "process_uploaded_file_task",
        "task_args": (pid, str(local_path), safe_filename),
        "task_kwargs": {"engine": engine, "whisper_model": whisper_model},
    })

    submit_task(
        process_uploaded_file_task, pid, str(local_path), safe_filename,
        project_id=pid, engine=engine, whisper_model=whisper_model,
    )
    logger.info("Файл загружен: %s -> проект %s (engine=%s)", safe_filename, pid[:8], engine)
    return CreateProjectResponse(id=pid)


@router.get("/batch/status", response_model=BatchStatusResponse)
async def batch_status(ids: str = Query(..., description="ID проектов через запятую")):
    """Возвращает статус всех проектов в пакете."""
    project_ids = [i.strip() for i in ids.split(",") if i.strip()]
    if len(project_ids) > 200:
        raise HTTPException(status_code=400, detail="Максимум 200 проектов за запрос")
    files: List[BatchFileStatus] = []
    completed = 0
    errors = 0
    in_progress = 0

    for pid in project_ids:
        proj = projects_db.get(pid)
        if not proj:
            files.append(BatchFileStatus(
                id=pid, filename="???", status="not_found",
                status_label="Не найден", error="Проект не найден",
            ))
            errors += 1
            continue

        status = proj["status"]
        label = STATUS_LABELS_RU.get(status, status.value)

        if status == ProjectStatusEnum.COMPLETED:
            completed += 1
        elif status == ProjectStatusEnum.ERROR:
            errors += 1
        else:
            in_progress += 1

        files.append(BatchFileStatus(
            id=pid,
            filename=proj.get("original_filename", "???"),
            status=status.value,
            status_label=label,
            error=proj.get("error"),
            progress_percent=proj.get("progress_percent"),
        ))

    return BatchStatusResponse(
        total=len(project_ids),
        completed=completed,
        errors=errors,
        in_progress=in_progress,
        files=files,
    )


@router.get("/batch/download")
@limiter.limit("10/minute")
async def batch_download(
    request: Request,
    background_tasks: BackgroundTasks,
    ids: str = Query(..., description="ID проектов через запятую"),
):
    """Авто-экспортирует все завершённые проекты и возвращает ZIP-архив."""
    project_ids = [i.strip() for i in ids.split(",") if i.strip()]

    if not project_ids:
        raise HTTPException(status_code=400, detail="Не указаны ID проектов")
    if len(project_ids) > 200:
        raise HTTPException(status_code=400, detail="Максимум 200 проектов за запрос")

    # Stream the archive to a temp file instead of holding it all in memory
    zip_path = TEMP_DIR / f"batch_{uuid.uuid4().hex}.zip"
    exported_count = 0

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for pid in project_ids:
            proj = projects_db.get(pid)
            if not proj or proj.get("status") != ProjectStatusEnum.COMPLETED:
                continue

            output_path = str(TEMP_DIR / f"batch_export_{pid}.docx")
            try:
                download_name = auto_export_project(pid, output_path)
                if download_name and os.path.exists(output_path):
                    zf.write(output_path, download_name)
                    exported_count += 1
            finally:
                _safe_unlink(output_path)

    if exported_count == 0:
        try:
            zip_path.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Нет завершённых проектов для экспорта")

    logger.info("Пакетный экспорт: %d файлов в ZIP", exported_count)
    background_tasks.add_task(_safe_unlink, str(zip_path))
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename="transcripts.zip",
    )


@router.get("/batch/download-saved")
async def download_saved(background_tasks: BackgroundTasks):
    """Скачивает все автосохранённые DOCX из папки completed_docx/ как ZIP."""
    docx_files = list(OUTPUT_DIR.glob("*.docx"))
    if not docx_files:
        raise HTTPException(status_code=400, detail="Нет сохранённых файлов в completed_docx/")

    zip_path = TEMP_DIR / f"saved_{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for f in docx_files:
            zf.write(str(f), f.name)

    logger.info("Скачивание сохранённых файлов: %d DOCX", len(docx_files))
    background_tasks.add_task(_safe_unlink, str(zip_path))
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename="transcripts.zip",
    )


# ==================== BATCH VERIFICATION ====================


@router.get("/batch/verification-data")
async def batch_verification_data(ids: str = Query(..., description="ID проектов через запятую")):
    """Возвращает данные спикеров для каждого завершённого проекта в батче."""
    project_ids = [i.strip() for i in ids.split(",") if i.strip()]
    if len(project_ids) > 200:
        raise HTTPException(status_code=400, detail="Максимум 200 проектов за запрос")
    results = []

    for pid in project_ids:
        proj = projects_db.get(pid)
        if not proj or proj.get("status") != ProjectStatusEnum.COMPLETED or "result" not in proj:
            continue

        result = proj["result"]
        speakers = result.get("speakers", {})
        segments = result.get("segments", [])

        name_map = {sid: info.get("suggested_name", f"Спикер {sid}") for sid, info in speakers.items()}
        abbr_map = _compute_smart_abbreviations(name_map)

        speaker_list = []
        for speaker_id, info in speakers.items():
            name = name_map[speaker_id]
            abbr = abbr_map.get(speaker_id, f"С{speaker_id}")
            speaker_list.append({
                "id": speaker_id,
                "name": name,
                "abbr": abbr,
                "duration_sec": info.get("duration_sec", 0),
            })

        preview = segments[:5] if segments else []

        results.append({
            "project_id": pid,
            "filename": proj.get("original_filename", "???"),
            "speakers": speaker_list,
            "preview_segments": preview,
            "total_segments": len(segments),
            "warnings": result.get("warnings", []),
        })

    return {"projects": results}


@router.post("/batch/export-with-mappings")
@limiter.limit("10/minute")
async def batch_export_with_mappings(
    request: Request, body: BatchExportRequest, background_tasks: BackgroundTasks
):
    """Экспортирует все проекты с пользовательскими маппингами спикеров."""
    project_mappings = body.projects

    if not project_mappings:
        raise HTTPException(status_code=400, detail="Не указаны проекты")

    zip_path = TEMP_DIR / f"verified_{uuid.uuid4().hex}.zip"
    exported_count = 0

    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
        for pm in project_mappings:
            pid = pm.project_id
            mappings = pm.mappings

            proj = projects_db.get(pid)
            if not proj or proj.get("status") != ProjectStatusEnum.COMPLETED or "result" not in proj:
                continue

            final_map = {m.speaker_id: m.name for m in mappings}
            abbr_map = {m.speaker_id: m.abbr for m in mappings}
            legend_exclude = _compute_legend_exclude(final_map)

            output_path = str(TEMP_DIR / f"batch_verified_{pid}.docx")
            try:
                download_name = generate_docx(proj, final_map, abbr_map, output_path, legend_exclude=legend_exclude)
                if download_name and os.path.exists(output_path):
                    zf.write(output_path, download_name)
                    exported_count += 1
            finally:
                _safe_unlink(output_path)

    if exported_count == 0:
        try:
            zip_path.unlink()
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="Нет проектов для экспорта")

    logger.info("Верифицированный экспорт: %d файлов в ZIP", exported_count)
    background_tasks.add_task(_safe_unlink, str(zip_path))
    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename="transcripts.zip",
    )


# ==================== DIAGNOSTICS ====================


@router.get("/batch/queue-info")
async def queue_info():
    """Диагностика: показывает состояние всех проектов в памяти."""
    from collections import Counter
    status_counts = Counter()
    files_by_status: dict = {}

    for pid, proj in list(projects_db.items()):
        st = proj.get("status", "unknown")
        st_val = st.value if hasattr(st, "value") else str(st)
        status_counts[st_val] += 1

        if st_val not in files_by_status:
            files_by_status[st_val] = []
        files_by_status[st_val].append({
            "id": pid[:8],
            "filename": proj.get("original_filename", "???"),
            "error": (proj.get("error") or "")[:200],  # truncate long tracebacks
        })

    return {
        "total_projects": len(projects_db),
        "counts": dict(status_counts),
        "files": files_by_status,
    }


# ==================== WHISPER MODEL MANAGEMENT ====================


@router.post("/whisper/preload")
async def preload_whisper_model(
    model: str = Form("medium"),
):
    """Предварительно скачивает и загружает модель Whisper.

    Вызовите перед началом пакетной обработки, чтобы убедиться что модель
    скачана и готова к работе. Включает retry и очистку повреждённого кэша.
    """
    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="faster-whisper не установлен. Выполните: pip install faster-whisper",
        )

    valid_models = {"tiny", "base", "small", "medium", "large"}
    if model not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Неизвестная модель: {model}. Допустимые: {', '.join(sorted(valid_models))}",
        )

    try:
        await asyncio.to_thread(get_whisper_model, model)
        return {"status": "ok", "model": model, "message": f"Модель '{model}' готова к работе"}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
