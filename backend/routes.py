import os
import time
import uuid
import zipfile
from io import BytesIO
from typing import List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config import ALLOWED_EXTENSIONS, OUTPUT_DIR, TEMP_DIR, logger
from backend.docx_export import generate_docx
from backend.models import (
    STATUS_LABELS_RU,
    BatchFileStatus,
    BatchStatusResponse,
    CreateProjectRequest,
    CreateProjectResponse,
    ExportRequest,
    ProjectStatusEnum,
    ProjectStatusResponse,
)
from backend.services import (
    auto_export_project,
    process_uploaded_file_task,
    process_video_task,
    projects_db,
)
from backend.utils import validate_file_extension, validate_url

router = APIRouter(prefix="/api/v1")
limiter = Limiter(key_func=get_remote_address)

# In-memory хранилище пакетов
batches_db: dict = {}


@router.post("/projects", response_model=CreateProjectResponse)
@limiter.limit("5/minute")
async def create_project(request: Request, req: CreateProjectRequest, background_tasks: BackgroundTasks):
    """Создает проект и запускает фоновую обработку."""
    url_error = validate_url(req.url)
    if url_error:
        raise HTTPException(status_code=400, detail=url_error)

    pid = str(uuid.uuid4())
    projects_db[pid] = {
        "id": pid,
        "status": ProjectStatusEnum.QUEUED,
        "created_at": time.time(),
    }
    background_tasks.add_task(process_video_task, pid, req.url)
    logger.info("Проект создан: %s для URL: %s", pid[:8], req.url[:60])
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

    output_path = str(TEMP_DIR / f"transcript_{pid}.docx")
    download_name = generate_docx(proj, final_map, abbr_map, output_path)

    background_tasks.add_task(os.unlink, output_path)

    return FileResponse(
        path=output_path,
        filename=download_name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# ==================== BATCH ENDPOINTS ====================


@router.post("/batch/upload", response_model=CreateProjectResponse)
async def upload_file(file: UploadFile, background_tasks: BackgroundTasks):
    """Загружает один файл с локальной машины и запускает обработку."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Имя файла не указано")

    ext_error = validate_file_extension(file.filename)
    if ext_error:
        raise HTTPException(status_code=400, detail=ext_error)

    pid = str(uuid.uuid4())
    local_path = TEMP_DIR / f"{pid}_video"

    # Потоковая запись на диск
    async with aiofiles.open(str(local_path), "wb") as f:
        while chunk := await file.read(65536):
            await f.write(chunk)

    projects_db[pid] = {
        "id": pid,
        "status": ProjectStatusEnum.QUEUED,
        "created_at": time.time(),
        "original_filename": file.filename,
    }

    background_tasks.add_task(process_uploaded_file_task, pid, str(local_path), file.filename)
    logger.info("Файл загружен: %s -> проект %s", file.filename, pid[:8])
    return CreateProjectResponse(id=pid)


@router.get("/batch/status", response_model=BatchStatusResponse)
async def batch_status(ids: str = Query(..., description="ID проектов через запятую")):
    """Возвращает статус всех проектов в пакете."""
    project_ids = [i.strip() for i in ids.split(",") if i.strip()]
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
async def batch_download(ids: str = Query(..., description="ID проектов через запятую")):
    """Авто-экспортирует все завершённые проекты и возвращает ZIP-архив."""
    project_ids = [i.strip() for i in ids.split(",") if i.strip()]

    if not project_ids:
        raise HTTPException(status_code=400, detail="Не указаны ID проектов")

    zip_buffer = BytesIO()
    exported_count = 0

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
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
                try:
                    if os.path.exists(output_path):
                        os.unlink(output_path)
                except OSError:
                    pass

    if exported_count == 0:
        raise HTTPException(status_code=400, detail="Нет завершённых проектов для экспорта")

    zip_buffer.seek(0)
    logger.info("Пакетный экспорт: %d файлов в ZIP", exported_count)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=transcripts.zip"},
    )


@router.get("/batch/download-saved")
async def download_saved():
    """Скачивает все автосохранённые DOCX из папки completed_docx/ как ZIP."""
    docx_files = list(OUTPUT_DIR.glob("*.docx"))
    if not docx_files:
        raise HTTPException(status_code=400, detail="Нет сохранённых файлов в completed_docx/")

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in docx_files:
            zf.write(str(f), f.name)

    zip_buffer.seek(0)
    logger.info("Скачивание сохранённых файлов: %d DOCX", len(docx_files))

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=transcripts.zip"},
    )
