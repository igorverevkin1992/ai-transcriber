import os
import time
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.config import TEMP_DIR, logger
from backend.docx_export import generate_docx
from backend.models import (
    STATUS_LABELS_RU,
    CreateProjectRequest,
    CreateProjectResponse,
    ExportRequest,
    ProjectStatusEnum,
    ProjectStatusResponse,
)
from backend.services import process_video_task, projects_db
from backend.utils import validate_url

router = APIRouter(prefix="/api/v1")
limiter = Limiter(key_func=get_remote_address)


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
