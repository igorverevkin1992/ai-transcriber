from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class ProjectStatusEnum(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    CONVERTING = "converting"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    ERROR = "error"


STATUS_LABELS_RU = {
    ProjectStatusEnum.QUEUED: "В очереди",
    ProjectStatusEnum.DOWNLOADING: "Скачивание",
    ProjectStatusEnum.CONVERTING: "Конвертация",
    ProjectStatusEnum.TRANSCRIBING: "Распознавание",
    ProjectStatusEnum.COMPLETED: "Готово",
    ProjectStatusEnum.ERROR: "Ошибка",
}


# --- Request models ---

class CreateProjectRequest(BaseModel):
    url: str


class SpeakerMapping(BaseModel):
    speaker_label: str
    mapped_name: str
    abbreviation: str = ""


class ExportRequest(BaseModel):
    mappings: List[SpeakerMapping]
    filename: str


# --- Response models ---

class CreateProjectResponse(BaseModel):
    id: str


class ProjectStatusResponse(BaseModel):
    status: str
    status_label: str
    error: Optional[str] = None
    progress_percent: Optional[int] = None


class SpeakerResult(BaseModel):
    duration_sec: float
    suggested_name: str


class MetaResult(BaseModel):
    speakers: List[str]
    start_tc: str
    original_filename: str


class SegmentResult(BaseModel):
    timecode: str
    speaker: str
    text: str


class ProjectResult(BaseModel):
    segments: List[SegmentResult]
    speakers: dict
    meta: MetaResult


class HealthResponse(BaseModel):
    status: str
    service: str
    message: str


# --- Batch models ---

class BatchFileStatus(BaseModel):
    id: str
    filename: str
    status: str
    status_label: str
    error: Optional[str] = None
    progress_percent: Optional[int] = None


class BatchStatusResponse(BaseModel):
    total: int
    completed: int
    errors: int
    in_progress: int
    files: List[BatchFileStatus]
