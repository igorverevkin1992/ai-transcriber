import { API_BASE_URL, API_KEY } from '../config';
import { ProjectData, MappingDecision, BatchStatus, Candidate } from '../types';

interface SpeakerDict {
  [key: string]: { duration_sec: number; suggested_name: string };
}

interface Segment {
  timecode: string;
  speaker: string;
  text: string;
}

interface ProjectResult {
  speakers: SpeakerDict;
  segments: Segment[];
  meta?: { original_filename?: string };
}

interface SpeakerVerification {
  id: string;
  name: string;
  abbr: string;
  duration_sec: number;
}

interface ProjectVerificationData {
  project_id: string;
  filename: string;
  speakers: SpeakerVerification[];
  preview_segments: Segment[];
  total_segments: number;
}

interface BatchVerificationResponse {
  projects: ProjectVerificationData[];
}

function authHeaders(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = { ...extra };
  if (API_KEY) h['X-API-Key'] = API_KEY;
  return h;
}

/** Разбирает JSON успешного ответа с понятной ошибкой вместо SyntaxError. */
async function parseJson<T>(response: Response): Promise<T> {
  try {
    return await response.json();
  } catch {
    throw new Error('Сервер вернул некорректный ответ (ожидался JSON)');
  }
}

export const api = {
  uploadFile: async (link: string): Promise<string> => {
    const response = await fetch(`${API_BASE_URL}/projects`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ url: link }),
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || `Ошибка при создании проекта: ${response.statusText}`);
    }

    const data = await parseJson<{ id: string }>(response);
    return data.id;
  },

  pollStatus: async (
    projectId: string,
    onProgress: (step: string, progress: number) => void,
    signal?: AbortSignal
  ): Promise<void> => {
    const POLL_INTERVAL_MS = 2000;

    while (true) {
      if (signal?.aborted) {
        throw new DOMException('Операция отменена', 'AbortError');
      }

      const response = await fetch(`${API_BASE_URL}/projects/${projectId}/status`, { signal, headers: authHeaders() });

      if (!response.ok) {
        throw new Error(`Ошибка проверки статуса: ${response.statusText}`);
      }

      const data = await parseJson<{ status: string; status_label: string; progress_percent?: number; error?: string }>(response);

      if (data.status === 'error') {
        throw new Error(data.error || 'Ошибка обработки на сервере');
      }

      if (data.status === 'completed') {
        return;
      }

      onProgress(data.status_label, data.progress_percent ?? 0);

      await new Promise<void>((resolve, reject) => {
        const onAbort = () => {
          clearTimeout(timeout);
          reject(new DOMException('Операция отменена', 'AbortError'));
        };
        const timeout = setTimeout(() => {
          signal?.removeEventListener('abort', onAbort);
          resolve();
        }, POLL_INTERVAL_MS);
        signal?.addEventListener('abort', onAbort, { once: true });
      });
    }
  },

  getVerificationData: async (projectId: string): Promise<ProjectData> => {
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}`, { headers: authHeaders() });
    if (!response.ok) throw new Error('Не удалось получить данные проекта');

    const result = await parseJson<ProjectResult>(response);

    const speakersDict: SpeakerDict = result.speakers || {};
    const totalDuration = Object.values(speakersDict).reduce(
      (acc, val) => acc + val.duration_sec, 0
    );

    const detected_speakers = Object.keys(speakersDict).map(tagId => {
      const s = speakersDict[tagId];
      return {
        tag_id: tagId,
        candidate_id: tagId,
        custom_name: s.suggested_name,
        total_duration_ms: s.duration_sec * 1000,
        percentage: totalDuration > 0 ? Math.round((s.duration_sec / totalDuration) * 100) : 0,
        is_tech: s.suggested_name === 'АЗК',
      };
    });

    const candidates: Candidate[] = detected_speakers.map(s => ({
      id: s.tag_id,
      name: s.custom_name || `Спикер ${s.tag_id}`,
      abbr: (s.custom_name || `S${s.tag_id}`).substring(0, 3).toUpperCase(),
    }));

    const preview_transcript = result.segments.map((seg: Segment) => ({
      timecode: seg.timecode,
      tag_id: seg.speaker,
      text: seg.text,
    }));

    return {
      id: projectId,
      original_filename: result.meta?.original_filename || 'video_source',
      duration_total_ms: totalDuration * 1000,
      candidates,
      detected_speakers,
      preview_transcript,
    };
  },

  confirmMapping: async (
    projectId: string,
    mapping: MappingDecision,
    editedSegments?: Array<{ timecode: string; speaker: string; text: string }>,
  ): Promise<Blob> => {
    const mappingList = Object.keys(mapping).map(tagId => ({
      speaker_label: tagId,
      mapped_name: mapping[tagId].name,
      abbreviation: mapping[tagId].abbreviation,
    }));

    const body: Record<string, unknown> = {
      mappings: mappingList,
      filename: 'transcript.docx',
    };
    if (editedSegments) {
      body.edited_segments = editedSegments;
    }

    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/export`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify(body),
    });

    if (!response.ok) throw new Error('Не удалось сгенерировать документ');
    return await response.blob();
  },

  // --- Batch methods ---

  batchUploadFile: async (file: File, engine: string = 'whisper', whisperModel: string = 'medium'): Promise<string> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('engine', engine);
    formData.append('whisper_model', whisperModel);

    const response = await fetch(`${API_BASE_URL}/batch/upload`, {
      method: 'POST',
      headers: authHeaders(),
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || `Ошибка загрузки файла: ${file.name}`);
    }

    const data = await parseJson<{ id: string }>(response);
    return data.id;
  },

  batchStatus: async (projectIds: string[]): Promise<BatchStatus> => {
    const response = await fetch(`${API_BASE_URL}/batch/status?ids=${encodeURIComponent(projectIds.join(','))}`, { headers: authHeaders() });
    if (!response.ok) throw new Error('Ошибка получения статуса пакета');
    return await parseJson<BatchStatus>(response);
  },

  resumeProject: async (projectId: string): Promise<void> => {
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/resume`, {
      method: 'POST',
      headers: authHeaders(),
    });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || 'Не удалось возобновить обработку');
    }
  },

  preloadWhisperModel: async (model: string = 'medium'): Promise<void> => {
    const formData = new FormData();
    formData.append('model', model);

    const response = await fetch(`${API_BASE_URL}/whisper/preload`, {
      method: 'POST',
      headers: authHeaders(),
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || 'Не удалось загрузить модель Whisper');
    }
  },

  batchDownload: async (projectIds: string[]): Promise<Blob> => {
    const response = await fetch(`${API_BASE_URL}/batch/download?ids=${encodeURIComponent(projectIds.join(','))}`, { headers: authHeaders() });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || 'Ошибка скачивания архива');
    }
    return await response.blob();
  },

  batchVerificationData: async (projectIds: string[]): Promise<BatchVerificationResponse> => {
    const response = await fetch(`${API_BASE_URL}/batch/verification-data?ids=${encodeURIComponent(projectIds.join(','))}`, { headers: authHeaders() });
    if (!response.ok) throw new Error('Ошибка получения данных верификации');
    return await parseJson<BatchVerificationResponse>(response);
  },

  batchExportWithMappings: async (projects: Array<{ project_id: string; mappings: Array<{ speaker_id: string; name: string; abbr: string }> }>): Promise<Blob> => {
    const response = await fetch(`${API_BASE_URL}/batch/export-with-mappings`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
      body: JSON.stringify({ projects }),
    });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || 'Ошибка экспорта');
    }
    return await response.blob();
  },

  batchDownloadSaved: async (): Promise<Blob> => {
    const response = await fetch(`${API_BASE_URL}/batch/download-saved`, { headers: authHeaders() });
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || 'Нет сохранённых файлов');
    }
    return await response.blob();
  },
};
