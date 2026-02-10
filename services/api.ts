import { API_BASE_URL } from '../config';
import { ProjectData, MappingDecision, BatchStatus } from '../types';

export const api = {
  uploadFile: async (link: string): Promise<string> => {
    const response = await fetch(`${API_BASE_URL}/projects`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: link }),
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || `Ошибка при создании проекта: ${response.statusText}`);
    }

    const data = await response.json();
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

      const response = await fetch(`${API_BASE_URL}/projects/${projectId}/status`, { signal });

      if (!response.ok) {
        throw new Error(`Ошибка проверки статуса: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.status === 'error') {
        throw new Error(data.error || 'Ошибка обработки на сервере');
      }

      if (data.status === 'completed') {
        return;
      }

      onProgress(data.status_label, data.progress_percent ?? 0);

      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(resolve, POLL_INTERVAL_MS);
        signal?.addEventListener('abort', () => {
          clearTimeout(timeout);
          reject(new DOMException('Операция отменена', 'AbortError'));
        }, { once: true });
      });
    }
  },

  getVerificationData: async (projectId: string): Promise<ProjectData> => {
    const response = await fetch(`${API_BASE_URL}/projects/${projectId}`);
    if (!response.ok) throw new Error('Не удалось получить данные проекта');

    const result = await response.json();

    const speakersDict = result.speakers || {};
    const totalDuration = (Object.values(speakersDict) as any[]).reduce(
      (acc: number, val: any) => acc + val.duration_sec, 0
    ) as number;

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

    const candidates = detected_speakers.map(s => ({
      id: s.tag_id,
      name: s.custom_name || `Speaker ${s.tag_id}`,
      abbr: (s.custom_name || `S${s.tag_id}`).substring(0, 3).toUpperCase(),
    }));

    const preview_transcript = result.segments.map((seg: any) => ({
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

  confirmMapping: async (projectId: string, mapping: MappingDecision): Promise<Blob> => {
    const mappingList = Object.keys(mapping).map(tagId => ({
      speaker_label: tagId,
      mapped_name: mapping[tagId].name,
      abbreviation: mapping[tagId].abbreviation,
    }));

    const response = await fetch(`${API_BASE_URL}/projects/${projectId}/export`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mappings: mappingList,
        filename: 'transcript.docx',
      }),
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
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || `Ошибка загрузки файла: ${file.name}`);
    }

    const data = await response.json();
    return data.id;
  },

  batchStatus: async (projectIds: string[]): Promise<BatchStatus> => {
    const response = await fetch(`${API_BASE_URL}/batch/status?ids=${projectIds.join(',')}`);
    if (!response.ok) throw new Error('Ошибка получения статуса пакета');
    return await response.json();
  },

  preloadWhisperModel: async (model: string = 'medium'): Promise<void> => {
    const formData = new FormData();
    formData.append('model', model);

    const response = await fetch(`${API_BASE_URL}/whisper/preload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || 'Не удалось загрузить модель Whisper');
    }
  },

  batchDownload: async (projectIds: string[]): Promise<Blob> => {
    const response = await fetch(`${API_BASE_URL}/batch/download?ids=${projectIds.join(',')}`);
    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || 'Ошибка скачивания архива');
    }
    return await response.blob();
  },
};
