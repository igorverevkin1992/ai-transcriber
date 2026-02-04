import { API_BASE_URL } from '../config';
import { ProjectData, MappingDecision } from '../types';

export const api = {
  /**
   * Sends the file link to the backend to start processing.
   * Method: POST /projects
   * Body: { url: string } <-- Changed to 'url' to match Python Pydantic model
   */
  uploadFile: async (link: string): Promise<string> => {
    try {
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
    } catch (error) {
      console.error("API Error in uploadFile:", error);
      throw error;
    }
  },

  /**
   * Polls the backend status until the project is ready for verification.
   * Handles Russian status strings from Python backend.
   */
  pollStatus: async (projectId: string, onProgress: (step: string, progress: number) => void): Promise<void> => {
    const POLL_INTERVAL_MS = 2000;
    let isComplete = false;

    while (!isComplete) {
      try {
        const response = await fetch(`${API_BASE_URL}/projects/${projectId}/status`);
        
        if (!response.ok) {
           throw new Error(`Ошибка проверки статуса: ${response.statusText}`);
        }

        const data = await response.json();
        // Backend returns: { status: "В очереди" | "Скачивание" | "Готово" | "Ошибка", error: ... }
        
        if (data.status === 'Ошибка') {
          throw new Error(data.error || 'Ошибка обработки на сервере');
        }

        if (data.status === 'Готово') {
          isComplete = true;
          return;
        }

        // Map backend strings to UI progress
        onProgress(data.status, 0);

      } catch (error) {
        console.error("Polling error:", error);
        throw error;
      }

      if (!isComplete) {
        await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS));
      }
    }
  },

  /**
   * Fetches the structured project data and Adapts it to the Frontend ProjectData interface.
   */
  getVerificationData: async (projectId: string): Promise<ProjectData> => {
     try {
       const response = await fetch(`${API_BASE_URL}/projects/${projectId}`);
       if (!response.ok) throw new Error('Не удалось получить данные проекта');
       
       const result = await response.json();
       // Result format from Python:
       // { 
       //   segments: [{timecode, speaker, text}, ...], 
       //   speakers: {"1": {duration_sec: 10, suggested_name: "X"}, ...},
       //   meta: { original_filename: ... }
       // }

       // ADAPTER: Convert Python result to Frontend ProjectData
       
       // 1. Convert Dictionary of speakers to Array
       const speakersDict = result.speakers || {};
       const totalDuration = (Object.values(speakersDict) as any[]).reduce((acc: number, val: any) => acc + val.duration_sec, 0) as number;
       
       const detected_speakers = Object.keys(speakersDict).map(tagId => {
         const s = speakersDict[tagId];
         return {
           tag_id: tagId,
           candidate_id: tagId, // Initially map to self as a "candidate"
           custom_name: s.suggested_name,
           total_duration_ms: s.duration_sec * 1000,
           percentage: totalDuration > 0 ? Math.round((s.duration_sec / totalDuration) * 100) : 0,
           is_tech: s.suggested_name === 'АЗК' // Simple heuristics
         };
       });

       // 2. Synthesize Candidates based on suggestions (simplification for UI)
       const candidates = detected_speakers.map(s => ({
         id: s.tag_id,
         name: s.custom_name || `Speaker ${s.tag_id}`,
         abbr: (s.custom_name || `S${s.tag_id}`).substring(0, 3).toUpperCase()
       }));

       // 3. Map Segments
       const preview_transcript = result.segments.map((seg: any) => ({
         timecode: seg.timecode,
         tag_id: seg.speaker,
         text: seg.text
       }));

       return {
         id: projectId,
         original_filename: result.meta?.original_filename || "video_source",
         duration_total_ms: totalDuration * 1000,
         candidates,
         detected_speakers,
         preview_transcript
       };

     } catch (error) {
       console.error("API Error in getVerificationData:", error);
       throw error;
     }
  },

  /**
   * Sends the final mapping and triggers the DOCX generation.
   * Method: POST /projects/:id/export
   * Body: { mappings: [{speaker_label, mapped_name}], filename: string }
   */
  confirmMapping: async (projectId: string, mapping: MappingDecision): Promise<Blob> => {
    try {
      // ADAPTER: Convert Mapping map to List for Pydantic
      const mappingList = Object.keys(mapping).map(tagId => ({
        speaker_label: tagId,
        mapped_name: mapping[tagId].name
      }));

      const response = await fetch(`${API_BASE_URL}/projects/${projectId}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          mappings: mappingList,
          filename: "transcript.docx" 
        }),
      });
      
      if (!response.ok) throw new Error('Не удалось сгенерировать документ');
      return await response.blob();
    } catch (error) {
      console.error("API Error in confirmMapping:", error);
      throw error;
    }
  }
};