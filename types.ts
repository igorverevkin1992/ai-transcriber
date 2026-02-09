export type ProcessingStatus = 'IDLE' | 'PROCESSING' | 'VERIFICATION' | 'COMPLETED' | 'BATCH_UPLOADING' | 'BATCH_PROCESSING' | 'BATCH_DONE';

export interface ProcessingStep {
  label: string;
  status: 'pending' | 'loading' | 'completed';
}

// Data structures from PRD Section 6
export interface Candidate {
  id: string;
  name: string;
  abbr: string; // Abbreviation (e.g., "АД")
}

export interface SpeakerInfo {
  tag_id: string; // "1", "2", "3" from SpeechKit
  candidate_id: string | null; // Mapped candidate
  custom_name?: string; // If user overrides
  custom_abbr?: string;
  total_duration_ms: number;
  percentage: number;
  is_tech: boolean; // Is likely AZK/Tech
}

export interface TranscriptSegment {
  timecode: string; // HH:MM:SS:FF
  tag_id: string;
  text: string;
}

export interface ProjectData {
  id: string;
  original_filename: string;
  duration_total_ms: number;
  candidates: Candidate[];
  detected_speakers: SpeakerInfo[];
  preview_transcript: TranscriptSegment[];
}

export interface MappingDecision {
  [tag_id: string]: {
    name: string;
    abbreviation: string;
  }
}

// --- Batch types ---

export interface BatchFileInfo {
  id: string;
  filename: string;
  status: string;
  status_label: string;
  error?: string;
  progress_percent?: number;
}

export interface BatchStatus {
  total: number;
  completed: number;
  errors: number;
  in_progress: number;
  files: BatchFileInfo[];
}
