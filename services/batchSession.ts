const STORAGE_KEY = "abtgs_batch_session";
const SESSION_TTL_MS = 48 * 60 * 60 * 1000; // 48 hours

export interface BatchSession {
  projectIds: string[];
  fileNames: string[];
  engine: string;
  whisperModel: string;
  startedAt: number;
}

export function saveBatchSession(session: BatchSession): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  } catch {
    // quota exceeded or private browsing
  }
}

const VALID_ENGINES = ['whisper', 'speechkit'];
const VALID_MODELS = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'];

export function loadBatchSession(): BatchSession | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return null;
    const s = parsed as Record<string, unknown>;

    if (!Array.isArray(s.projectIds) || s.projectIds.length === 0) return null;
    if (!s.projectIds.every(id => typeof id === 'string')) return null;
    if (!Array.isArray(s.fileNames) || !s.fileNames.every(n => typeof n === 'string')) return null;
    if (typeof s.startedAt !== 'number') return null;
    if (typeof s.engine !== 'string' || !VALID_ENGINES.includes(s.engine)) return null;
    if (typeof s.whisperModel !== 'string' || !VALID_MODELS.includes(s.whisperModel)) return null;

    if (Date.now() - s.startedAt > SESSION_TTL_MS) {
      clearBatchSession();
      return null;
    }
    return {
      projectIds: s.projectIds as string[],
      fileNames: s.fileNames as string[],
      engine: s.engine,
      whisperModel: s.whisperModel,
      startedAt: s.startedAt,
    };
  } catch {
    return null;
  }
}

export function clearBatchSession(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}
