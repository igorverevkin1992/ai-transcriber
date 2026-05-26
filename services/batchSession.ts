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

export function loadBatchSession(): BatchSession | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const session = JSON.parse(raw) as BatchSession;
    if (!Array.isArray(session.projectIds) || session.projectIds.length === 0) return null;
    if (Date.now() - session.startedAt > SESSION_TTL_MS) {
      clearBatchSession();
      return null;
    }
    return session;
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
