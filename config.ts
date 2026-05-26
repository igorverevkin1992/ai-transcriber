// Base URL for the backend API.
// In dev mode, Vite proxies /api/* to http://localhost:8000 (see vite.config.ts),
// so we use a relative path by default. Override with VITE_API_BASE_URL if needed.
declare const import_meta_env: Record<string, string | undefined>;
export const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || '/api/v1';

// Optional API key for authenticated deployments.
// If the backend has API_KEY set, this must match or all /api/* calls get 401.
export const API_KEY: string | undefined = import.meta.env?.VITE_API_KEY;
