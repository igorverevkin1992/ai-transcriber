# ABTGS — Automated Broadcast Transcript Generation System

## Stack
- **Backend**: Python 3.11+ / FastAPI / SQLite (via `backend/store.py`)
- **Frontend**: React 18 / TypeScript / Vite / Tailwind CSS 4
- **Speech**: faster-whisper (local, INT8/CUDA) + Yandex SpeechKit v3 gRPC
- **Export**: python-docx with SMPTE timecodes
- **Deploy**: Docker multi-stage (Dockerfile) + docker-compose

## Project layout
```
main.py              # FastAPI app, lifespan, startup recovery
backend/
  config.py          # Env vars, paths, limits
  auth.py            # API key middleware
  store.py           # ProjectStore — SQLite + in-memory cache
  services.py        # Core: Whisper, SpeechKit, task executor, retry
  routes.py          # API endpoints (12 total)
  models.py          # Pydantic models, ProjectStatusEnum
  utils.py           # Filename parsing, timecode math, FPS detection
  docx_export.py     # DOCX generation with speaker mapping
App.tsx              # React root — view routing, batch state
components/          # BatchUploadForm, BatchProgress, VerificationDashboard, etc.
services/api.ts      # Fetch wrapper with auth headers
services/batchSession.ts  # localStorage persistence for batch recovery
tests/               # pytest: test_api, test_utils, test_store
```

## Running locally
```bash
# Backend
pip install -r requirements.txt
python main.py                    # http://localhost:8000

# Frontend
npm install && npm run dev        # http://localhost:3000 → proxies /api to :8000

# Docker
docker compose up --build         # production: serves frontend as static
docker compose --profile dev up   # dev: separate Vite dev server
```

## Key env vars
- `YANDEX_API_KEY` — SpeechKit (optional, only for cloud engine)
- `API_KEY` / `VITE_API_KEY` — API auth (optional, disabled if unset)
- `MAX_CONCURRENT_TASKS` — parallel transcriptions (default: 2)
- `MAX_RETRIES` — retry failed tasks (default: 3)
- `SQLITE_DB_PATH` — database location (default: projects.db)
- `MIN_RAM_MB` — low-RAM warning threshold (default: 500)
- `TURN_MERGE_ENABLED` — merge same-speaker segments into turn paragraphs (default: true)
- `TURN_INLINE_TC_SECONDS` / `TECH_BREAK_GAP_SECONDS` — inline timecode interval / tech-break gap (60 / 30)
- `WHISPER_BEAM_SIZE` / `WHISPER_INITIAL_PROMPT` — ASR accuracy tuning
- `SPEECHKIT_LITERATURE_TEXT` — SpeechKit literary mode (default: false, better for interviews)
- `GEMINI_MODEL` — Gemini model for text polishing (default: gemini-2.0-flash)
- `HALLUCINATION_BLACKLIST` — Whisper hallucination phrases to drop (replaces builtin list)
- `UNCLEAR_LOGPROB_THRESHOLD` / `NO_SPEECH_PROB_THRESHOLD` — ASR confidence gating (-1.2 / 0.85)

## Testing
```bash
pip install -r requirements-dev.txt   # pytest/httpx/ruff (not in prod image)
pytest tests/ -v
npx tsc --noEmit --skipLibCheck       # TS type check
```

## Architecture notes
- `ProjectStore` keeps all data in a dict for fast reads; SQLite persists on status changes and results. `progress_percent` is memory-only (too frequent for disk).
- `ThreadPoolExecutor(max_workers=N)` queues tasks; `submit_task()` auto-retries on failure.
- On server restart, `_recover_inflight_projects()` resubmits QUEUED/in-flight tasks from SQLite.
- Frontend saves batch projectIds to localStorage; on reload shows "Resume / Discard" banner.
