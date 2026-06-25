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
- `TECH_BREAK_DOT` — trailing dot after the tech-break marker: `(Технические моменты).` (true, as in f7/f8 references) vs `(Технические моменты)` without dot (default: true). Applies to both the pause-based marker (`backend/turns.py`) and the Gemini content-based one.
- `WHISPER_BEAM_SIZE` / `WHISPER_INITIAL_PROMPT` — ASR accuracy tuning
- `STRICT_DIARIZATION` — fail the task with an actionable error when diarization can't run (no HF_TOKEN / pyannote load fails) instead of silently emitting a single-speaker draft (default: true)
- `DIARIZATION_MODEL` — pyannote diarization model passed to whisperx (default: `pyannote/speaker-diarization-3.1`). Newer whisperx otherwise defaults to the gated `pyannote/speaker-diarization-community-1`, which needs separate terms acceptance; accept terms for whichever model is set under the same account as `HF_TOKEN`
- `DIARIZATION_NUM_SPEAKERS` — exact speaker count hint for pyannote when known and the filename carries no guest names (without a count hint, pyannote on CPU tends to merge the interviewer's short questions into the guest's cluster). Priority: `sN` filename token (e.g. `17.06.2026_s3_f15.wmv`) → this env → number of names in the filename. When ≥2 from a token/env it is passed as `min_speakers == max_speakers` (exact); from the name count it stays the looser `n..n+1`. Default 0 (unset). Parsed in `parse_filename_metadata` / applied in `backend/services.py`.
- `TRANSCRIPT_GLOSSARY` — comma-separated correct spellings (names, terms, titles) injected into the Whisper prompt and Gemini correction prompt, e.g. `Мордюкова, Гурченко, Мосфильм, star quality, НТВ`
- `GLOSSARY_REPLACEMENTS` — deterministic `wrong=>right` pairs (comma-separated) applied **always** in `regex_cleanup`, independent of Gemini, so known recognition errors of rare names/terms are fixed even if Gemini is off/fails. Single words match on word boundaries; phrases tolerate variable whitespace; case-insensitive. e.g. `Мурдюкова=>Мордюкова,Горченко=>Гурченко,прияды=>плеяда,просто квашено=>Простоквашино`
- `INTERVIEWER_AUTODETECT` / `INTERVIEWER_LABEL` — auto-detect the off-camera interviewer (the speaker who alternates with every guest) and label it `АЗК`, excluding it from the speaker legend (default: true / `АЗК`). Tuning: `INTERVIEWER_LABEL_SINGLE_GUEST` (label the interviewer in 1-on-1 interviews too, default false), `INTERVIEWER_MIN_DISTINCT_GUESTS` (2), `INTERVIEWER_MAJORITY_RATIO` (0.5). When detected, filename guest names map to the remaining speakers by order of first appearance (not by duration). Logic lives in `backend/diarization_post.py`.
- `SPEAKER_NAME_AUTODETECT` — when the filename carries no guest names, auto-derive each guest's **name+patronymic** from how the interviewer addresses them in the dialogue («Олег Александрович, …») and replace the generic `Спикер N` label (default: true). Uses Gemini when `GEMINI_API_KEY` is set (`gemini_infer_speaker_names` in `backend/postprocess.py`), otherwise a deterministic vocative heuristic (`infer_speaker_names_by_vocative` in `backend/diarization_post.py`: a **direct comma-delimited address** «Имя Отчество, …» is attributed to the speaker who answers next; one such address is enough, but a name that is the top pick for more than one speaker is dropped as ambiguous). Bare third-person mentions («…а Галина Васильевна как опытный тренер…») are ignored, so the name doesn't leak to the wrong speaker. Filename names take precedence; only unnamed guests are filled. The interviewer's name isn't spoken in the audio, so they stay `АЗК`. Detected name+patronymic abbreviates to initials (`Олег Александрович`→`ОА`), matching the human reference; filename «Фамилия Имя» keeps the single-letter surname abbreviation.
- `SPEECHKIT_LITERATURE_TEXT` — SpeechKit literary mode (default: false, better for interviews)
- `GEMINI_MODEL` — Gemini model for text polishing (default: gemini-3.5-flash)
- `OCR_TIMECODE` — when no start timecode is found in the filename or container metadata, read the **burned-in** SMPTE timecode from the video frames via OCR (easyocr) and use it as the start timecode (default: true). Samples frames across the first ~15 s (1–10, 12, 15 s — robust to a black leader/slate) in a single ffmpeg pass, OCRs the timecode region (cropped + upscaled ×3), and back-computes the first-frame timecode, accepting it only when ≥2 frames **agree to the second** (output is second-precision anyway). Robust to OCR quirks: big digits split into separate tokens are re-joined, separators are optional (`16 : 39 : 57 : 11` and `16395711` both parse), and frame fields ≥ fps are rejected as misreads. If the configured region yields nothing, it retries OCR on the whole frame. Raw per-frame OCR text is logged (`[OCR ТК] кадр N: …`) for tuning, and a UI-visible warning is added when easyocr is missing or the TC can't be read. Priority order: filename token → container metadata → OCR → `00:00:00:00`. Logic in `backend/timecode_ocr.py`.
- `OCR_TIMECODE_REGION` — frame region to OCR, as `left,top,right,bottom` fractions 0-1 (default: `0.3,0.7,1.0,1.0` = bottom strip from center to right edge, where the wide timecode box usually sits). Empty string → scan the whole frame.
- `TECH_MOMENT_DETECTION` — conservatively detect non-interview crew talk (camera restarts, mic checks, on-set directions) via Gemini and replace it with the italic `(Технические моменты).` marker (default: true; needs `GEMINI_API_KEY`). Deliberately conservative — when in doubt the fragment is kept as interview text, since losing a real reply is worse than leaving stray crew chatter. This is content-based and independent of the pause-based marker that `backend/turns.py` still emits on gaps ≥ `TECH_BREAK_GAP_SECONDS`.
- `TECH_MOMENT_AGGRESSIVE` — opt-in stricter tech-moment detection that also flags off-camera crew/organizational chatter (addressing bystanders by name, «пойду умоюсь», «давай с тебя») in addition to pure technical talk. Default false (conservative — flagging real interview replies is worse). Needs `GEMINI_API_KEY`; uses a separate aggressive prompt in `backend/postprocess.py`.
- `LEGEND_DASH` / `LEGEND_TRAILING_DOT` — legend/section-divider style: dash between name and abbreviation (`–` en-dash default, as in f7/f8; set `—` for em-dash references) and whether a trailing dot is appended (default true). Built via `_legend_line` in `backend/docx_export.py`. Uppercase rendering is via `<w:caps/>`, independent of these.
- `SECTION_DIVIDERS_ENABLED` — insert per-guest legend dividers before each guest's first turn when >1 guest (default true). Set false for a single top legend block with no repeats.
- `DOCX_AUTHOR` — dc:creator / lastModifiedBy in generated DOCX (default: empty)
- `DOCX_TEMPLATE_PATH` — reference-derived DOCX template the generator builds on (default: `backend/transcript_template.docx`; regenerate via `python scripts/make_template.py`)
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
- **Filename convention** (`parse_filename_metadata`): encode guest names and start timecode in the filename, e.g. `Кравченко_Артемьева_Нилов_04.41.18.00_f21.wmv`. Names become diarization speaker-count hints, speaker-name suggestions, and Whisper-prompt context; the `HH.MM.SS.FF` token sets the start timecode (needed for `.wmv`, which carries no embedded SMPTE track). Per-turn timecodes and `(Технические моменты)` markers are emitted automatically once diarization separates speakers. With >1 guest, a section divider (the guest's legend line) is auto-inserted before each guest's first turn, approximating the per-interview blocks of the human reference (the transitional АЗК opener stays in the previous block, since the tape-cut boundary isn't in the audio). The document title is the source filename **without** extension.
